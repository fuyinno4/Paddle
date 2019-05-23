// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/fleet/fleet_geo_wrapper.h"
#include "paddle/fluid/framework/data_feed.h"

namespace paddle {
namespace framework {

std::shared_ptr<FleetGeoWrapper> FleetGeoWrapper::s_instance_ = NULL;
bool FleetGeoWrapper::is_initialized_ = false;

#ifdef PADDLE_WITH_PSLIB
template<class AR>
paddle::ps::Archive<AR>& operator << (
    paddle::ps::Archive<AR>& ar,
    const MultiSlotType& ins) {
  ar << ins.GetType();
  ar << ins.GetOffset();
  ar << ins.GetFloatData();
  ar << ins.GetUint64Data();
return ar;
}

template<class AR>
paddle::ps::Archive<AR>& operator >> (
    paddle::ps::Archive<AR>& ar,
    MultiSlotType& ins) {
  ar >> ins.MutableType();
  ar >> ins.MutableOffset();
  ar >> ins.MutableFloatData();
  ar >> ins.MutableUint64Data();
return ar;
}
#endif

#ifdef PADDLE_WITH_PSLIB
std::shared_ptr<paddle::distributed::GeoPSlib> FleetGeoWrapper::pslib_ptr_ = NULL;
#endif

void FleetGeoWrapper::Init() {
#ifdef PADDLE_WITH_PSLIB
  if (pslib_ptr_ == NULL && !is_initialized_) {
    VLOG(3) << "Going to init fleet";
    pslib_ptr_ = std::shared_ptr<paddle::distributed::GeoPSlib>(
        new paddle::distributed::GeoPSlib());
    int ret = pslib_ptr_->init();
    if (ret != 0) {
        LOG(ERROR) << "Fail to init fleet";
    }
  } else {
    VLOG(3) << "Server can be initialized only once";
  }
#endif
}

void FleetGeoWrapper::InitServer() {
#ifdef PADDLE_WITH_PSLIB
  if (!is_initialized_) {
    VLOG(3) << "Going to init server";
    int ret = pslib_ptr_->init_server();
    if (ret != 0) {
        VLOG(3) << "ERROR: fail to init_server";
    }
    is_initialized_ = true;
  } else {
    VLOG(3) << "Server can be initialized only once";
  }
#endif
}

void FleetGeoWrapper::InitWorker() {
#ifdef PADDLE_WITH_PSLIB
  if (!is_initialized_) {
    VLOG(3) << "Going to init worker";
    pslib_ptr_->init_worker();
    is_initialized_ = true;
  } else {
    VLOG(3) << "Worker can be initialized only once";
  }
#endif
}

void FleetGeoWrapper::StopServer() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to stop server";
  pslib_ptr_->stop_server();
#endif
}

uint64_t FleetGeoWrapper::RunServer() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to run server";
  return pslib_ptr_->run_server();
#else
  return 0;
#endif
}

void FleetGeoWrapper::StopWorker() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to start server";
  pslib_ptr_->stop_worker();
#endif
}

uint64_t FleetGeoWrapper::RunWorker(int thread_num) {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to run worker";
  return pslib_ptr_->run_worker(thread_num);
#else
  return 0;
#endif
}

bool FleetGeoWrapper::IsServer() {
#ifdef PADDLE_WITH_PSLIB
    return pslib_ptr_->is_server();
#endif
}

bool FleetGeoWrapper::IsWorker() {
#ifdef PADDLE_WITH_PSLIB
    return pslib_ptr_->is_worker();
#endif
}

int FleetGeoWrapper::GetWorkerNum() {
#ifdef PADDLE_WITH_PSLIB
    return pslib_ptr_->get_worker_num();
#endif
}

int FleetGeoWrapper::GetWorkerIndex() {
#ifdef PADDLE_WITH_PSLIB
    return pslib_ptr_->get_worker_index();
#endif
}

bool FleetGeoWrapper::IsFirstWorker() {
#ifdef PADDLE_WITH_PSLIB
    return pslib_ptr_->is_first_worker();
#endif
}

bool FleetGeoWrapper::IsFirstProcOneMachine() {
#ifdef PADDLE_WITH_PSLIB
    return pslib_ptr_->is_first_proc_one_machine();
#endif
}

int FleetGeoWrapper::GetRankId() {
#ifdef PADDLE_WITH_PSLIB
    return pslib_ptr_->get_rankid();
#endif
}

void FleetGeoWrapper::BarrierAll() {
#ifdef PADDLE_WITH_PSLIB
    return pslib_ptr_->barrier_all();
#endif
}

void FleetGeoWrapper::BarrierWorker() {
#ifdef PADDLE_WITH_PSLIB
    return pslib_ptr_->barrier_worker();
#endif
}

void FleetGeoWrapper::InitModel(
    const Scope& scope,
    const std::vector<std::string>& sparse_var_names,
    const std::vector<std::string>& dense_var_names) {
  //TODO-zhihua: check order with AddFeatures
#ifdef PADDLE_WITH_PSLIB
  VLOG(0) << "Sparse param num: " << sparse_var_names.size();
  int i = 0;
  for (auto& t : sparse_var_names) {
    Variable* var = scope.FindVar(t);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* w = tensor->data<float>();
    auto dims = tensor->dims();
    if (pslib_ptr_->is_worker()) {
        VLOG(3) << "sparse" << i << ": " << w;
        ++i;
        pslib_ptr_->_worker_ptr->register_sparse_table(w, dims[0], dims[1]);
    } else {
        pslib_ptr_->_server_ptr->register_sparse_table(w, dims[0], dims[1]);
    }
  }

  VLOG(0) << "Dense param num: " << dense_var_names.size();
  i = 0;
  for (auto& t : dense_var_names) {
    Variable* var = scope.FindVar(t);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* w = tensor->data<float>();
    auto dims = tensor->dims();
    int count = 1;
    for (int j = 0; j < dims.size(); ++j) {
        count *= dims[j];
    }

    int col_num = dims[dims.size() - 1];
    int row_num = count / col_num;
    if (pslib_ptr_->is_worker()) {
        VLOG(3) << "dense" << i << ": " << w;
        ++i;
        pslib_ptr_->_worker_ptr->register_dense_table(w, row_num, col_num);
    } else {
        pslib_ptr_->_server_ptr->register_dense_table(w, row_num, col_num);
    }
  }
#endif
}

void FleetGeoWrapper::SetWorkerModel(
    const Scope& scope,
    const std::vector<std::string>& sparse_var_names,
    const std::vector<std::string>& dense_var_names) {
  //TODO-zhihua: check order with AddFeatures
#ifdef PADDLE_WITH_PSLIB
  if (!pslib_ptr_->is_worker()) {
      return;
  }
  VLOG(3) << "set sparse param num: " << sparse_var_names.size();
  for (size_t i = 0; i < sparse_var_names.size(); ++i) {
    Variable* var = scope.FindVar(sparse_var_names[i]);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* w = tensor->data<float>();
    VLOG(3) << "sparse" << i << ": " << w;
    //pslib_ptr_->_worker_ptr->set_sparse_table(i, w);
  }

  VLOG(3) << "set dense param num: " << dense_var_names.size();
  for (size_t i = 0; i < dense_var_names.size(); ++i) {
    Variable* var = scope.FindVar(dense_var_names[i]);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* w = tensor->data<float>();
    VLOG(3) << "dense" << i << ": " << w;
    //pslib_ptr_->_worker_ptr->set_dense_table(i, w);
  }
#endif
}

void FleetGeoWrapper::AddFeatures(int table_id, int thread_id, int fea_num, int* feas) {
#ifdef PADDLE_WITH_PSLIB
   //VLOG(3) << "FleetGeoWrapper::GeoAddFeatures";
   pslib_ptr_->_worker_ptr->add_sparse_keys(table_id, thread_id, fea_num, feas);
#endif
}

void FleetGeoWrapper::PushParams() {
#ifdef PADDLE_WITH_PSLIB
   //VLOG(3) << "FleetGeoWrapper::GeoPushParms";
   pslib_ptr_->_worker_ptr->ready_to_push(0);
#endif
}

bool FleetGeoWrapper::WaitPush() {
   bool ret = false;
#ifdef PADDLE_WITH_PSLIB
   //VLOG(3) << "FleetGeoWrapper::GeoWaitPush";
   ret = pslib_ptr_->_worker_ptr->wait_for_push();
#endif
   return ret;
}

template <typename T>
void FleetGeoWrapper::Serialize(const std::vector<T*>& t, std::string* str) {
#ifdef PADDLE_WITH_PSLIB
  paddle::ps::BinaryArchive ar;
  for (size_t i = 0; i < t.size(); ++i) {
    ar << *(t[i]);
  }
  *str = std::string(ar.buffer(), ar.length());
#else
  VLOG(0) << "FleetGeoWrapper::Serialize does nothing when no pslib";
#endif
}

template <typename T>
void FleetGeoWrapper::Deserialize(std::vector<T>* t, const std::string& str) {
#ifdef PADDLE_WITH_PSLIB
  if (str.length() == 0) {
    return;
  }
  paddle::ps::BinaryArchive ar;
  ar.set_read_buffer(const_cast<char*>(str.c_str()), str.length(), nullptr);
  if (ar.cursor() == ar.finish()) {
    return;
  }
  while (ar.cursor() < ar.finish()) {
    t->push_back(ar.get<T>());
  }
  CHECK(ar.cursor() == ar.finish());
  VLOG(3) << "Deserialize size " << t->size();
#else
  VLOG(0) << "FleetGeoWrapper::Deserialize does nothing when no pslib";
#endif
}

template void FleetGeoWrapper::Serialize<std::vector<MultiSlotType>>(
    const std::vector<std::vector<MultiSlotType>*>&, std::string*);
template void FleetGeoWrapper::Deserialize<std::vector<MultiSlotType>>(
    std::vector<std::vector<MultiSlotType>>*, const std::string&);

}  // end namespace framework
}  // end namespace paddle
