/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/platform/cpu_helper.h"

namespace paddle {
namespace framework {

void GeoWorker::Initialize(const TrainerDesc& desc) {
  param_ = desc.geo_param();
  comm_batch_ = param_.comm_batch();
  for (int i = 0; i < param_.sparse_table_size(); ++i) {
    int table_id = i;
    TableParameter table = param_.sparse_table(i);
    for (int j = 0; j < table.sparse_key_name_size(); ++j) {
      sparse_input_tableid_[table.sparse_key_name(j)] = table_id;
    }
  }

  fetch_config_ = desc.fetch_config(); 
  fleet_ptr_ = FleetGeoWrapper::GetInstance();
}

void GeoWorker::AddFeatures() {
	for (std::map<std::string, int>::iterator it = sparse_input_tableid_.begin();
		it != sparse_input_tableid_.end();
		++it) {
        int table_id = it->second;
		Variable * sc_var = thread_scope_->FindVar(it->first);
		LoDTensor * sc_tensor = sc_var->GetMutable<LoDTensor>();
		int64_t * ids = sc_tensor->data<int64_t>();
		int len = sc_tensor->lod()[0].back();
		ids_buffer_.resize(len);
		for (int j = 0; j < len; ++j) {
			ids_buffer_[j] = static_cast<int32_t>(ids[j]);
		}

		VLOG(3) << "add sparse keys: " << it->first << ", " << thread_id_ << ", " << len;
		fleet_ptr_->AddFeatures(table_id,
									thread_id_,
									len,
									ids_buffer_.data());
	}
}

void GeoWorker::TrainFilesWithProfiler() {
  VLOG(3) << "Begin to train files with profiler";
  platform::SetNumThreads(1);
  device_reader_->Start();
  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  for (auto& op : ops_) {
    op_name.push_back(op->Type());
  }

  VLOG(3) << "op name size: " << op_name.size();
  op_total_time.resize(op_name.size());
  for (size_t i = 0; i < op_total_time.size(); ++i) {
    op_total_time[i] = 0.0;
  }
  platform::Timer timeline;
  double total_time = 0.0;
  double read_time = 0.0;
  double comm_time = 0.0;
  double total_op_time = 0.0;
  int cur_batch;
  int batch_cnt = 0;
  uint64_t total_inst = 0;
  timeline.Start();
  while ((cur_batch = device_reader_->Next()) > 0) {
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();

    int run_op_idx = 0;
    for (auto& op : ops_) {
      timeline.Start();
      VLOG(3) << "Going to run op " << op_name[run_op_idx];
      op->Run(*thread_scope_, place_);
      VLOG(3) << "Op " << op_name[run_op_idx] << " Finished";
      timeline.Pause();
      op_total_time[run_op_idx++] += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      total_op_time += timeline.ElapsedSec();
    }

    timeline.Start();
    AddFeatures();
	if (thread_id_ == 0 && learned_batch_ != 0 && learned_batch_ >= comm_batch_ && \
		(!fleet_ptr_->WaitPush())) {
	    VLOG(0) << "thread" << thread_id_ << " need to push, learn " << learned_batch_;
		learned_batch_ = 0;
		fleet_ptr_->PushParams();
	}
    timeline.Pause();
    comm_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();

    PrintFetchVars();
    thread_scope_->DropKids();
    total_inst += cur_batch;
    ++batch_cnt;

    if (thread_id_ == 0) {
      // should be configured here
      if (batch_cnt > 0 && batch_cnt % 100 == 0) {
        for (size_t i = 0; i < op_total_time.size(); ++i) {
          fprintf(stderr, "op_name:[%zu][%s], op_mean_time:[%fs]\n", i,
                  op_name[i].c_str(), op_total_time[i] / batch_cnt);
        }
        fprintf(stderr, "mean read time: %fs\n", read_time / batch_cnt);
        fprintf(stderr, "comm time: %fs\n", comm_time / batch_cnt);
        fprintf(stderr, "op time: %fs\n", total_op_time / batch_cnt);
        fprintf(stderr, "IO percent: %f\n", read_time / total_time * 100);
        fprintf(stderr, "comm percent: %f\n", comm_time / total_time * 100);
        fprintf(stderr, "op percent: %f\n", total_op_time / total_time * 100);
        fprintf(stderr, "%6.2f instances/s\n", total_inst / total_time);
      }
    }
    timeline.Start();
  }
}

void GeoWorker::TrainFiles() {
  VLOG(3) << "Begin to train files";
  VLOG(3) << "Comm batch" << comm_batch_;
  platform::SetNumThreads(1);
  device_reader_->Start();
  int batch_cnt = 0;
  int cur_batch;
  learned_batch_ = 0;
  
  while ((cur_batch = device_reader_->Next()) > 0) {
    // do computation here
    for (auto& op : ops_) {
        op->Run(*thread_scope_, place_);
    }

    AddFeatures();
	if (thread_id_ == 0 && learned_batch_ != 0 && learned_batch_ >= comm_batch_ && \
		(!fleet_ptr_->WaitPush())) {
	    VLOG(0) << "thread" << thread_id_ << " need to push, learn " << learned_batch_;
		learned_batch_ = 0;
		fleet_ptr_->PushParams();
	}

    PrintFetchVars();
    thread_scope_->DropKids();
    ++batch_cnt;
    ++learned_batch_;
  }
}

}  // end namespace framework
}  // end namespace paddle
