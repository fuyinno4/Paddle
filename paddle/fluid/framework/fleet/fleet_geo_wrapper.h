/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#ifdef PADDLE_WITH_PSLIB
#include <pslib.h>
#include <archive.h>
#endif
#include <random>
#include <atomic>
#include <ctime>
#include <string>
#include <vector>
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {

// A wrapper class for pslib.h, this class follows Singleton pattern
// i.e. only initialized once in the current process
// Example:
//    std::shared_ptr<FleetGeoWrapper> fleet_ptr =
//         FleetGeoWrapper::GetInstance();
//    string dist_desc;
//    fleet_ptr->InitServer(dist_desc, 0);
// interface design principles:
// Pull
//   Sync: PullSparseVarsSync
//   Async: PullSparseVarsAsync(not implemented currently)
// Push
//   Sync: PushSparseVarsSync
//   Async: PushSparseVarsAsync(not implemented currently)
//   Async: PushSparseVarsWithLabelAsync(with special usage)
// Push dense variables to server in Async mode
// Param<in>: scope, table_id, var_names
// Param<out>: push_sparse_status

class FleetGeoWrapper {
 public:
  FleetGeoWrapper() {}
  ~FleetGeoWrapper() {}
  void Init();
  void InitServer();
  void InitWorker();
  void StopServer();
  uint64_t RunServer();
  void StopWorker();
  uint64_t RunWorker(int thread_num);
  void AddFeatures(int table_id, int thread_id, int fea_num, int* feas);
  void PushParams();
  bool WaitPush();
  void InitModel(
    const Scope& scope,
    const std::vector<std::string>& sparse_var_names,
    const std::vector<std::string>& dense_var_names);

  void SetWorkerModel(
    const Scope& scope,
    const std::vector<std::string>& sparse_var_names,
    const std::vector<std::string>& dense_var_names);

  bool IsServer();
  bool IsWorker();
  int GetWorkerNum();
  int GetWorkerIndex();
  bool IsFirstWorker(); 
  bool IsFirstProcOneMachine();
  int GetRankId();
  void BarrierAll();
  void BarrierWorker();

  template <typename T>
  void Serialize(const std::vector<T*>& t, std::string* str);
  template <typename T>
  void Deserialize(std::vector<T>* t, const std::string& str);
  static std::shared_ptr<FleetGeoWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::FleetGeoWrapper());
    }
    return s_instance_;
  }

#ifdef PADDLE_WITH_PSLIB
  static std::shared_ptr<paddle::distributed::GeoPSlib> pslib_ptr_;
#endif

 private:
  static std::shared_ptr<FleetGeoWrapper> s_instance_;

 protected:
  static bool is_initialized_;
  DISABLE_COPY_AND_ASSIGN(FleetGeoWrapper);
};

}  // end namespace framework
}  // end namespace paddle
