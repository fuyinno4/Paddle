/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <vector>
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/fleet/fleet_geo_wrapper.h"

namespace paddle {
namespace framework {

void GeoDistMultiTrainer::Initialize(const TrainerDesc& trainer_desc,
                                  Dataset* dataset) {
  thread_num_ = trainer_desc.thread_num();
  param_ = trainer_desc.geo_param();
  SetDataset(dataset);
  workers_.resize(thread_num_);

  dataset->CreateReaders();
  const std::vector<std::shared_ptr<paddle::framework::DataFeed>> readers =
      dataset->GetReaders();

  for (int i = 0; i < thread_num_; ++i) {
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    workers_[i]->Initialize(trainer_desc);
    workers_[i]->SetDeviceIndex(i);
    workers_[i]->SetDataFeed(readers[i]);
  }
}

void GeoDistMultiTrainer::CollectVarNames() {
  for (int i = 0; i < param_.sparse_table_size(); ++i) {
    auto table = param_.sparse_table(i);
    for (int j = 0; j < table.sparse_value_name_size(); ++j) {
      sparse_var_names_.push_back(table.sparse_value_name(j));
    }
  }

  for (int i = 0; i < param_.dense_table_size(); ++i) {
    auto table = param_.dense_table(i);
    for (int j = 0; j < table.dense_value_name_size(); ++j) {
      dense_var_names_.push_back(table.dense_value_name(j));
    }
  }
}

void GeoDistMultiTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  VLOG(3) << "Init other env";
  CollectVarNames();
  if (fleet_ptr_->IsWorker()) {
      fleet_ptr_->SetWorkerModel(*root_scope_, sparse_var_names_, dense_var_names_);
  }

  VLOG(3) << "init other env done.";
}

void GeoDistMultiTrainer::Run() {
  VLOG(3) << "going to run, thread num: " << thread_num_;
  for (int thidx = 0; thidx < thread_num_; ++thidx) {
    if (!debug_) {
      VLOG(3) << "start thread" << thidx;
      threads_.push_back(
          std::thread(&DeviceWorker::TrainFiles, workers_[thidx].get()));
    } else {
      threads_.push_back(std::thread(&DeviceWorker::TrainFilesWithProfiler,
                                     workers_[thidx].get()));
    }
  }
}

void GeoDistMultiTrainer::Finalize() {
  for (auto& th : threads_) {
    th.join();
  }

  dataset_ptr_->DestroyReaders();
}

}  // end namespace framework
}  // end namespace paddle
