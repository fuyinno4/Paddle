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

#include "paddle/fluid/operators/controlflow/conditional_block_op.h"

namespace paddle {
namespace operators {

class ConditionalBlockOp : public ConditionalOp {
 public:
  ConditionalBlockOp(const std::string &type,
                     const framework::VariableNameMap &inputs,
                     const framework::VariableNameMap &outputs,
                     const framework::AttributeMap &attrs)
      : ConditionalOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    bool need_run;
    if (Attr<bool>("is_scalar_condition")) {
      // When is_scalar_condition is True, the conditional variable is a scalar,
      // whether need to execute the operators in sub-block depends on the
      // conditional variable (Cond).
      auto xs = InputTensors(scope, "Cond");
      need_run = ScalarCondition(xs);
    } else {
      // When is_scalar_condition is False, the conditional variable maybe a
      // vector or tensor, whether need to execute the operators in sub-block
      // depends on the input variables (Input).
      auto xs = InputTensors(scope, "Input");
      need_run = std::all_of(
          xs.begin(), xs.end(),
          [](const framework::LoDTensor *t) { return t->numel() != 0; });
    }

    if (need_run) {
      auto *scope_var = scope.FindVar(Output("Scope"));
      PADDLE_ENFORCE(scope_var != nullptr, "Must set scope");
      auto *scopes = scope_var->GetMutable<std::vector<framework::Scope *>>();
      scopes->resize(1);
      scopes->front() = &scope.NewScope();
      auto &cur_scope = *scopes->front();

      framework::Executor exec(dev_place);
      auto *block = Attr<framework::BlockDesc *>("sub_block");
      exec.Run(*block->Program(), &cur_scope, block->ID(), false);
    }
  }
};

class ConditionalBlockGradOp : public ConditionalOp {
 public:
  ConditionalBlockGradOp(const std::string &type,
                         const framework::VariableNameMap &inputs,
                         const framework::VariableNameMap &outputs,
                         const framework::AttributeMap &attrs)
      : ConditionalOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    bool need_run;
    if (Attr<bool>("is_scalar_condition")) {
      auto xs = this->InputTensors(scope, "Cond");
      need_run = ScalarCondition(xs);
    } else {
      auto xs = this->InputTensors(scope, "Input");
      need_run = std::all_of(
          xs.begin(), xs.end(),
          [](const framework::LoDTensor *t) { return t->numel() != 0; });
    }

    if (need_run) {
      auto *scope_var = scope.FindVar(Input("Scope"));
      PADDLE_ENFORCE(scope_var != nullptr, "Must set scope");
      auto &scopes = scope_var->Get<std::vector<framework::Scope *>>();
      framework::Scope &cur_scope = *scopes[0];

      framework::Executor exec(dev_place);
      auto *block = Attr<framework::BlockDesc *>("sub_block");

      const auto &ins = Inputs("Input");
      const auto &d_ins = Outputs(framework::GradVarName("Input"));
      const auto &conds = Inputs("Cond");
      const auto &d_conds = Outputs(framework::GradVarName("Cond"));

      std::vector<std::string> ins_conds_grads;
      ins_conds_grads.reserve(ins.size() + conds.size());
      for (auto &in : ins) {
        ins_conds_grads.emplace_back(framework::GradVarName(in));
      }
      for (auto &cond : conds) {
        ins_conds_grads.emplace_back(framework::GradVarName(cond));
      }

      exec.Run(*block->Program(), &cur_scope, block->ID(), false, true,
               ins_conds_grads);

      AssignLocalGradientToGlobal(dev_place, cur_scope, ins_conds_grads.data(),
                                  ins.size(), d_ins);

      AssignLocalGradientToGlobal(dev_place, cur_scope,
                                  ins_conds_grads.data() + ins.size(),
                                  conds.size(), d_conds);
    }
  }

 private:
  void AssignLocalGradientToGlobal(
      const platform::Place &place, const framework::Scope &cur_scope,
      const std::string *p_grad_names, size_t p_grad_names_num,
      const std::vector<std::string> &pg_names) const {
    for (size_t i = 0; i < p_grad_names_num; ++i) {
      auto out_grad_name = pg_names[i];
      const auto &in_grad_name = p_grad_names[i];
      auto *in_var = cur_scope.FindVar(in_grad_name);
      if (in_var == nullptr) {
        continue;
      }
      auto new_in_grad_name = cur_scope.Rename(in_grad_name);
      auto assign = framework::OpRegistry::CreateOp(
          "assign", {{"X", {new_in_grad_name}}}, {{"Out", {out_grad_name}}},
          framework::AttributeMap{});
      assign->Run(cur_scope, place);
      cur_scope.Rename(new_in_grad_name, in_grad_name);
    }
  }
};

class ConditionalBlockGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInputs("Cond"));
    if (context->HasInputs("Input")) {
      PADDLE_ENFORCE(context->HasOutputs(framework::GradVarName("Input")));
      context->SetOutputsDim(framework::GradVarName("Input"),
                             context->GetInputsDim("Input"));
    }
    if (context->HasOutputs(framework::GradVarName("Cond"))) {
      context->SetOutputsDim(framework::GradVarName("Cond"),
                             context->GetInputsDim("Cond"));
    }
  }
};

class ConditionalBlockGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto grad_op = new framework::OpDesc();
    grad_op->SetType("conditional_block_grad");
    grad_op->SetInput("Cond", Input("Cond"));
    grad_op->SetInput("Input", Input("Input"));
    grad_op->SetInput("Out", Output("Out"));
    grad_op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    grad_op->SetInput("Scope", Output("Scope"));
    grad_op->SetOutput(framework::GradVarName("Cond"),
                       InputGrad("Cond", false));
    grad_op->SetOutput(framework::GradVarName("Input"),
                       InputGrad("Input", false));
    grad_op->SetBlockAttr("sub_block", this->grad_block_[0]);
    grad_op->SetAttr("is_scalar_condition", GetAttr("is_scalar_condition"));
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(conditional_block, ops::ConditionalBlockOp,
                  ops::ConditionalBlockOpProtoMaker,
                  ops::ConditionalBlockGradMaker);
REGISTER_OPERATOR(conditional_block_grad, ops::ConditionalBlockGradOp,
                  ops::ConditionalBlockGradInferShape);
