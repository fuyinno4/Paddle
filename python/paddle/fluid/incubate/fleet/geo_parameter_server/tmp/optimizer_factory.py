#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["DistributedGeoSGD"]
import ps_pb2 as pslib
import paddle.fluid as fluid
from google.protobuf import text_format
from .node import GeoServer


class DistributedOptimizerImplBase(object):
    def __init__(self, optimizer):
        self.optimizer_ = optimizer
        self.learning_rate_ = optimizer._learning_rate
        self.regularization_ = optimizer.regularization

    def minimize(self,
                 losses,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        pass


class DistributedGeoSGD(DistributedOptimizerImplBase):
    def __init__(self, optimizer):
        # todo(guru4elephant): add more optimizers here as argument
        # todo(guru4elephant): make learning_rate as a variable
        super(DistributedGeoSGD, self).__init__(optimizer)
        self.type = "geosgd"

    def __find_lookup_table(self, program):
        """
        Find lookup table in program.
        We only support one distribute table now.
        Args:
        program(Program): given program, locate distributed lookup table
        Returns:
        table_name or None
        """
        input_name = []
        table_name = []
        output_name = []
        for op in program.global_block().ops:
            if op.type == "lookup_table":
                input_name.append(op.input("Ids")[0])
                table_name.append(op.input("W")[0])
                output_name.append(op.output("Out")[0])
        return input_name, table_name, output_name
    
    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        Currently, backward function from SGDOptimizer
        """
        print "###backward"
        self.optimizer_.backward(loss, startup_program, parameter_list, no_grad_set, callbacks)

    def apply_gradients(self, params_grads):
        """
        Currently, apply_gradients function from SGDOptimizer
        """
        print "###apply_gradients"
        self.optimizer_.apply_gradients(params_grads)

    def minimize(self,
                 losses,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        GeoSGD is a distributed optimizer so
        that user can call minimize to generate backward
        operators and optimization operators within minmize function
        Args:
            loss(Variable): loss variable defined by user
            startup_program(Program): startup program that defined by user
            parameter_list(str list): parameter names defined by users
            no_grad_set(set): a set of variables that is defined by users
            so that these variables do not need gradient computation
        Returns:
            [optimize_ops, grads_and_weights]
        """
        if not isinstance(losses, list):
            losses = [losses]

        #TODO-zhihua check multi-loss
        #print losses[0].block.program
        input_names, emb_names, output_names = self.__find_lookup_table(losses[0].block.program)
        #print "input:", input_names
        #print "emb:", emb_names
        #print "output:", output_names
        #embedding name, input keys
        table_dict = {}
        for i in range(0, len(input_names)):
            emb_name = emb_names[i]
            if emb_name not in table_dict:
                table_dict[emb_name] = []
            table_dict[emb_name].append(input_names[i])
        ps_param = pslib.PSParameter()
        server = GeoServer()
        sparse_table_index = 0
        for emb_name in table_dict:        
            print "add sparse table:", emb_name, ":", table_dict[emb_name]
            server.add_sparse_table(sparse_table_index, table_dict[emb_name], emb_name)
            sparse_table_index += 1

        dense_table_index = 0
        param_grads_list = []
        params = []

        #TODO-zhihua
        #for loss_index in range(len(losses)):
        loss_index = 0
        params_grads = sorted(
            fluid.backward.append_backward(losses[loss_index],
                                           parameter_list, no_grad_set),
            key=lambda x: x[0].name)
        optimize_ops = self.apply_gradients(params_grads) 
        param_grads_list.append(params_grads)
        for i in params_grads:
            param = i[0].name
            if param not in table_dict:
                params.append(param)

        for param in params:
            print "add dense table:", param
            server.add_dense_table(dense_table_index, param)
            dense_table_index += 1
        ps_param.geo_trainer_param.CopyFrom(server.get_desc())
        opt_info = {}
        opt_info["trainer"] = "GeoDistMultiTrainer"
        opt_info["device_worker"] = "GeoSGD"
        opt_info["optimizer"] = "GeoSGD"
        opt_info["fleet_desc"] = ps_param
        opt_info["thread_num"] = 0

        for loss in losses:
            loss.block.program._fleet_opt = opt_info
        
        #program = loss.block.program
        #with fluid.framework.program_guard(program, startup_program):
            #params_grads = self.backward(loss, startup_program,
            #                             parameter_list, no_grad_set)
            #optimize_ops = self.apply_gradients(params_grads)


        return optimize_ops, param_grads_list[0], opt_info
