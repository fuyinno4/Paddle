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

import sys
import os
#from ..base.role_maker import MPISymetricRoleMaker
from .optimizer_factory import *
from google.protobuf import text_format
import paddle.fluid.optimizer as local_optimizer
import paddle.fluid as fluid


class GeoFleet(object):
    """
    Fleet in Python. Fleet is used in distributed training. It is designed as a singlton instance
    in c++. A Fleet() object will be initialized automatically when a user import this package as
    fleet. The General interface Fleet supports are:
    init(): which should be called only once in user's python scripts. init() will initialize
            FleetWrapper in CPP, it will also initialize a RoleMaker which is used for identifying
            current node's role, e.g. worker, server, etc.
    stop(): will be called after a user finishes his/her training task. Fleet instance will be
            destroyed when stop() is called.
    init_pserver(): will be called by user. When a user knows current process is_worker(), he/she
                    should call init_pserver() to initialize global information about parameter server
    init_worker(): will be called by user. When a user knows current process is_server(), he/she
                    should call init_worker() to initialize global information about worker and connect
                    worker with pserver.
    get_worker_num(): return the number of current task's worker node
    is_worker(): return whether current process is a worker
    is_server(): return thether current process is a server
    init_pserver_model(): initialize model parameters in pserver, called from a worker node
    save_pserver_model(): save model parameters in pserver, called from a server node

    Example:

        .. code-block:: python
           import paddle.fluid.incubate.fleet.parameter_server as fleet
           from my_model import bow_net
           model = bow_net()
           fleet.init()
           sgd_optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.0001)
           sgd_optimizer = fleet.DistributedOptimizer(sgd_optimizer)
           sgd_optimizer.minimize(model.loss)
           exe = paddle.fluid.Executor(paddle.fluid.CPUPlace())
           if fleet.is_worker():
              exe.run(paddle.fluid.default_startup_program())
              fleet.init_worker() # init worker should be called before training
              # do other things like training
           elif fleet.is_server():
              fleet.init_pserver()
           fleet.stop()
    """

    def __init__(self):
        self._opt_info = None  # for fleet only
        #self.role_maker_ = None
        self.local_ip_ = 0
        self.is_initialized_ = False

    def init(self):
        # TODO(guru4elephant)
        # this is a temporary solution
        # we will support more configurable RoleMaker for users in the future
        """
        init(): which should be called only once in user's python scripts. init() will initialize
            FleetWrapper in CPP, it will also initialize a RoleMaker which is used for identifying
            current node's role, e.g. worker, server, etc.
        """
        if not self.is_initialized_:
            #self.role_maker_ = MPISymetricRoleMaker()
            #self.role_maker_.generate_role()
            self._fleet_ptr = fluid.core.GeoFleet()
            self._fleet_ptr.init()
            self.is_initialized_ = True

    def stop(self):
        """
        stop(): will be called after a user finishes his/her training task. Fleet instance will be
            destroyed when stop() is called.
        """
        '''
        self.role_maker_.barrier_worker()
        if self.role_maker_.is_first_worker():
            self._fleet_ptr.stop_server()
        self.role_maker_.barrier_worker()
        self.role_maker_.barrier_all()
        self.role_maker_.finalize()
        '''
        self.barrier_all()
        
    def collect_var_names(self, scope):
        geo_param = self._opt_info["fleet_desc"].geo_trainer_param
        sparse_var_names = []
        for table in geo_param.sparse_table:
            sparse_var_names.extend(table.slot_value)
        dense_var_names = []
        for table in geo_param.dense_table:
            dense_var_names.extend(table.dense_variable_name)
        return sparse_var_names, dense_var_names

    def init_pserver(self, scope):
        """
        init_pserver(): will be called by user. When a user knows current process is_server(), he/she
            should call init_pserver() to initialize global information about parameter server
        """
        if self.is_initialized_:
            self._fleet_ptr.init_server()
            sparse_var_names, dense_var_names = self.collect_var_names(scope)
            #print sparse_var_names
            #print dense_var_names
            self._fleet_ptr.init_model(scope, sparse_var_names, dense_var_names)
        else:
            print("You should run fleet.init() first")
            sys.exit(-1)
        self.barrier_all()

    def init_worker(self, scope):
        """
        init_worker(): will be called by user. When a user knows current process is_worker(), he/she
                    should call init_worker() to initialize global information about worker and connect
                    worker with pserver.
        """
        if self.is_initialized_:
            self._fleet_ptr.init_worker()
            #TODO-zhihua: init from file
            sparse_var_names, dense_var_names = self.collect_var_names(scope)
            self._fleet_ptr.init_model(scope, sparse_var_names, dense_var_names)
        else:
            print("You should run fleet.init() first")
            sys.exit(-1)

        self.barrier_all()

    def run_pserver(self):
        """
        run_pserver(): will be called by user. When a user knows current process is_server(), he/she
            should call run_server to start parameter server
        """
        self._fleet_ptr.run_server()

    def run_worker(self):
        """
        run_worker(): will be called by user. When a user knows current process is_worker(), he/she
            should call run_server to start client
        """
        self._fleet_ptr.run_worker(self._opt_info["thread_num"])

    def stop_worker(self):
        """
        stop_worker(): will be called by user. When a user knows current process is_worker(), he/she
            should call stop_worker to stop client
        """
        self._fleet_ptr.stop_worker()
    
    def get_worker_num(self):
        """
        return the number of current job's worker num
        """
        return self._fleet_ptr.get_worker_num()

    def get_worker_index(self):
        """
        return the number of current job's worker num
        """
        return self._fleet_ptr.get_worker_index()

    def is_worker(self):
        """
        return whether current node is a worker
        """
        return self._fleet_ptr.is_worker()

    def is_server(self):
        """
        return whether current node is pserver
        """
        return self._fleet_ptr.is_server()

    def barrier_all(self):
        """
        barrier all process
        """
        self._fleet_ptr.barrier_all()
  
    def barrier_worker(self):
        """
        barrier all worker
        """
        self._fleet_ptr.barrier_worker()
 
    def init_pserver_model(self):
        """
        init pserver model called from pserver
        """
        return self._fleet_ptr.init_model()

    def save_pserver_model(self, save_path):
        """
        save pserver model called from a worker
        """
        return self._fleet_ptr.save_model(save_path)

    def _set_opt_info(self, opt_info):
        """
        this function saves the result from DistributedOptimizer.minimize()
        """
        self._opt_info = opt_info

class DistributedOptimizer(object):
    """
    DistributedOptimizer is a wrapper for paddle.fluid.optimizer
    A user should pass a paddle.fluid.optimizer to DistributedOptimizer
    minimize() function is implemented.
    DistributedOptimizer is the starting point for a user who wants to
    run distributed training. The optimized information will be stored in
    GeoFleet() instance who holds the global information about current distributed
    training.
    """

    def __init__(self, optimizer, dist_config={}):
        super(DistributedOptimizer, self).__init__()
        self._optimizer = optimizer
        '''
        self._optimizer_name = "Distributed%s" % optimizer.type.capitalize()
        if optimizer.type != "sgd":
            print("Currently, distributed optimizer only supports sgd"
                  "Will config built-in adam for you."
                  "We will support more functions in DistributedOptimizer",
                  sys.stderr)
            self._optimizer_name = "DistributedGeoSGD"
        '''
        self._optimizer_name = "DistributedGeoSGD"
        self._distributed_optimizer = globals()[self._optimizer_name](optimizer)

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        Currently, backward function can not be called through DistributedOptimizer
        """
        raise NotImplementedError()

    def apply_gradients(self, params_grads):
        """
        Currently, apply_gradients function can not be called through DistributedOptimizer
        """
        raise NotImplementedError()

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        minimize a program through loss, loss can be a list in DistributedOptimizer
        Args:
            loss (Variable|Variable List): loss variable or loss variable list to run optimization.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables to update.
            no_grad_set (set|None): set of Variables should be ignored.
        Returns:
            tuple: (optimize_ops, params_grads) which are, list of operators appended;
            and list of (param, grad) Variables pair for optimization.
        Note that in parameter server mode, a worker will not get anything about optimize_os
        Because optmizer algorithms run on pserver side. We will make this usable in pserver
        process, but currently the optimization part is written into Fleet(). A user does not
        need to care about how to startup a pserver node.
        """
        optimize_ops, param_grads, opt_info = \
                      self._distributed_optimizer.minimize(
                          loss,
                          startup_program,
                          parameter_list,
                          no_grad_set)

        fleet_instance._set_opt_info(opt_info)
        #fleet_instance.init()
        return [optimize_ops, param_grads]


# this is a temporary solution
# TODO(guru4elephant)
# will make this more flexible for more Parameter Server Archs
fleet_instance = GeoFleet()
#fleet_instance.init()

init = fleet_instance.init
stop = fleet_instance.stop
init_server = fleet_instance.init_pserver
init_worker = fleet_instance.init_worker
run_server = fleet_instance.run_pserver
run_worker = fleet_instance.run_worker
stop_worker = fleet_instance.stop_worker
is_worker = fleet_instance.is_worker
is_server = fleet_instance.is_server
init_server_model = fleet_instance.init_pserver_model
save_pserver_model = fleet_instance.save_pserver_model
worker_num = fleet_instance.get_worker_num
worker_index = fleet_instance.get_worker_index
barrier_all = fleet_instance.barrier_all
barrier_worker = fleet_instance.barrier_worker
