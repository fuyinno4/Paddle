#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import ps_pb2 as pslib


class Server(object):
    """
        A Server basic class.
    """

    def __init__(self):
        pass

class GeoServer(Server):
    """
        GeoSGDServer class is used to generate server program_desc
        Args:
            server: it is pslib.ServerParameter() 
        Examples:
            server = GeoSGDServer()
    """

    def __init__(self):
        self.server_ = pslib.GeoTrainerParameter()

    def add_sparse_table(self, table_id, slot_key_vars, slot_value_var):
        """
        Args:
            table_id(int): id of sparse params table
            slot_key_vars(list): slot key id list (one embedding with multiple input keys) 
            slot_value_var(string): one slot key value after embedding
        Returns:
            return None 
        """
        table = self.server_.sparse_table.add()
        table.table_id = table_id
        table.slot_key.extend(slot_key_vars)
        table.slot_value.extend([slot_value_var])
        return

    def add_dense_table(self, table_id, param_var):
        """
        Args:
            table_id(int): id of sparse params table
            param_var(string): one dense param
        Returns:
            return None 
        """
        table = self.server_.dense_table.add()
        table.table_id = table_id
        table.dense_variable_name.extend([param_var])
        return

    def get_desc(self):
        """
        Return downpour server program_desc
        """
        return self.server_
