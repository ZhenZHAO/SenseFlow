import os
import torch
from torch.utils.data import Dataset
import json

from torch_geometric.data import HeteroData
import networkx as nx

class PowerFlowDataset(Dataset):
    def __init__(self, data_root, split_txt, pq_len, pv_len, slack_len, mask_num=0):
        self.data_root = data_root
        with open(split_txt, 'r') as f:
            self.file_list = [json.loads(line) for line in f]
        self.pq_len = pq_len
        self.pv_len = pv_len
        self.slack_len = slack_len
        self.mask_num = mask_num
        
        # for shortest path
        self.flag_distance_once_calculated = False
        self.shortest_paths = None
        self.node_type_to_global_index = None
        self.max_depth = 16

    def __len__(self):
        return len(self.file_list)
    
    def update_max_depth(self):
        tmp_distance =  max(list(self.shortest_paths.values()))
        if tmp_distance < self.max_depth:
            self.max_depth = tmp_distance

    def __getitem__(self, idx):
        file_dict = self.file_list[idx]
        data = torch.load(os.path.join(file_dict['file_path']))
        pq_num = data['PQ'].x.shape[0]
        pv_num = data['PV'].x.shape[0]
        slack_num = data['Slack'].x.shape[0]

        Vm, Va, P_net, Q_net, Gs, Bs = 0, 1, 2, 3, 4, 5

        # ------- add initial values --------
        # y = Vm, Va, P_net, Q_net
        data['PQ'].y = data['PQ'].x[:,[Vm, Va, P_net, Q_net]].clone().detach()
        data['PQ'].x[:, Vm] = 1.0  # Vm unknown
        data['PQ'].x[:, Va] = data['Slack'].x[0, Va].item() # Va unkonwn, uses value from Slack

        non_zero_indices = torch.nonzero(data['PQ'].x[:, Q_net])
        data['PQ'].q_mask = torch.ones((pq_num,),dtype=torch.bool)
        if self.mask_num > 0:
            if file_dict.get('masked_node') is None:
                mask_indices = non_zero_indices[torch.randperm(non_zero_indices.shape[0])[:self.mask_num]]
            else:
                mask_indices = file_dict['masked_node'][:self.mask_num]
            data['PQ'].q_mask[mask_indices] = False
            data['PQ'].x[~data['PQ'].q_mask, Q_net] = 0

        data['PV'].y = data['PV'].x[:,[Vm, Va, P_net, Q_net]].clone().detach()
        data['PV'].x[:, Va] = data['Slack'].x[0, Va].item()  # Va unkonwn, uses value from Slack
        data['PV'].x[:, Q_net] = 0  # Q unknown for PV node, set to 0

        data['Slack'].y = data['Slack'].x[:,[Vm, Va, P_net, Q_net]].clone().detach()
        data['Slack'].x[:, P_net] = 0  # P_net unkonwn for slack node
        data['Slack'].x[:, Q_net] = 0  # Q_net unknown for slack node

        return data
