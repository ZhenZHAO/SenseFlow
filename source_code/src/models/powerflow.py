import torch.nn as nn
import torch_geometric
from torch_geometric.nn import Linear, ResGatedGraphConv, HeteroConv, GATv2Conv, HGTConv, MLP
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from loguru import logger
from ..losses.deltapq_loss import deltapq_loss, create_Ybus
from scipy.sparse.csgraph import floyd_warshall
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention

# -------------------------- #
#     1. various modules     #
# -------------------------- #

class GLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout_ratio=0.1):
        # in A*2, hidden:A2, out:A
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(x) * v
        x = self.fc2(x)
        x = self.drop(x)
        return x

# -------------------------- #
#     2. various layers      #
# -------------------------- #

# ----- graph layers
class GraphLayer(torch.nn.Module):
    def __init__(self, 
                 emb_dim, 
                 edge_dim,
                 num_heads,
                 batch_size,
                 with_norm,
                 gcn_type="resgatedgraphconv",
                 act_layer=nn.ReLU):
        super().__init__()
        self.gcn_type = gcn_type
        self.flag_need_edge_attr = False
        conv_list = []
        from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GINConv, TransformerConv, GraphConv
        # https://github.com/pyg-team/pytorch_geometric/discussions/3479 gcn is not for hetero
        if gcn_type == 'gcnconv':
            conv_list = [GCNConv(emb_dim, emb_dim, add_self_loops=False) for _ in range(8)]
        if gcn_type == 'graphconv':
            conv_list = [GraphConv(emb_dim, emb_dim, add_self_loops=False) for _ in range(8)]
        elif gcn_type == 'sageconv':
            conv_list = [SAGEConv(emb_dim, emb_dim) for _ in range(8)]
        elif gcn_type == 'ginconv':
            conv_list = [GINConv(nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU())) for _ in range(8)]
        elif gcn_type == 'gatconv':
            conv_list = [GATv2Conv(emb_dim, emb_dim, heads=num_heads, edge_dim=edge_dim, add_self_loops=False, concat=False) for _ in range(8)]
            self.flag_need_edge_attr = True
        elif gcn_type == 'resgatedgraphconv':
            conv_list = [ResGatedGraphConv(emb_dim, emb_dim, edge_dim=edge_dim) for _ in range(8)]
            self.flag_need_edge_attr = True
        elif gcn_type == 'transformerconv':
            conv_list = [TransformerConv(emb_dim, emb_dim, heads=num_heads, edge_dim=edge_dim, concat=False) for _ in range(8)]
            self.flag_need_edge_attr = True
        else:
            raise ValueError(f"Unknown gcn_type: {gcn_type}")
        
        self.graph_layers = HeteroConv({
                    ('PQ', 'default', 'PQ'): conv_list[0],
                    ('PQ', 'default', 'PV'): conv_list[1],
                    ('PQ', 'default', 'Slack'): conv_list[2],
                    ('PV', 'default', 'PQ'): conv_list[3],
                    ('PV', 'default', 'PV'): conv_list[4],
                    ('PV', 'default', 'Slack'): conv_list[5],
                    ('Slack', 'default', 'PQ'): conv_list[6],
                    ('Slack', 'default', 'PV'): conv_list[7],
                }, 
                aggr='sum')
        self.act_layer = act_layer()
        self.norm = torch.nn.LayerNorm(emb_dim) if with_norm else nn.Identity()
        self.batch_size = batch_size


    def forward(self, batch: HeteroData):
        if self.flag_need_edge_attr:
            graph_x_dict = self.graph_layers(batch.x_dict, 
                                            batch.edge_index_dict,
                                            batch.edge_attr_dict)
        else:
            graph_x_dict = self.graph_layers(batch.x_dict, 
                                            batch.edge_index_dict)

        graph_x_dict = {key: self.act_layer(x) for key, x in graph_x_dict.items()}

        return graph_x_dict


# ----- ffn layers
class FFNLayer(torch.nn.Module):

    def __init__(self, 
                embed_dim_in: int,
                embed_dim_hid: int,
                embed_dim_out: int, 
                mlp_dropout: float, 
                with_norm: bool,
                act_layer=nn.GELU):
        super().__init__()

        # in: embed_dim_out, hidden: embed_dim_hid*2, out: embed_dim_out
        self.mlp = GLUFFN(in_features=embed_dim_in, 
                          hidden_features=embed_dim_hid, 
                          out_features=embed_dim_out,
                          act_layer=act_layer,
                          dropout_ratio=mlp_dropout)

        self.norm = torch.nn.LayerNorm(embed_dim_out) if with_norm else nn.Identity()

    def forward(self, x):
        x = x + self.mlp(x)
        return self.norm(x)

# -------------------------- #
#     3. building block      #
# -------------------------- #
class HybridBlock(nn.Module):
    def __init__(self, 
                 emb_dim_in, 
                 emb_dim_out, 
                 with_norm, 
                 edge_dim, 
                 batch_size,
                 gcn_type="resgatedgraphconv",
                 flag_use_ffn=True,
                 dropout_ratio=0.1,
                 heads_num=4):
        super(HybridBlock, self).__init__()
        self.emb_dim_in = emb_dim_in
        self.with_norm = with_norm
        self.branch_graph = GraphLayer(emb_dim=emb_dim_in,
                                       edge_dim=edge_dim, 
                                       num_heads=heads_num, 
                                       batch_size=batch_size,
                                       with_norm=with_norm, 
                                       gcn_type=gcn_type)

        # ---- mlp: activation + increase dimension
        self.flag_use_ffn = flag_use_ffn
        if self.flag_use_ffn:
            self.ffn = nn.ModuleDict()
            self.ffn['PQ'] = FFNLayer(embed_dim_in=emb_dim_in, embed_dim_hid=emb_dim_out,
                                        embed_dim_out=emb_dim_out,
                                        mlp_dropout=dropout_ratio, 
                                        with_norm=with_norm)
            self.ffn['PV'] = FFNLayer(embed_dim_in=emb_dim_in, embed_dim_hid=emb_dim_out,
                                        embed_dim_out=emb_dim_out,
                                        mlp_dropout=dropout_ratio, 
                                        with_norm=with_norm)
            self.ffn['Slack'] = FFNLayer(embed_dim_in=emb_dim_in, embed_dim_hid=emb_dim_out,
                                        embed_dim_out=emb_dim_out,
                                        mlp_dropout=dropout_ratio, 
                                        with_norm=with_norm)
        else:
            self.ffn = torch.nn.LayerNorm(emb_dim_in) if with_norm else nn.Identity()


    def forward(self, batch: HeteroData):
        res_graph = self.branch_graph(batch)

        for key in res_graph:
            x = res_graph[key]
            if self.flag_use_ffn:
                batch[key].x = self.ffn[key](x)
            else:
                batch[key].x = self.ffn(x)
        
        return batch

# -------------------------- #
#     4. powerflow net       #
# -------------------------- #
class PFNet(nn.Module):
    def __init__(self, 
                 hidden_channels, 
                 num_block, 
                 with_norm,  
                 batch_size,
                 dropout_ratio,
                 heads_num, 
                 gcn_type="resgatedgraphconv",
                 flag_use_ffn=True,
                 flag_use_fusion=True,
                 flag_use_edge_feat=False):
        super(PFNet, self).__init__()

        # ---- parse params ----
        if isinstance(hidden_channels, list):
            hidden_block_layers = hidden_channels
            num_block = len(hidden_block_layers) - 1
        elif isinstance(hidden_channels, int):
            hidden_block_layers = [hidden_channels] * (num_block+1)
        else:
            raise TypeError("Unsupported type: {}".format(type(hidden_channels)))
        self.hidden_block_layers = hidden_block_layers
        self.flag_use_edge_feat = flag_use_edge_feat

        # ---- edge encoder ----
        if self.flag_use_edge_feat:
            self.edge_encoder = Linear(5, hidden_channels)
            edge_dim = hidden_channels
        else:
            self.edge_encoder = None
            edge_dim = 5

        # ---- node encoder ----
        self.encoders = nn.ModuleDict()
        self.encoders['PQ'] = Linear(6, hidden_block_layers[0])
        self.encoders['PV'] = Linear(6, hidden_block_layers[0])
        self.encoders['Slack'] = Linear(6, hidden_block_layers[0])
        
        # ---- blocks ----
        self.blocks = nn.ModuleList()
        for channel_in, channel_out in zip(hidden_block_layers[:-1], hidden_block_layers[1:]):
            self.blocks.append(
                HybridBlock(emb_dim_in=channel_in, 
                    emb_dim_out=channel_out, 
                    with_norm=with_norm, 
                    edge_dim=edge_dim, 
                    batch_size=batch_size,
                    dropout_ratio=dropout_ratio,
                    gcn_type=gcn_type,
                    flag_use_ffn=flag_use_ffn,
                    heads_num=heads_num)
            )
        self.num_blocks = len(self.blocks)
        
        # predictor        
        self.flag_use_fusion = flag_use_fusion
        if self.flag_use_fusion:
            final_dim = sum(hidden_block_layers) - hidden_block_layers[0]
        else:
            final_dim = hidden_block_layers[-1]

        self.predictor = nn.ModuleDict()
        self.predictor['PQ'] = Linear(final_dim, 6)
        self.predictor['PV'] = Linear(final_dim, 6)
        

    def forward(self, batch):
        # construct edge feats if neccessary
        if self.flag_use_edge_feat:
            for key in batch.edge_attr_dict:
                cur_edge_attr = batch.edge_attr_dict[key]
                r, x = cur_edge_attr[:, 0], cur_edge_attr[:, 1]
                cur_edge_attr[:, 0], cur_edge_attr[:, 1] = \
                    1.0 / torch.sqrt(r ** 2 + x ** 2), torch.arctan(r / x)
                # edge_attr_dict[key] = self.edge_encoder(cur_edge_attr)
                batch[key].edge_attr = self.edge_encoder(cur_edge_attr)
        
        # encoding
        for key, x in batch.x_dict.items():
            # print("="*20, key, "\t", x.shape)
            batch[key].x = self.encoders[key](x)

        # blocks and aspp
        if self.flag_use_fusion:
            multi_level_pq = []
            multi_level_pv = []
            for index, block in enumerate(self.blocks):
                    batch = block(batch)
                    multi_level_pq.append(batch["PQ"].x)
                    multi_level_pv.append(batch["PV"].x)

            output = {
                'PQ': self.predictor['PQ'](torch.cat(multi_level_pq, dim=1)),
                'PV': self.predictor['PV'](torch.cat(multi_level_pv, dim=1))
            }
        else:

            for block in self.blocks:
                batch = block(batch)

            output = {
                'PQ': self.predictor['PQ'](batch['PQ'].x),
                'PV': self.predictor['PV'](batch['PV'].x),
            }

        return output

# -------------------------- #
#     5. iterative pf       #
# -------------------------- #

class IterGCN(nn.Module):
    def __init__(self, 
                 hidden_channels, 
                 num_block, 
                 with_norm,
                 num_loops_train, 
                 scaling_factor_vm, 
                 scaling_factor_va, 
                 loss_type,
                 batch_size, **kwargs):
        super(IterGCN, self).__init__()
        # param
        self.scaling_factor_vm = scaling_factor_vm
        self.scaling_factor_va = scaling_factor_va
        self.num_loops = num_loops_train

        # model
        self.net = PFNet(hidden_channels=hidden_channels, 
                         num_block=num_block, 
                         with_norm=with_norm, 
                         batch_size=batch_size, 
                         dropout_ratio=kwargs.get("dropout_ratio", 0.1), 
                         heads_num=kwargs.get("heads_num", 4),
                         gcn_type=kwargs.get("gcn_type", "resgatedgraphconv"),
                         flag_use_ffn=kwargs.get("flag_use_ffn", True),
                         flag_use_fusion=kwargs.get("flag_use_fusion", True),
                         flag_use_edge_feat=kwargs.get("flag_use_edge_feat", False)
                    )
        # include a ema model for better I/O
        self.ema_warmup_epoch = kwargs.get("ema_warmup_epoch", 0)
        self.ema_decay_param = kwargs.get("ema_decay_param", 0.99)
        self.flag_use_ema = kwargs.get("flag_use_ema", False)
        if self.flag_use_ema:
            self.ema_model = PFNet(hidden_channels=hidden_channels, 
                         num_block=num_block, 
                         with_norm=with_norm, 
                         batch_size=batch_size, 
                         dropout_ratio=kwargs.get("dropout_ratio", 0.1), 
                         heads_num=kwargs.get("heads_num", 4),
                         gcn_type=kwargs.get("gcn_type", "resgatedgraphconv"),
                         flag_use_ffn=kwargs.get("flag_use_ffn", True),
                         flag_use_fusion=kwargs.get("flag_use_fusion", True),
                         flag_use_edge_feat=kwargs.get("flag_use_edge_feat", False)
                    )

            for p in self.ema_model.parameters():
                p.requires_grad = False
        else:
            self.ema_model = None

        # loss
        if loss_type == 'l1':
            self.critien = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.critien = nn.SmoothL1Loss()
        elif loss_type == 'l2':
            self.critien = nn.MSELoss()
        elif loss_type == 'l3':
            self.critien = nn.HuberLoss()   
        else:
            raise TypeError(f"no such loss type: {loss_type}")

        # loss weights
        self.flag_weighted_loss = kwargs.get("flag_weighted_loss", False)
        self.loss_weight_equ = kwargs.get("loss_weight_equ", 1.0)
        self.loss_weight_vm = kwargs.get("loss_weight_vm", 1.0)
        self.loss_weight_va = kwargs.get("loss_weight_va", 1.0)

    def update_ema_model(self, epoch, i_iter, len_loader):
        if not self.flag_use_ema:
            return 
        
        # update teacher model with EMA
        with torch.no_grad():
            if epoch > self.ema_warmup_epoch:
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len_loader * self.ema_warmup_epoch
                        + 1
                    ),
                    self.ema_decay_param,
                )
            else:
                ema_decay = 0.0

            # update weight
            for param_train, param_eval in zip(self.net.parameters(), self.ema_model.parameters()):
                param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
            # update bn
            for buffer_train, buffer_eval in zip(self.net.buffers(), self.ema_model.buffers()):
                buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)
                # buffer_eval.data = buffer_train.data


    def forward(self, batch, flag_return_losses=False, flag_use_ema_infer=False, num_loop_infer=0):
        # get size
        num_PQ = batch['PQ'].x.shape[0]
        num_PV = batch['PV'].x.shape[0]
        num_Slack = batch['Slack'].x.shape[0]
        Vm, Va, P_net, Q_net, Gs, Bs = 0, 1, 2, 3, 4, 5

        # use different loops during inference phase
        if num_loop_infer < 1:
            num_loops = self.num_loops
        else:
            num_loops = num_loop_infer
        
        # whether use ema model for inference
        if not self.flag_use_ema:
            flag_use_ema_infer = False

        # loss record
        loss = 0.0
        res_dict = {"loss_equ": 0.0, "loss_pq_vm": 0.0, "loss_pq_va": 0.0, "loss_pv_va": 0.0}
        Ybus = create_Ybus(batch.detach())
        delta_p, delta_q = deltapq_loss(batch, Ybus)

        # iterative loops
        for i in range(num_loops):
            # print("-"*50, i)
            # ----------- updated input ------------
            cur_batch = batch.clone()

            # use ema for better iterative fittings
            if self.flag_use_ema and i > 0 and not flag_use_ema_infer:
                self.ema_model.eval()
                with torch.no_grad():
                    output_ema = self.ema_model(cur_batch_hist)
                del cur_batch_hist
                cur_batch['PV'].x[:, Va] = cur_batch['PV'].x[:, Va] - output['PV'][:, Va] * self.scaling_factor_va + output_ema['PV'][:, Va] * self.scaling_factor_va
                cur_batch['PQ'].x[:, Vm] = cur_batch['PQ'].x[:, Vm] - output['PQ'][:, Vm] * self.scaling_factor_vm + output_ema['PQ'][:, Vm] * self.scaling_factor_vm
                cur_batch['PQ'].x[:, Va] = cur_batch['PQ'].x[:, Va] - output['PQ'][:, Va] * self.scaling_factor_va + output_ema['PQ'][:, Va] * self.scaling_factor_va

                delta_p, delta_q = deltapq_loss(cur_batch, Ybus)
                self.ema_model.train()
                # print("#"*20, cur_batch['PQ'].x.shape)

            # update the inputs --- use deltap and deltaq
            cur_batch['PQ'].x[:, P_net] = delta_p[:num_PQ]  # deltap
            cur_batch['PQ'].x[:, Q_net] = delta_q[:num_PQ]  # deltaq
            cur_batch['PV'].x[:, P_net] = delta_p[num_PQ:num_PQ+num_PV]
            cur_batch = cur_batch.detach()
            cur_batch_hist = cur_batch.clone().detach()
            
            # ----------- forward ------------
            if flag_use_ema_infer:
                output = self.ema_model(cur_batch)
            else:
                output = self.net(cur_batch)

            # --------------- update vm and va --------------
            batch['PV'].x[:, Va] += output['PV'][:, Va] * self.scaling_factor_va
            batch['PQ'].x[:, Vm] += output['PQ'][:, Vm] * self.scaling_factor_vm
            batch['PQ'].x[:, Va] += output['PQ'][:, Va] * self.scaling_factor_va

            # --------------- calculate loss --------------
            delta_p, delta_q = deltapq_loss(batch, Ybus)

            equ_loss = self.critien(delta_p[:num_PQ+num_PV],
                                    torch.zeros_like(delta_p[:num_PQ+num_PV]))\
                    + self.critien(delta_q[:num_PQ][batch['PQ'].q_mask],
                                    torch.zeros_like(delta_q[:num_PQ][batch['PQ'].q_mask]))
            
            pq_vm_loss = self.critien(batch['PQ'].x[:,Vm], batch['PQ'].y[:,Vm])
            pv_va_loss = self.critien(batch['PV'].x[:,Va], batch['PV'].y[:,Va])
            pq_va_loss = self.critien(batch['PQ'].x[:,Va], batch['PQ'].y[:,Va])

            if flag_return_losses:
                res_dict['loss_equ'] += equ_loss.cpu().item()
                res_dict['loss_pq_vm'] += pq_vm_loss.cpu().item()
                res_dict['loss_pq_va'] += pq_va_loss.cpu().item()
                res_dict['loss_pv_va'] += pv_va_loss.cpu().item()
            
            if self.flag_weighted_loss:
                loss = loss + equ_loss * self.loss_weight_equ + pq_vm_loss * self.loss_weight_vm + (pv_va_loss + pq_va_loss) * self.loss_weight_va
            else:
                loss = loss + equ_loss + pq_vm_loss + pv_va_loss + pq_va_loss
            

        batch['PQ'].x[~batch['PQ'].q_mask, Q_net] = -delta_q[:num_PQ][~batch['PQ'].q_mask]
        batch['PV'].x[:, Q_net] = -delta_q[num_PQ:num_PQ+num_PV]
        batch['Slack'].x[:, P_net] = -delta_p[num_PQ+num_PV:num_PQ+num_PV+num_Slack]
        batch['Slack'].x[:, Q_net] = -delta_q[num_PQ+num_PV:num_PQ+num_PV+num_Slack]

        if flag_return_losses:
            return batch, loss, res_dict
        return batch, loss
