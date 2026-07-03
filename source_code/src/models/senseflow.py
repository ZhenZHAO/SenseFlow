import torch.nn as nn
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
def compute_shortest_path_distances(adj_matrix):
    distances = floyd_warshall(csgraph=adj_matrix, directed=False)
    return distances


def convert_x_to_tanhx(tensor_in):
    return torch.tanh(tensor_in)


# ----- ca
class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)
        
    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)
        
        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        
        attn = torch.matmul(q1, k2) / self.k_dim**0.5
        # print("s1", q1.shape, k2.shape, attn.shape)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3)
        # print("s2", output.shape)
        output= output.contiguous().view(batch_size, seq_len1, -1)
        # print("s3", output.shape)
        output = self.proj_o(output)
        # print("s4", output.shape)
    
        return output


# ------- ffn ---
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


class GatedFusion(nn.Module):
    def __init__(self, in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 batch_size=100,
                 dropout_ratio=0.1):
        super(GatedFusion, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features * 2, hidden_features * 2)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout_ratio)
        self.batch_size = batch_size

    def forward(self, pq_features, slack_features):
        # get size
        BK, D = pq_features.size()
        B = self.batch_size
        K = BK // B
        pq_features = pq_features.view(B, K, D)  # (B, K, D)
        slack_expanded = slack_features.unsqueeze(1).expand(-1, K, -1)  # (B, K, D)
        combined = torch.cat([pq_features, slack_expanded], dim=-1)  # (B, K, 2D)

        x = self.fc1(combined)  # (B, K, 2 * hidden_features)
        x, v = x.chunk(2, dim=-1)  # (B, K, hidden_features) each
        x = self.act(x) * v  # (B, K, hidden_features)
        x = self.fc2(x)  # (B, K, D)
        x = self.drop(x)  # (B, K, D)

        return x.contiguous().view(B*K, D)


# -------------------------- #
#     2. various layers      #
# -------------------------- #
class GraphLayer(torch.nn.Module):
    def __init__(self, 
                 emb_dim, 
                 edge_dim,
                 num_heads,
                 batch_size,
                 with_norm,
                 act_layer=nn.ReLU,
                 gcn_layer_per_block=2):
        super().__init__()
        
        self.graph_layers = nn.ModuleList()
        for _ in range(gcn_layer_per_block):
            self.graph_layers.append(
                HeteroConv({
                        ('PQ', 'default', 'PQ'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('PQ', 'default', 'PV'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('PQ', 'default', 'Slack'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('PV', 'default', 'PQ'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('PV', 'default', 'PV'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('PV', 'default', 'Slack'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('Slack', 'default', 'PQ'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('Slack', 'default', 'PV'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                    }, 
                    aggr='sum')
            )
        self.act_layer = act_layer()
        self.global_transform = nn.Linear(emb_dim, emb_dim)

        self.cross_attention = CrossAttention(in_dim1=emb_dim,
                                              in_dim2=emb_dim,
                                              k_dim=emb_dim//num_heads,
                                              v_dim=emb_dim//num_heads,
                                              num_heads=num_heads)

        self.norm = torch.nn.LayerNorm(emb_dim) if with_norm else nn.Identity()
        self.batch_size = batch_size


    def forward(self, batch: HeteroData):
        graph_x_dict = batch.x_dict

        # vitual global node
        pq_x = torch.stack(torch.chunk(graph_x_dict['PQ'], self.batch_size, dim=0), dim=0) # B, 29, D
        pv_x = torch.stack(torch.chunk(graph_x_dict['PV'], self.batch_size, dim=0), dim=0)
        slack_x = torch.stack(torch.chunk(graph_x_dict['Slack'], self.batch_size, dim=0), dim=0)
        global_feature = torch.cat((pq_x,pv_x,slack_x), dim=1) # B, (29+9+1), D
        global_feature = self.global_transform(global_feature)
        global_feature_mean = global_feature.mean(dim=1, keepdim=True)
        global_feature_max, _ = global_feature.max(dim=1, keepdim=True)

        # forward gcn
        for layer in self.graph_layers:
            graph_x_dict = layer(graph_x_dict, 
                                 batch.edge_index_dict,
                                 batch.edge_attr_dict)
            ## NEW: add non-linear
            graph_x_dict = {key: self.act_layer(x) for key, x in graph_x_dict.items()}

        global_node_feat = torch.cat([global_feature_mean, global_feature_max], dim=1)
        
        # cross attent the global feat.
        res = {}
        for key in ["PQ", "PV"]:
            # get size
            BN, K = batch[key].x.size()
            B = self.batch_size
            N = BN // B
            # ca
            graph_x_dict[key] = graph_x_dict[key] + self.cross_attention(graph_x_dict[key].view(B, N, K), global_node_feat).contiguous().view(B*N, K)
            # norm
            res[key] = self.norm(graph_x_dict[key])
        res["Slack"] = graph_x_dict["Slack"]

        return res


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
    

class FFNFuseLayer(torch.nn.Module):

    def __init__(self, 
                embed_dim_in: int,
                embed_dim_hid: int,
                embed_dim_out: int, 
                mlp_dropout: float, 
                with_norm: bool,
                batch_size: int,
                act_layer=nn.GELU):
        super().__init__()
        self.mlp = GatedFusion(in_features=embed_dim_in, 
                          hidden_features=embed_dim_hid, 
                          out_features=embed_dim_out,
                          act_layer=act_layer, 
                          batch_size=batch_size,
                          dropout_ratio=mlp_dropout)

        self.norm = torch.nn.LayerNorm(embed_dim_out) if with_norm else nn.Identity()

    def forward(self, x, x_aux):
        x = x + self.mlp(x, x_aux)
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
                 dropout_ratio=0.1,
                 layers_in_gcn=2,
                 heads_ca=4):
        super(HybridBlock, self).__init__()
        self.emb_dim_in = emb_dim_in
        self.with_norm = with_norm

        self.branch_graph = GraphLayer(emb_dim=emb_dim_in,
                                       edge_dim=edge_dim, 
                                       num_heads=heads_ca, 
                                       batch_size=batch_size,
                                       with_norm=with_norm, 
                                       gcn_layer_per_block=layers_in_gcn)

        # ---- mlp: activation + increase dimension
        self.ffn = nn.ModuleDict()
        self.ffn['PQ'] = FFNFuseLayer(embed_dim_in=emb_dim_in, embed_dim_hid=emb_dim_out,
                                    embed_dim_out=emb_dim_out,
                                    batch_size=batch_size,
                                    mlp_dropout=dropout_ratio, 
                                    with_norm=with_norm)
        self.ffn['PV'] = FFNFuseLayer(embed_dim_in=emb_dim_in, embed_dim_hid=emb_dim_out,
                                    embed_dim_out=emb_dim_out,
                                    batch_size=batch_size,
                                    mlp_dropout=dropout_ratio, 
                                    with_norm=with_norm)
        self.ffn['Slack'] = FFNLayer(embed_dim_in=emb_dim_in, embed_dim_hid=emb_dim_out,
                                    embed_dim_out=emb_dim_out,
                                    mlp_dropout=dropout_ratio, 
                                    with_norm=with_norm)

    def forward(self, batch: HeteroData):
        res_graph = self.branch_graph(batch)

        feat_slack = res_graph["Slack"]

        for key in res_graph:
            x = res_graph[key]
            if "slack" in key.lower():
                batch[key].x = self.ffn[key](x)
            else:
                batch[key].x = self.ffn[key](x, feat_slack)

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
                 heads_ca, 
                 layers_per_graph=2,
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
                    layers_in_gcn=layers_per_graph,
                    heads_ca=heads_ca)
            )
        self.num_blocks = len(self.blocks)
        
        # predictor        
        final_dim = sum(hidden_block_layers) - hidden_block_layers[0]
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
        param_kan = kwargs.get("kan", dict())
        self.net = PFNet(hidden_channels=hidden_channels, 
                         num_block=num_block, 
                         with_norm=with_norm, 
                         batch_size=batch_size, 
                         dropout_ratio=kwargs.get("dropout_ratio", 0.1), 
                         heads_ca=kwargs.get("heads_ca", 4),
                         layers_per_graph=kwargs.get("layers_per_graph", 2),
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
                            heads_ca=kwargs.get("heads_ca", 4),
                            layers_per_graph=kwargs.get("layers_per_graph", 2),
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
