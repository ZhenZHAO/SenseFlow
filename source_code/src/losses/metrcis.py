from torch_geometric.data import HeteroData
import torch
import torch.nn as nn
from .deltapq_loss import deltapq_loss, create_Ybus

def vm_va_matrix(batch: HeteroData, mode="train"):
    Vm, Va, P_net, Q_net, Gs, Bs = 0, 1, 2, 3, 4, 5
    mse_eval = nn.MSELoss()
    Ybus = create_Ybus(batch)
    delta_p, delta_q = deltapq_loss(batch, Ybus)
    matrix = {
        f"{mode}/PQ_Vm_rmse":
            torch.sqrt(mse_eval(batch['PQ'].x[:, Vm], batch['PQ'].y[:, Vm])).item(),
        f"{mode}/PQ_Va_rmse":
            torch.sqrt(mse_eval(batch['PQ'].x[:, Va], batch['PQ'].y[:, Va])).item(),
        f"{mode}/PV_Va_rmse":
            torch.sqrt(mse_eval(batch['PV'].x[:, Va], batch['PV'].y[:, Va])).item(),
        f"{mode}/delta_p":
            delta_p.abs().mean().item(),
        f"{mode}/delta_q":
            delta_q.abs().mean().item(),
    }

    return matrix
