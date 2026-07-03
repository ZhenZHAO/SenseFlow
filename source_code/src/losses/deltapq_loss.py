import torch
from torch_geometric.data import HeteroData

def create_Ybus(batch: HeteroData):
    homo_batch = batch.to_homogeneous().detach()
    bus = homo_batch.x
    index_diff = homo_batch.edge_index[1, :] - homo_batch.edge_index[0, :]
    # to index bigger than from index
    edge_attr = homo_batch.edge_attr[index_diff > 0, :]
    edge_index_ori = homo_batch.edge_index[:, index_diff > 0]
    device = batch['PQ'].x.device
    with torch.no_grad():
        edge_mask = torch.isnan(edge_attr[:,0])
        edge_attr = edge_attr[~edge_mask]
        edge_index = torch.vstack([edge_index_ori[0][~edge_mask],edge_index_ori[1][~edge_mask]])
        # makeYbus, reference to pypower makeYbus
        nb = bus.shape[0]  # number of buses
        nl = edge_index.shape[1]  # number of edges
        Vm, Va, P_net, Q_net, Gs, Bs = 0, 1, 2, 3, 4, 5
        BR_R, BR_X, BR_B, TAP, SHIFT = 0, 1, 2, 3, 4

        Ys = 1.0 / (edge_attr[:, BR_R] + 1j * edge_attr[:, BR_X])
        Bc = edge_attr[:, BR_B]
        tap = torch.ones(nl).to(device)
        i = torch.nonzero(edge_attr[:, TAP])
        tap[i] = edge_attr[i, TAP]
        tap = tap * torch.exp(1j * edge_attr[:, SHIFT])

        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt / (tap * torch.conj(tap))
        Yft = - Ys / torch.conj(tap)
        Ytf = - Ys / tap

        Ysh = bus[:, Gs] + 1j * bus[:, Bs]

        # build connection matrices
        f = edge_index[0]
        t = edge_index[1]
        Cf = torch.sparse_coo_tensor(
            torch.vstack([torch.arange(nl).to(device), f]),
            torch.ones(nl).to(device),
            (nl, nb)
        ).to(torch.complex64)
        Ct = torch.sparse_coo_tensor(
            torch.vstack([torch.arange(nl).to(device), t]),
            torch.ones(nl).to(device),
            (nl, nb)
        ).to(torch.complex64)

        i_nl = torch.cat([torch.arange(nl), torch.arange(nl)], dim=0).to(device)
        i_ft = torch.cat([f, t], dim=0)

        Yf = torch.sparse_coo_tensor(
            torch.vstack([i_nl, i_ft]),
            torch.cat([Yff, Yft], dim=0),
            (nl, nb),
            dtype=torch.complex64
        )

        Yt = torch.sparse_coo_tensor(
            torch.vstack([i_nl, i_ft]),
            torch.cat([Ytf, Ytt], dim=0),
            (nl, nb),
            dtype=torch.complex64
        )

        Ysh_square = torch.sparse_coo_tensor(
            torch.vstack([torch.arange(nb), torch.arange(nb)]).to(device),
            Ysh,
            (nb, nb),
            dtype=torch.complex64
        )

        Ybus = torch.matmul(Cf.T.to(torch.complex64), Yf) +\
                torch.matmul(Ct.T.to(torch.complex64), Yt) + Ysh_square
    return Ybus

def deltapq_loss(batch, Ybus):
    Vm, Va, P_net, Q_net = 0, 1, 2, 3
    bus = batch.to_homogeneous().x
    v = bus[:, Vm] * torch.exp(1j * bus[:, Va])
    i = torch.conj(torch.matmul(Ybus, v))
    s = v * i + bus[:, P_net] + 1j * bus[:, Q_net]

    delta_p = torch.real(s)
    delta_q = torch.imag(s)
    return delta_p, delta_q
