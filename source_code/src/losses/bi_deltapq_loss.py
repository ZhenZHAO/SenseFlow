import torch
from torch_geometric.data import HeteroData


def bi_deltapq_loss(graph_data: HeteroData, need_clone=False,
                    filt_type=True, aggr='abs'):
    """compute deltapq loss

    Args:
        graph_data (Hetero Graph): Batched Hetero graph data
        preds (dict): preds results

    Returns:
        torch.float: deltapq loss
    """
    def inner_deltapq_loss(bus, branch, edge_index, device):
        # makeYbus, reference to pypower makeYbus
        nb = bus.shape[0]  # number of buses
        nl = edge_index.shape[1]  # number of branch

        # branch = homo_graph_data.edge_attr
        BR_R, BR_X, BR_B, TAP, SHIFT = 0, 1, 2, 3, 4
        # bus = homo_graph_data.x
        PD, QD, GS, BS, PG, QG, VM, VA = 0, 1, 2, 3, 4, 5, 6, 7

        Ys = 1.0 / (branch[:, BR_R] + 1j * branch[:, BR_X])
        Bc = branch[:, BR_B]
        tap = torch.ones(nl).to(device)
        i = torch.nonzero(branch[:, TAP])
        tap[i] = branch[i, TAP]
        tap = tap * torch.exp(1j * branch[:, SHIFT])

        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt / (tap * torch.conj(tap))
        Yft = - Ys / torch.conj(tap)
        Ytf = - Ys / tap

        Ysh = bus[:, GS] + 1j * bus[:, BS]

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

        v = bus[:, VM] * torch.exp(1j * bus[:, VA])

        i = torch.matmul(Ybus, v)
        i = torch.conj(i)
        s = v * i
        pd = bus[:, PD] + 1j * bus[:, QD]
        pg = bus[:, PG] + 1j * bus[:, QG]
        s = s + pd - pg

        delta_p = torch.real(s)
        delta_q = torch.imag(s)
        return delta_p, delta_q

    # preprocess
    if need_clone:
        graph_data = graph_data.clone()
    device = graph_data['PQ'].x.device

    # PQ: PD, QD, GS, BS, PG, QG, Vm, Va
    graph_data['PQ'].x = torch.cat([
        graph_data['PQ'].supply,
        graph_data['PQ'].x[:, :2]],
        dim=1)
    # PV: PD, QD, GS, BS, PG, QG, Vm, Va
    graph_data['PV'].x = torch.cat([
        graph_data['PV'].supply,
        graph_data['PV'].x[:, :2]],
        dim=1)
    # Slack PD, QD, GS, BS, PG, QG, Vm, Va
    graph_data['Slack'].x = torch.cat([
        graph_data['Slack'].supply,
        graph_data['Slack'].x[:, :2]],
        dim=1)

    # convert to homo graph for computing Ybus loss
    homo_graph_data = graph_data.to_homogeneous()

    index_diff = homo_graph_data.edge_index[1, :] - homo_graph_data.edge_index[0, :]
    # to index bigger than from index
    edge_attr_1 = homo_graph_data.edge_attr[index_diff > 0, :]
    edge_index_1 = homo_graph_data.edge_index[:, index_diff > 0]
    delta_p_1, delta_q_1 = inner_deltapq_loss(homo_graph_data.x, edge_attr_1, edge_index_1, device)

    # from index bigger than to index
    edge_index_2 = homo_graph_data.edge_index[:, index_diff < 0]
    edge_attr_2 = homo_graph_data.edge_attr[index_diff < 0, :]
    delta_p_2, delta_q_2 = inner_deltapq_loss(homo_graph_data.x, edge_attr_2, edge_index_2, device)

    delta_p, delta_q = (delta_p_1 + delta_p_2) / 2.0, (delta_q_1 + delta_q_2) / 2.0

    if filt_type:
        PQ_mask = homo_graph_data['node_type'] == 0
        PV_mask = homo_graph_data['node_type'] == 1
        delta_p = delta_p[PQ_mask | PV_mask]
        delta_q = delta_q[PQ_mask]

    if aggr == "abs":
        loss = delta_p.abs().mean() + delta_q.abs().mean()
    elif aggr == "square":
        loss = (delta_p**2).mean() + (delta_q**2).mean()
    else:
        raise TypeError(f"no such aggr: {aggr}")
    return loss
