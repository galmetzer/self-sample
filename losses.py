import torch
from torch_cluster import knn


def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor, mse=False):
    pc1, pc2 = pc1.transpose(1, 2), pc2.transpose(1, 2)

    # Batch X N1 X N2 X 3
    distance_matrix = (pc1[:, :, None, :3] - pc2[:, None, :, :3]).norm(dim=-1)
    if mse:
        distance_matrix = distance_matrix ** 2
    d12, _ = distance_matrix.min(dim=2)
    d21, _ = distance_matrix.min(dim=1)
    return d12.mean(dim=-1).sum() + d21.mean(dim=-1).sum()


def chamfer_distance_cluster(pc1: torch.Tensor, pc2: torch.Tensor):
    def get_batch(data):
        batch_size, N, _ = data.shape  # (batch_size, num_points, 3/6)
        pos = data.view(batch_size * N, -1)
        batch = torch.zeros((batch_size, N), device=pos.device, dtype=torch.long)
        for i in range(batch_size): batch[i] = i
        batch = batch.view(-1)
        return pos, batch

    batch_size = pc1.shape[0]
    pc1, pc2 = pc1.transpose(1, 2), pc2.transpose(1, 2)
    pc1, pc2 = pc1.contiguous(), pc2.contiguous()
    pc1, batch1 = get_batch(pc1)
    pc2, batch2 = get_batch(pc2)
    nn2_1 = knn(pc1, pc2, 1, batch_x=batch1, batch_y=batch2)[1]
    nn1_2 = knn(pc2, pc1, 1, batch_x=batch2, batch_y=batch1)[1]
    d21 = (pc1[nn2_1] - pc2)[:, :3].norm(dim=-1).mean() * batch_size
    d12 = (pc2[nn1_2] - pc1)[:, :3].norm(dim=-1).mean() * batch_size
    return d21 + d12
