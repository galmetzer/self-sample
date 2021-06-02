import torch
import numpy as np
import math
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from torch_cluster import knn_graph


def density(pc: torch.Tensor, k=10):
    knn = knn_graph(pc[:, :3], k, loop=False)
    knn_indx, _ = knn.view(2, pc.shape[0], k)
    knn_data = pc[knn_indx, :]
    max_distance, _ = (knn_data[:, :, :3] - pc[:, None, :3]).norm(dim=-1).max(dim=-1)
    dense = k / (max_distance ** 3)
    inf_mask = torch.isinf(dense)
    max_val = dense[~inf_mask].max()
    dense[inf_mask] = max_val
    return dense


def get_input(args, center=False) -> torch.Tensor:
    if args.pc is not None:
        with open(args.pc) as file:
            pc = xyz2tensor(file.read(), force_normals=args.force_normal_estimation, k=args.k)
    else:
        raise NotImplementedError('no recognized input type was found')

    if center:
        cm = pc[:, :3].mean(dim=0)
        pc[:, :3] = pc[:, :3] - cm

    return pc


def truncate(number: float, digits: int) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def show_truncated(n: float, digits: int):
    x = str(truncate(n, digits))
    if len(x) < digits + 2:
        x += '0' * (digits + 2 - len(x))
    return x


def n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def xyz2tensor(txt, append_normals=False, estimate_normals_flag=True, force_normals=False, k=20):
    pts = []
    for line in txt.split('\n'):
        line = line.strip()
        spt = line.split(' ')
        if 'nan' in line:
            continue
        if len(spt) == 6:
            pts.append(torch.tensor([float(x) for x in spt]))
        if len(spt) == 3:
            t = [float(x) for x in spt]
            if append_normals and not estimate_normals_flag:
                t += [0.0 for _ in range(3)]
            pts.append(torch.tensor(t))

    rtn = torch.stack(pts, dim=0)
    if (rtn.shape[1] == 3 and estimate_normals_flag) or force_normals:
        print('estimating normals')
        rtn = estimate_normals_torch(rtn, k)
    return rtn


def angle_diff(pc, k):
    INNER_PRODUCT_THRESHOLD = math.pi / 2
    knn = knn_graph(pc[:, :3], k, loop=False)
    knn, _ = knn.view(2, pc.shape[0], k)

    inner_prod = (pc[knn, 3:] * pc[:, None, 3:]).sum(dim=-1)
    inner_prod[inner_prod > 1] = 1
    inner_prod[inner_prod < -1] = -1
    angle = torch.acos(inner_prod)
    angle[angle > INNER_PRODUCT_THRESHOLD] = math.pi - angle[angle > INNER_PRODUCT_THRESHOLD]
    angle = angle.sum(dim=-1)
    return angle


def scalar_to_color(scalar_tensor, minn=-1, maxx=1):
    jet = plt.get_cmap('jet')
    if minn == -1 or maxx == -1:
        cNorm = colors.Normalize(vmin=scalar_tensor.min(), vmax=scalar_tensor.max())
    else:
        cNorm = colors.Normalize(vmin=minn, vmax=maxx)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    p = scalarMap.to_rgba(scalar_tensor.cpu())
    return torch.tensor(p).to(scalar_tensor.device).permute(1, 0)[:3, :]


def estimate_normals_torch(inputpc, max_nn):
    knn = knn_graph(inputpc[:, :3], max_nn, loop=False)
    knn = knn.view(2, inputpc.shape[0], max_nn)[0]
    x = inputpc[knn][:, :, :3]
    temp = x[:, :, :3] - x.mean(dim=1)[:, None, :3]
    cov = temp.transpose(1, 2) @ temp / x.shape[0]
    e, v = torch.symeig(cov, eigenvectors=True)
    n = v[:, :, 0]
    return torch.cat([inputpc[:, :3], n], dim=-1)


def args_to_str(args):
    d = args.__dict__
    txt = ''
    for k in d.keys():
        txt += f'{k}: {d[k]}\n'
    return txt.strip('\n')


def voxel_downsample(point_cloud: torch.Tensor, size=0.003, npoints=-1, max_iters=int(1e2)):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    normals = point_cloud.shape[1] == 6
    if normals:
        pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, 3:])
    if npoints == -1:
        pcd = pcd.voxel_down_sample(size)
        return torch.tensor(pcd.points)

    upper = 0.0001
    lower = 0.5
    for i in range(max_iters):
        mid = (upper + lower) / 2
        tmp = pcd.voxel_down_sample(mid)

        # minimal grid quantization, maximal resolution
        if np.asanyarray(tmp.points).shape[0] > npoints:
            upper = mid
        else:
            lower = mid
    if normals:
        pts = torch.tensor(tmp.points).to(point_cloud.device).type(point_cloud.dtype)
        n = torch.tensor(tmp.normals).to(point_cloud.device).type(point_cloud.dtype)
        return torch.cat([pts, n], dim=-1)
    else:
        return torch.tensor(tmp.points).to(point_cloud.device).type(point_cloud.dtype)


def export_pc(pc, dest, color=None):
    txt = ''

    def t2txt(t):
        return ' '.join(map(lambda x: str(x.item()), t))

    if color is None:
        for i in range(pc.shape[1]):
            txt += f'{t2txt(pc[:, i])}\n'
        txt.strip()
    else:
        for i in range(pc.shape[1]):
            txt += f'v {t2txt(pc[:3, i])} {t2txt(color[:3, i])}\n'
        txt.strip()
        dest = str(dest).replace('.xyz', '.obj')

    with open(dest, 'w+') as file:
        file.write(txt)

