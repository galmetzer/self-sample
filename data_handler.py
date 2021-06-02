import torch
import random
import util
from torch.utils.data import Dataset
from pathlib import Path
import os
from typing import List
from kmeans_pytorch import kmeans
NUM_CLUSTERS = 5


def get_dataset(mode='sweep'):
    mode = mode.lower().strip()

    if mode == 'sweep':
        return SweepData

    if mode == 'random':
        return Random

    if mode == 'curvature':
        return CurvatureData

    if mode == 'density':
        return DensityData


class SweepData(Dataset):
    def __init__(self, pc, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.pc: torch.Tensor = pc.to(self.device)
        self.n_pts = self.pc.shape[0]

        self.const = None

        # for sweep
        self.reminder = -1
        self.blocks = int(self.pc.shape[0] / (args.D2 + args.D1))

    def single(self, size):
        return self.pc[torch.randperm(self.n_pts)[:size], :3].permute(1, 0).unsqueeze(0).to(self.real_device)

    def __getitem__(self, item):
        current_block = int(item / self.blocks)
        if self.reminder != current_block:
            # shuffle pc
            self.pc = self.pc[torch.randperm(self.pc.shape[0]), :]
            self.reminder = current_block

        index = item % self.blocks
        offset = index * (self.args.D1 + self.args.D2)
        d1 = self.pc[offset: offset + self.args.D1, :3].transpose(0, 1)
        d2 = self.pc[offset + self.args.D1: offset + self.args.D1 + self.args.D2,
             :3].transpose(0, 1)

        return d1, d2

    def __len__(self):
        return self.args.iterations


class Random(Dataset):
    def __init__(self, pc, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.pc: torch.Tensor = pc.to(self.device)
        self.n_pts = self.pc.shape[0]

    def single(self, size):
        return self.pc[torch.randperm(self.n_pts)[:size], :3].permute(1, 0)

    def __getitem__(self, item):
        return self.single(self.args.D1).to(self.device), self.single(self.args.D2).to(self.device)

    def __len__(self):
        return self.args.iterations


class SubsetData(Dataset):
    def __init__(self, pc, weight, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.pc: torch.Tensor = pc.to(self.device)
        self.n_pts = self.pc.shape[0]

        if self.args.kmeans:
            cluster_indx, cluster_centers = kmeans(
                X= weight.unsqueeze(-1), num_clusters=NUM_CLUSTERS, distance='euclidean', device=real_device
            )
            min_indx = torch.argmin(cluster_centers)
            self.criterion_mask = (cluster_indx == min_indx)

        elif self.args.percentile == -1.0:
            self.criterion_mask = weight < weight.mean().to(self.device)
        else:
            kth = weight.kthvalue(int(weight.shape[0] * self.args.percentile))[0]
            self.criterion_mask = weight < kth

        self.high_pc = pc[self.criterion_mask, :].to(self.device)
        self.low_pc = pc[~self.criterion_mask, :].to(self.device)

        self.export_marked()

        # subset ratios
        self.p1 = self.args.p1
        self.p2 = self.args.p2

        self.const = None

    def single(self, size):
        return self.pc[torch.randperm(self.n_pts)[:size], :3].permute(1, 0).unsqueeze(0).to(self.real_device)

    def __getitem__(self, item):
        size1, size2 = self.args.D1, self.args.D2
        D1, D2 = SubsetData.double_sub_sample(self.high_pc, self.low_pc, self.p1,
                                                 self.p2, size1, size2,
                                                 allow_residual=True)

        return D1[:, :3].permute(1, 0), D2[:, :3].permute(1, 0)

    def __len__(self):
        return self.args.iterations

    @staticmethod
    def double_sub_sample(pc1: torch.Tensor, pc2: torch.Tensor, p1, p2, n1, n2, allow_residual=True):
        high_perm = torch.randperm(pc1.shape[0])
        low_perm = torch.randperm(pc2.shape[0])

        d1np1 = int(n1 * p1)
        d1np2 = int(n1 * (1 - p1))
        if d1np1 > pc1.shape[0]:
            d1np2 += (d1np1 - pc1.shape[0])
            d1np1 = pc1.shape[0]
        if d1np2 > pc2.shape[0]:
            d1np1 += (d1np2 - pc2.shape[0])
            d1np2 = pc2.shape[0]

        d2np1 = int(n2 * p2)
        d2np2 = int(n2 * (1 - p2))
        if d2np1 > pc1.shape[0]:
            d2np2 += (d2np1 - pc1.shape[0])
            d2np1 = pc1.shape[0]
        if d2np2 > pc2.shape[0]:
            d2np1 += (d2np2 - pc2.shape[0])
            d2np2 = pc2.shape[0]

        d1idx1, d2idx1 = SubsetData.disjoint_select(high_perm, d1np1, d2np1, allow_residue=allow_residual)

        d2np2 += max(0, d2np1 - d2idx1.shape[0])

        d1idx2, d2idx2 = SubsetData.disjoint_select(low_perm, d1np2, d2np2, allow_residue=allow_residual)

        return torch.cat([pc1[d1idx1, :], pc2[d1idx2, :]], dim=0), torch.cat([pc1[d2idx1, :], pc2[d2idx2, :]],
                                                                             dim=0)

    @staticmethod
    def disjoint_select(pc, n1, n2, allow_residue=True):
        idx1 = pc[:n1]
        if allow_residue:
            residual = max(n1 + n2 - pc.shape[0], 0)
        else:
            residual = 0
        idx2 = pc[max(n1 - residual, 0): n1 + n2]
        return idx1, idx2

    def export_marked(self):
        c = torch.zeros_like(self.pc)
        c[~self.criterion_mask, 0] = 255
        c[self.criterion_mask, 0] = 16

        c[self.criterion_mask, 1] = 255
        c[self.criterion_mask, 2] = 255
        c = c.transpose(0, 1)
        util.export_pc(self.pc.transpose(0, 1), self.args.save_path / 'target-marked.xyz', color=c)


class CurvatureData(SubsetData):
    def __init__(self, pc, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.pc: torch.Tensor = pc.to(self.device)
        self.n_pts = self.pc.shape[0]
        if args.curvature_cache and os.path.exists(args.curvature_cache):
            self.curvature = torch.load(args.curvature_cache)
            print('loaded curvature metric from cache')
        else:
            self.curvature: torch.Tensor = self.get_curvature().to(self.device)
            if args.curvature_cache:
                torch.save(self.curvature, args.curvature_cache)

        # lowest values are selected for subset division
        self.curvature *= -1
        super().__init__(pc, self.curvature, real_device, args)

        print(f'Sharp Shape: {self.high_pc.shape}; Low Shape {self.low_pc.shape}')

    def get_curvature(self):
        div = util.angle_diff(self.pc, self.args.k)
        div[div != div] = 0
        return div


class DensityData(SubsetData):
    def __init__(self, pc, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.pc: torch.Tensor = pc.to(self.device)
        self.n_pts = self.pc.shape[0]

        self.density: torch.Tensor = util.density(pc, args.k).to(
            self.device)

        super().__init__(pc, torch.log(self.density), real_device, args)

        print(f'Not Dense Shape: {self.high_pc.shape}; Dense Shape {self.low_pc.shape}')

