import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def tt_to_tensor(tt_cores):
    # 假设 tt_cores 是一个包含多个张量的列表
    out = tt_cores[0]
    
    # 确保每次合并后，维度是对的
    for i in range(1, len(tt_cores)):
        out = torch.tensordot(out, tt_cores[i], dims=[[out.dim() - 1], [0]])

    # 计算最终的形状并进行 reshape
    final_shape = out.shape  # 确保 final_shape 是最终合并后的张量形状
    return out.reshape(final_shape)

def restore_grid_coefs(grid_coefs):
    full_grid_coefs = nn.ParameterList()
    for tt_cores in grid_coefs:
        full_tensor = tt_to_tensor(list(tt_cores))  # 关键修复：把 ParameterList 转成 list
        full_tensor = full_tensor.squeeze(-1)
                
        core = nn.Parameter(full_tensor)
        # 初始化代码移动到此
        # 这里初始化要改 查看原来是怎么初始化core的
        nn.init.ones_(core)
        full_grid_coefs.append(core)  # 注意需要转回 Parameter 否则无法被管理
    return full_grid_coefs



def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

# deformation_net.grid.grids.0.0                      shape=(1, 16, 64, 64)       size=0.25 MB
# deformation_net.grid.grids.0.1                      shape=(1, 16, 64, 64)       size=0.25 MB
# deformation_net.grid.grids.0.2                      shape=(1, 16, 150, 64)      size=0.59 MB
# deformation_net.grid.grids.0.3                      shape=(1, 16, 64, 64)       size=0.25 MB
# deformation_net.grid.grids.0.4                      shape=(1, 16, 150, 64)      size=0.59 MB
# deformation_net.grid.grids.0.5                      shape=(1, 16, 150, 64)      size=0.59 MB
# deformation_net.grid.grids.1.0                      shape=(1, 16, 128, 128)     size=1.00 MB
# deformation_net.grid.grids.1.1                      shape=(1, 16, 128, 128)     size=1.00 MB
# deformation_net.grid.grids.1.2                      shape=(1, 16, 150, 128)     size=1.17 MB
# deformation_net.grid.grids.1.3                      shape=(1, 16, 128, 128)     size=1.00 MB
# deformation_net.grid.grids.1.4                      shape=(1, 16, 150, 128)     size=1.17 MB
# deformation_net.grid.grids.1.5                      shape=(1, 16, 150, 128)     size=1.17 MB
# 核心代码 coo_combs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]  # 共6个planes




def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp
def calculate_memory(tensor_list):
    total_memory = 0
    for tt_cores in tensor_list:
        for core in tt_cores:
            # 每个张量的内存 = 元素数量 × 每个元素的字节数 (float32 每个元素 4字节)
            total_memory += core.numel() * core.element_size()
    return total_memory

class HexPlaneField_tt(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            print("prepare---------")

            # Step 1: 获取 grid_coefs 此处返回的已经是张量链列表
            gp_tt = self.init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )

            # Step 2: 立即还原所有张量
            gp = restore_grid_coefs(gp_tt)

            # 计算 gp_tt 和 gp 的内存
            gp_tt_memory = calculate_memory(gp_tt)
            gp_memory = calculate_memory(gp)

            # 转换为 MB
            gp_tt_memory_mb = gp_tt_memory / (1024 ** 2)
            gp_memory_mb = gp_memory / (1024 ** 2)

            print(f"gp_tt 内存占用: {gp_tt_memory_mb:.2f} MB")
            print(f"gp 内存占用: {gp_memory_mb:.2f} MB")

            # 打印 gp 中所有张量的形状
            for idx, tt_cores in enumerate(gp):
                print(f"Shape of tensor chain {idx}:")
                for core_idx, core in enumerate(tt_cores):
                    print(f"  Shape of core {core_idx}: {core.shape}")

            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)

        print("feature_dim:",self.feat_dim)

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features



    def init_grid_param(
            self,
            grid_nd: int,
            in_dim: int,
            out_dim: int,
            reso: Sequence[int],
            a: float = 0.1,
            b: float = 0.5):
        
        assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
        has_time_planes = in_dim == 4
        assert grid_nd <= in_dim
        coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
        grid_coefs = nn.ParameterList()

        for ci, coo_comb in enumerate(coo_combs):

            # Step 1：确定维度
            shape = [out_dim] + [reso[cc] for cc in coo_comb[::-1]]
            order = len(shape)
            # 设定 TT 秩（可以用固定值，如 8）
            tt_rank = 8
            ranks = [1] + [tt_rank] * (order - 1) + [1]  # eg: [1, 8, 8, 1]

            # Step 2：构造 TT 核
            tt_cores = nn.ParameterList()
            for i in range(order):
                r1 = ranks[i]
                n = shape[i]
                r2 = ranks[i+1]
                core = nn.Parameter(torch.empty(r1, n, r2))  # 👈 明确 device

                # 初始化核而不是 dense tensor
                if has_time_planes and 3 in coo_comb:
                    nn.init.ones_(core)
                else:
                    nn.init.uniform_(core, a=a, b=b)

                tt_cores.append(core)

            grid_coefs.append(tt_cores)

        return grid_coefs  # 返回的是多个 TT 表示，每个是一个 ParameterList