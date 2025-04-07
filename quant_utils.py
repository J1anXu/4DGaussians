import os
import sys
from os.path import join
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))
import torch
import sys
from bitarray import bitarray
import numpy as np
from scene.kmeans_quantize import Quantize_kMeans

class QuantizeKMeansManager:
    def __init__(self, gaussians, quantized_params, n_cls_dc, n_cls_sh, n_cls, n_it):
        # 定义量化参数和簇数量的映射关系
        self.quantizer_configs = {
            'pos': n_cls_dc, # gaussians._xyz
            'dc': n_cls_dc, # gaussians._features_dc
            'sh': n_cls_sh, # gaussians._features_rest
            'scale': n_cls, # gaussians._scaling
            'rot': n_cls, # gaussians._rotation
            'scale_rot': n_cls,
            'sh_dc': n_cls_sh
        }
        self.gaussians = gaussians
        # 创建一个字典来存储量化实例
        self.quantized_kmeans = {}
        self.quantized_params = quantized_params
        # 创建对应的量化实例
        self._initialize_quantizers(quantized_params, n_it)

    def _initialize_quantizers(self, quantized_params, n_it):
        """
        根据 `quantized_params` 创建 `Quantize_kMeans` 实例并存储在字典中
        """
        for param, num_clusters in self.quantizer_configs.items():
            if param in quantized_params:
                self.quantized_kmeans[param] = Quantize_kMeans(num_clusters=num_clusters, num_iters=n_it)


    def prune(self, valid_indices):
        if 'pos' in self.quantized_params:
            self.quantized_kmeans['pos'].prune(valid_indices, self.gaussians._xyz)
        if 'dc' in self.quantized_params:
            self.quantized_kmeans['dc'].prune(valid_indices, self.gaussians._features_dc)
        if 'sh' in self.quantized_params:
            self.quantized_kmeans['sh'].prune(valid_indices, self.gaussians._features_rest)
        if 'scale' in self.quantized_params:
            self.quantized_kmeans['scale'].prune(valid_indices, self.gaussians._scaling)
        if 'rot' in self.quantized_params:
            self.quantized_kmeans['rot'].prune(valid_indices, self.gaussians._rotation)


    def has_quantizer(self, param):
        """
        检查是否已创建指定类型的量化器
        """
        return param in self.quantized_kmeans

    def get_quantizer(self, param):
        """
        获取指定类型的量化器实例
        """
        if self.has_quantizer(param):
            return self.quantized_kmeans[param]
        else:
            raise KeyError(f"Quantizer for '{param}' is not initialized.")
    
    def forward_all(self, gaussians, quantized_params, assign=False, update_centers_flag=False):
        """
        批量执行所有量化器的 `forward` 方法，只要在 `quantized_params` 列表中
        """
        for param in quantized_params:
            if param not in self.quantized_kmeans:
                continue  # 跳过没有初始化的量化器

            quantizer = self.quantized_kmeans[param]

            # 根据参数选择调用对应的 `forward` 方法
            if param == 'pos':
                quantizer.forward_pos(gaussians, assign=assign, update_centers_flag=update_centers_flag)
            elif param == 'dc':
                quantizer.forward_dc(gaussians, assign=assign, update_centers_flag=update_centers_flag)
            elif param == 'sh':
                quantizer.forward_frest(gaussians, assign=assign, update_centers_flag=update_centers_flag)
            elif param == 'scale':
                quantizer.forward_scale(gaussians, assign=assign, update_centers_flag=update_centers_flag)
            elif param == 'rot':
                quantizer.forward_rot(gaussians, assign=assign, update_centers_flag=update_centers_flag)
            elif param == 'scale_rot':
                quantizer.forward_scale_rot(gaussians, assign=assign, update_centers_flag=update_centers_flag)
            elif param == 'sh_dc':
                quantizer.forward_dcfrest(gaussians, assign=assign, update_centers_flag=update_centers_flag)


def dec2binary(x, n_bits=None):
    """Convert decimal integer x to binary.

    Code from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    """
    if n_bits is None:
        n_bits = torch.ceil(torch.log2(x)).type(torch.int64)
    mask = 2**torch.arange(n_bits-1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)

def save_kmeans(kmeans_list, quantized_params, out_dir):
    """Save the codebook and indices of KMeans.

    """
    # Convert to bitarray object to save compressed version
    # saving as npy or pth will use 8bits per digit (or boolean) for the indices
    # Convert to binary, concat the indices for all params and save.
    bitarray_all = bitarray([])
    for kmeans in kmeans_list:
        n_bits = int(np.ceil(np.log2(len(kmeans.cls_ids))))
        assignments = dec2binary(kmeans.cls_ids, n_bits)
        bitarr = bitarray(list(map(bool, assignments.cpu().numpy().flatten())))
        bitarray_all.extend(bitarr)
    os.makedirs(out_dir,exist_ok=True)
    with open(join(out_dir, 'kmeans_inds.bin'), 'wb') as file:
        bitarray_all.tofile(file)

    # Save details needed for loading
    args_dict = {}
    args_dict['params'] = quantized_params
    args_dict['n_bits'] = n_bits
    args_dict['total_len'] = len(bitarray_all)
    np.save(join(out_dir, 'kmeans_args.npy'), args_dict)
    centers_dict = {param: kmeans.centers for (kmeans, param) in zip(kmeans_list, quantized_params)}

    # Save codebook
    torch.save(centers_dict, join(out_dir, 'kmeans_centers.pth'))