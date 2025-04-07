import os
import pdb
from tqdm import tqdm
import time

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Quantize_kMeans():
    def __init__(self, num_clusters=100, num_iters=10):
        self.num_clusters = num_clusters
        self.num_kmeans_iters = num_iters
        self.nn_index = torch.empty(0)
        self.centers = torch.empty(0)
        self.vec_dim = 0
        self.cluster_ids = torch.empty(0)
        self.cls_ids = torch.empty(0)
        #self.excl_clusters = []
        #self.excl_cluster_ids = []
        #self.cluster_len = torch.empty(0)
        #self.max_cnt = 0
        #self.n_excl_cls = 0

    def get_loss(self, feat):
        """Calculate the loss between the feature and the quantized centers.

        """
        feat = feat.reshape(-1, self.vec_dim)
        # Initialize a list to store covariance matrices for each cluster
        cluster_covariances = []
        # Iterate over each cluster
        l2_norms = 0
        points_num =0
        for cluster_id in range(self.num_clusters):
            # Get indices of points belonging to the current cluster
            #print("cluster_ids=",self.cluster_ids)
            #print("nn_index=",self.nn_index)
            #print("self.cls_ids=",self.cls_ids)
            #print("number=",self.num_clusters)
            try:
                cluster_indices = (self.nn_index == cluster_id).nonzero(as_tuple=True)[0].cuda()
            except RuntimeError as e:
                print(f"[Error] RuntimeError during nonzero(). Likely numel overflow.")
                print(f"self.nn_index.shape = {self.nn_index.shape}, numel = {self.nn_index.numel()}")
                print(f"cluster_id = {cluster_id}")
                print(f"Error message: {e}")
                raise  # 重新抛出原始异常

            points_num+=cluster_indices.shape[0]
            #print("cluster_indices=",cluster_indices)
            #excl_cluster_indices = (torch.tensor(self.excl_cluster_ids) == cluster_id).nonzero(as_tuple=True)[0].cuda()
            #cluster_indices = torch.cat((cluster_indices, excl_cluster_indices), dim=0)
            #print("after cluster_indices=",cluster_indices)
            if cluster_indices.numel() > 1:  # Check if there are at least 2 points in the cluster
            # Extract features for the current cluster
                #print(torch.cuda.memory_summary(device="cuda", abbreviated=False))
                cluster_feats = feat[cluster_indices]
                #print("cluster_feats=",cluster_feats.shape)
                cluster_mean = self.centers[cluster_id]
                cluster_centered = cluster_feats - cluster_mean
                cluster_covariances = cluster_centered.t() @ cluster_centered / (cluster_feats.shape[0]-1) 
                #print("cluster_covariances=",cluster_covariances)
                l2_norms += torch.trace(cluster_covariances)
        print("l2_norms=",l2_norms)
        #print("points_nums",points_num)
        #print("after std=",l2_norms/torch.std(l2_norms.detach()))
        return l2_norms
    
    def get_dist(self, x, y, mode='sq_euclidean'):
        """Calculate distance between all vectors in x and all vectors in y.

        x: (m, dim)
        y: (n, dim)
        dist: (m, n)
        """
        if mode == 'sq_euclidean_chunk':
            step = 65536
            if x.shape[0] < step:
                step = x.shape[0]
            dist = []
            for i in range(np.ceil(x.shape[0] / step).astype(int)):
                dist.append(torch.cdist(x[(i*step): (i+1)*step, :].unsqueeze(0), y.unsqueeze(0))[0])
            dist = torch.cat(dist, 0)
        elif mode == 'sq_euclidean':
            dist = torch.cdist(x.unsqueeze(0).detach(), y.unsqueeze(0).detach())[0]
        return dist

    # # Update centers in non-cluster assignment iters using cached nn indices.
    # def update_centers(self, feat):
    #     # Detach and reshape the features
    #     feat = feat.detach().reshape(-1, self.vec_dim)
    #     # One-hot encode cluster assignments
    #     one_hot = torch.nn.functional.one_hot(self.nn_index, num_classes=self.num_clusters).float()
    #     # Compute sum of features per cluster
    #     cluster_sums = one_hot.T @ feat  # Shape: (num_clusters, vec_dim)
    #     # Compute count of features per cluster
    #     cluster_counts = one_hot.sum(dim=0).unsqueeze(1)  # Shape: (num_clusters, 1)
    #     # Avoid division by zero and compute mean
    #     self.centers = cluster_sums / (cluster_counts + 1e-6)
    def update_centers(self, feat):
        """
        Calculate the center of each cluster rapidly.
        
        Args:
            feat (torch.Tensor): Feature tensor of shape (N, vec_dim).
            
        Returns:
            torch.Tensor: Cluster centers of shape (num_clusters, vec_dim).
        """
        feat = feat.detach().reshape(-1, self.vec_dim)
        device = feat.device  # 👈 自动获取 device，统一用它

        # 确保 nn_index 也搬到相同的 device 上
        nn_index = self.nn_index.to(device).long()

        # 初始化
        cluster_sums = torch.zeros((self.num_clusters, self.vec_dim), device=device)
        cluster_sizes = torch.zeros(self.num_clusters, device=device)

        # 特征加到对应 cluster 上
        index = nn_index.unsqueeze(1).expand(-1, self.vec_dim)
        cluster_sums = torch.scatter_add(cluster_sums, 0, index, feat)

        # 每个 cluster 的元素数量
        cluster_sizes = torch.bincount(nn_index, minlength=self.num_clusters).float()

        # 防止除以 0
        cluster_centers = cluster_sums / (cluster_sizes.unsqueeze(1) + 1e-6)

        return cluster_centers

        

    # Update centers during cluster assignment using mask matrix multiplication
    # Mask is obtained from distance matrix
    def update_centers_(self, feat, cluster_mask=None, nn_index=None, avg=False):
        feat = feat.detach().reshape(-1, self.vec_dim)
        centers = (cluster_mask.T @ feat)
        if avg:
            self.centers /= counts.unsqueeze(-1)
        return centers
    
    def cluster_assign(self, feat, feat_scaled=None):

        # quantize with kmeans
        feat = feat.detach()
        feat = feat.reshape(-1, self.vec_dim)
        if feat_scaled is None:
            feat_scaled = feat
            scale = feat[0] / (feat_scaled[0] + 1e-8)
        if len(self.centers) == 0:
            self.centers = feat[torch.randperm(feat.shape[0])[:self.num_clusters], :]

        # start kmeans
        chunk = True
        counts = torch.zeros(self.num_clusters, dtype=torch.float32).cuda() + 1e-6
        centers = torch.zeros_like(self.centers)
        for iteration in range(self.num_kmeans_iters):
            # chunk for memory issues
            if chunk:
                self.nn_index = None
                i = 0
                chunk = 10000
                while True:
                    dist = self.get_dist(feat[i*chunk:(i+1)*chunk, :], self.centers)
                    curr_nn_index = torch.argmin(dist, dim=-1)
                    # Assign a single cluster when distance to multiple clusters is same
                    dist = F.one_hot(curr_nn_index, self.num_clusters).type(torch.float32)
                    curr_centers = self.update_centers_(feat[i*chunk:(i+1)*chunk, :], dist, curr_nn_index, avg=False)
                    counts += dist.detach().sum(0) + 1e-6
                    centers += curr_centers
                    if self.nn_index == None:
                        self.nn_index = curr_nn_index
                    else:
                        self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                    i += 1
                    if i*chunk > feat.shape[0]:
                        break

            self.centers = centers / counts.unsqueeze(-1)
            # Reinitialize to 0
            centers[centers != 0] = 0.
            counts[counts > 0.1] = 0.

        if chunk:
            self.nn_index = None
            i = 0
            # chunk = 100000
            while True:
                dist = self.get_dist(feat_scaled[i * chunk:(i + 1) * chunk, :], self.centers)
                curr_nn_index = torch.argmin(dist, dim=-1)
                if self.nn_index == None:
                    self.nn_index = curr_nn_index
                else:
                    self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                i += 1
                if i * chunk > feat.shape[0]:
                    break
        #self.equalize_cluster()
        self.cls_ids = self.nn_index

    def rescale(self, feat, scale=None):
        """Scale the feature to be in the range [-1, 1] by dividing by its max value.

        """
        if scale is None:
            return feat / (abs(feat).max(dim=0)[0] + 1e-8)
        else:
            return feat / (scale + 1e-8)

    def prune(self, valid_indices, feat):
        valid_indices = valid_indices.to(self.nn_index.device)
        self.nn_index = self.nn_index[valid_indices]


    def forward_pos(self, gaussian, assign=False, update_centers_flag=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._xyz.shape[1]
        if assign:
            self.cluster_assign(gaussian._xyz)
        if update_centers_flag:
            self.update_centers(gaussian._xyz)
            sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
            gaussian._xyz_q = sampled_centers.detach()

    def forward_dc(self, gaussian, assign=False, update_centers_flag=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._features_dc.shape[1] * gaussian._features_dc.shape[2]
        if assign:
            self.cluster_assign(gaussian._features_dc)
        if update_centers_flag:
            self.update_centers(gaussian._features_dc)
            sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
            gaussian._features_dc_q = sampled_centers.reshape(-1, 1, 3).detach()

    def forward_frest(self, gaussian, assign=False, update_centers_flag=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._features_rest.shape[1] * gaussian._features_rest.shape[2]
        if assign:
            self.cluster_assign(gaussian._features_rest)
        if update_centers_flag:
            self.update_centers(gaussian._features_rest)
            deg = gaussian._features_rest.shape[1]
            sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
            gaussian._features_rest_q = sampled_centers.reshape(-1, deg, 3).detach()

    def forward_scale(self, gaussian, assign=False, update_centers_flag=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._scaling.shape[1]
        if assign:
            self.cluster_assign(gaussian._scaling)
        if update_centers_flag:
            self.update_centers(gaussian._scaling)
            sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
            gaussian._scaling_q = sampled_centers.detach()

    def forward_rot(self, gaussian, assign=False, update_centers_flag=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._rotation.shape[1]
        if assign:
            self.cluster_assign(gaussian._rotation)
        if update_centers_flag:
            self.update_centers(gaussian._rotation)
            sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
            gaussian._rotation_q = sampled_centers.detach()

    def forward_scale_rot(self, gaussian, assign=False, update_centers_flag=False):
        """Combine both scaling and rotation for a single k-Means"""
        if self.vec_dim == 0:
            self.vec_dim = gaussian._rotation.shape[1] + gaussian._scaling.shape[1]
        feat_scaled = torch.cat([self.rescale(gaussian._scaling), self.rescale(gaussian._rotation)], 1)
        feat = torch.cat([gaussian._scaling, gaussian._rotation], 1)
        if assign:
            self.cluster_assign(feat, feat_scaled)
        if update_centers_flag:
            self.update_centers(feat)
            sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
            gaussian._scaling_q = sampled_centers[:, :3].detach()
            gaussian._rotation_q = sampled_centers[:, 3:].detach()

    def forward_dcfrest(self, gaussian, assign=False, update_centers_flag=False):
        """Combine both features_dc and rest for a single k-Means"""
        if self.vec_dim == 0:
            self.vec_dim = (gaussian._features_rest.shape[1] * gaussian._features_rest.shape[2] +
                            gaussian._features_dc.shape[1] * gaussian._features_dc.shape[2])
        if assign:
            self.cluster_assign(torch.cat([gaussian._features_dc, gaussian._features_rest], 1))
        if update_centers_flag:
            self.update_centers(torch.cat([gaussian._features_dc, gaussian._features_rest], 1))
            deg = gaussian._features_rest.shape[1]
            sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
            gaussian._features_dc_q = sampled_centers[:, :3].reshape(-1, 1, 3).detach()
            gaussian._features_rest_q = sampled_centers[:, 3:].reshape(-1, deg, 3).detach()

    def replace_with_centers(self, gaussian):
        deg = gaussian._features_rest.shape[1]
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_rest = sampled_centers.reshape(-1, deg, 3)

