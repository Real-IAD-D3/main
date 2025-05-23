import torch
from feature_extractors.features import Features
from utils.mvtec3d_util import *
from feature_extractors.change_ex import *
import numpy as np
import math
import os
from sklearn.metrics import roc_auc_score
from utils.au_pro_util import calculate_au_pro
from torchvision import transforms
import cv2
from PIL import Image

class RGBFeatures(Features):

    def add_sample_to_mem_bank(self, sample):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, _, _, center_idx, _ = self(sample[0],unorganized_pc_no_zeros.contiguous())
        # 检查特征是否包含 NaN
        if any(torch.isnan(fm).any() for fm in rgb_feature_maps + xyz_feature_maps):
            print("警告: 提取的特征包含 NaN 值。")

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.patch_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(sample[0], unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.compute_s_s_map(rgb_patch, rgb_feature_maps[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def run_coreset(self):

        self.patch_lib = torch.cat(self.patch_lib, 0)
        self.mean = torch.mean(self.patch_lib)
        self.std = torch.std(self.patch_lib)
        self.patch_lib = (self.patch_lib - self.mean)/self.std

        # self.patch_lib = self.rgb_layernorm(self.patch_lib)

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]


    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx, nonzero_patch_indices = None):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        patch = (patch - self.mean)/self.std

        # self.patch_lib = self.rgb_layernorm(self.patch_lib)
        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

class PointFeatures(Features):

    def add_sample_to_mem_bank(self, sample):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
 
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        self.patch_lib.append(xyz_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        self.compute_s_s_map(xyz_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def run_coreset(self):

        self.patch_lib = torch.cat(self.patch_lib, 0)

        if self.args.rm_zero_for_project:
            self.patch_lib = self.patch_lib[torch.nonzero(torch.all(self.patch_lib!=0, dim=1))[:,0]]

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]
            
        if self.args.rm_zero_for_project:

            self.patch_lib = self.patch_lib[torch.nonzero(torch.all(self.patch_lib!=0, dim=1))[:,0]]
            self.patch_lib = torch.cat((self.patch_lib, torch.zeros(1, self.patch_lib.shape[1])), 0)


    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx, nonzero_patch_indices = None):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''


        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)
        # 处理s中的nan值
        if torch.isnan(s):
            s = torch.tensor(0.0)  # 将nan替换为0
            
        # 处理s_map中的nan值
        s_map = torch.nan_to_num(s_map, 0.0)  # 将所有nan替换为0

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

FUSION_BLOCK= True

class FusionFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, rgb_patch2], dim=1)

        if class_name is not None:
            torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_lib.append(fusion_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, rgb_patch2], dim=1)

        self.compute_s_s_map(fusion_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps)
            self.patch_lib = self.patch_lib[self.coreset_idx]

class DoubleRGBPointFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        patch = torch.cat([xyz_patch, rgb_patch_resize], dim=1)

        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.compute_s_s_map(xyz_patch, rgb_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
    
        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb]])
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        s_map = s_map.view(1, 224, 224)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)
        if torch.isnan(min_val).any():
            print("警告: 计算 min_val 时出现 NaN 值。")
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        if torch.isnan(self.xyz_mean):
            print("警告: 计算 xyz_mean 时出现 NaN 值。")
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

class DoubleRGBPointFeatures_add(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        patch = torch.cat([xyz_patch, rgb_patch_resize], dim=1)

        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.compute_s_s_map(xyz_patch, rgb_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
    
        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[s_xyz, s_rgb]])
        s_map = torch.cat([s_map_xyz, s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = s_xyz + s_rgb
        s_map = s_map_xyz + s_map_rgb
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map

class TripleFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        self.patch_rgb_lib.append(rgb_patch)

        if self.args.asy_memory_bank is None or len(self.patch_xyz_lib) < self.args.asy_memory_bank:

            xyz_patch = torch.cat(xyz_feature_maps, 1)
            xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
            xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
            xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
            xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
            xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

            xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
            xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

            if FUSION_BLOCK:
                with torch.no_grad():
                    fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
                fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
            else:
                fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)

            self.patch_xyz_lib.append(xyz_patch)
            self.patch_fusion_lib.append(fusion_patch)
    

        if class_name is not None:
            torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        
    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    

        self.compute_s_s_map(xyz_patch, rgb_patch, fusion_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    
        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        # 3 memory bank results
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')
        

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.fusion_mean = torch.mean(self.patch_xyz_lib)
        self.fusion_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean)/self.fusion_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]


        self.patch_xyz_lib = self.patch_xyz_lib[torch.nonzero(torch.all(self.patch_xyz_lib!=0, dim=1))[:,0]]
        self.patch_xyz_lib = torch.cat((self.patch_xyz_lib, torch.zeros(1, self.patch_xyz_lib.shape[1])), 0)


    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

  
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)
 
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
  
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        # equation 7 from the paper
        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1)

        # sparse reweight
        # if modal=='rgb':
        #     _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # else:
        #     _, nn_idx = torch.topk(w_dist, k=4*self.n_reweight, largest=False)  # pt.2

        # if modal=='xyz':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1::4]], dim=1) 
        # elif modal=='rgb':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        # else:
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1::4]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))

        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map
class DoubleRGB_PS_Features(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        # if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
        #     print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        print(unorganized_pc.shape)
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        print(unorganized_pc_no_zeros.shape)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        
        rgb_feature_maps = [(rgb + ps) / 2 for rgb, ps in zip(rgb_feature_maps, ps_feature_maps)]
        # 防止数值爆炸的后处理
        rgb_feature_maps = [torch.clamp(fm, min=0, max=1) for fm in rgb_feature_maps]
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        patch = torch.cat([xyz_patch, rgb_patch_resize], dim=1)

        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        
        rgb_feature_maps = [(rgb + ps) / 2 for rgb, ps in zip(rgb_feature_maps, ps_feature_maps)]
        # 防止数值爆炸的后处理
        rgb_feature_maps = [torch.clamp(fm, min=0, max=1) for fm in rgb_feature_maps]
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.compute_s_s_map(xyz_patch, rgb_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        
        rgb_feature_maps = [(rgb + ps) / 2 for rgb, ps in zip(rgb_feature_maps, ps_feature_maps)]
        # 防止数值爆炸的后处理
        rgb_feature_maps = [torch.clamp(fm, min=0, max=1) for fm in rgb_feature_maps]
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
    
        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb]])
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        s_map = s_map.view(1, 224, 224)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)
        if torch.isnan(min_val).any():
            print("警告: 计算 min_val 时出现 NaN 值。")
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        if torch.isnan(self.xyz_mean):
            print("警告: 计算 xyz_mean 时出现 NaN 值。")
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

class FusionFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, rgb_patch2], dim=1)

        if class_name is not None:
            torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_lib.append(fusion_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, rgb_patch2], dim=1)

        self.compute_s_s_map(fusion_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps)
            self.patch_lib = self.patch_lib[self.coreset_idx]


class TripleFeatures_PS(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        
        rgb_feature_maps = [(rgb + ps) / 2 for rgb, ps in zip(rgb_feature_maps, ps_feature_maps)]
        # 防止数值爆炸的后处理
        rgb_feature_maps = [torch.clamp(fm, min=0, max=1) for fm in rgb_feature_maps]
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        self.patch_rgb_lib.append(rgb_patch)
        
        if self.args.asy_memory_bank is None or len(self.patch_xyz_lib) < self.args.asy_memory_bank:

            xyz_patch = torch.cat(xyz_feature_maps, 1)
            xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
            xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
            xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
            xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
            xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

            xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
            xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

            if FUSION_BLOCK:
                with torch.no_grad():
                    print(xyz_patch2.shape)
                    print(rgb_patch2.shape)
                    fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
                fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
            else:
                fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)

            self.patch_xyz_lib.append(xyz_patch)
            self.patch_fusion_lib.append(fusion_patch)
    

        if class_name is not None:
            torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        
    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        
        rgb_feature_maps = [(rgb + ps) / 2 for rgb, ps in zip(rgb_feature_maps, ps_feature_maps)]
        # 防止数值爆炸的后处理
        rgb_feature_maps = [torch.clamp(fm, min=0, max=1) for fm in rgb_feature_maps]
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    

        self.compute_s_s_map(xyz_patch, rgb_patch, fusion_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        
        rgb_feature_maps = [(rgb + ps) / 2 for rgb, ps in zip(rgb_feature_maps, ps_feature_maps)]
        # 防止数值爆炸的后处理
        rgb_feature_maps = [torch.clamp(fm, min=0, max=1) for fm in rgb_feature_maps]
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    
        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        # 3 memory bank results
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')
        

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.fusion_mean = torch.mean(self.patch_xyz_lib)
        self.fusion_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean)/self.fusion_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]


        self.patch_xyz_lib = self.patch_xyz_lib[torch.nonzero(torch.all(self.patch_xyz_lib!=0, dim=1))[:,0]]
        self.patch_xyz_lib = torch.cat((self.patch_xyz_lib, torch.zeros(1, self.patch_xyz_lib.shape[1])), 0)


    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

  
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)
        s = torch.nan_to_num(s, nan=0.0)
        s = torch.tensor(self.detect_fuser.score_samples(s))
        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
  
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        # equation 7 from the paper
        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1)

        # sparse reweight
        # if modal=='rgb':
        #     _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # else:
        #     _, nn_idx = torch.topk(w_dist, k=4*self.n_reweight, largest=False)  # pt.2

        # if modal=='xyz':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1::4]], dim=1) 
        # elif modal=='rgb':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        # else:
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1::4]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))

        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map
    
class PSRGBPointFeatures_add(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        ps_patch_resize = ps_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        patch = torch.cat([xyz_patch, rgb_patch_resize,ps_patch_resize], dim=1)

        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)
        self.patch_ps_lib.append(ps_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        self.compute_s_s_map(xyz_patch, rgb_patch, ps_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T
    
        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')

        s = torch.tensor([[s_xyz, s_rgb,s_ps]])
        s_map = torch.cat([s_map_xyz, s_map_rgb,s_map_ps], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_ps_lib = torch.cat(self.patch_ps_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib)
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.ps_mean = torch.mean(self.patch_ps_lib)
        self.ps_std = torch.std(self.patch_ps_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_ps_lib = (self.patch_ps_lib - self.ps_mean)/self.ps_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_ps_lib,
                                                            n=int(self.f_coreset * self.patch_ps_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_ps_lib = self.patch_ps_lib[self.coreset_idx]

    def compute_s_s_map(self, xyz_patch, rgb_patch, ps_patch,feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')

        s = s_xyz + s_rgb + s_ps
        s_map = s_map_xyz + s_map_rgb + s_map_ps
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_ps_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_ps_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_ps_lib[nn_idx[0, 1:]], dim=1)

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map
    
class TripleFeatures_PS2(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        ps = sample[2]
        rgb = sample[1]
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(ps+rgb,unorganized_pc_no_zeros.contiguous())
        
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        self.patch_rgb_lib.append(rgb_patch)
        
        if self.args.asy_memory_bank is None or len(self.patch_xyz_lib) < self.args.asy_memory_bank:

            xyz_patch = torch.cat(xyz_feature_maps, 1)
            xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
            xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
            xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
            xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
            xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

            xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
            xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

            if FUSION_BLOCK:
                with torch.no_grad():
                    print(xyz_patch2.shape)
                    print(rgb_patch2.shape)
                    fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
                fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
            else:
                fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)

            self.patch_xyz_lib.append(xyz_patch)
            self.patch_fusion_lib.append(fusion_patch)
    

        if class_name is not None:
            torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        
    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        ps = sample[2]
        rgb = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(ps+rgb,unorganized_pc_no_zeros.contiguous())

        
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    

        self.compute_s_s_map(xyz_patch, rgb_patch, fusion_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        ps = sample[2]
        rgb = sample[1]
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(ps+rgb,unorganized_pc_no_zeros.contiguous())
        
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    
        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        # 3 memory bank results
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')
        

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.fusion_mean = torch.mean(self.patch_xyz_lib)
        self.fusion_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean)/self.fusion_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]


        self.patch_xyz_lib = self.patch_xyz_lib[torch.nonzero(torch.all(self.patch_xyz_lib!=0, dim=1))[:,0]]
        self.patch_xyz_lib = torch.cat((self.patch_xyz_lib, torch.zeros(1, self.patch_xyz_lib.shape[1])), 0)


    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

  
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)
 
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
  
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        # equation 7 from the paper
        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1)

        # sparse reweight
        # if modal=='rgb':
        #     _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # else:
        #     _, nn_idx = torch.topk(w_dist, k=4*self.n_reweight, largest=False)  # pt.2

        # if modal=='xyz':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1::4]], dim=1) 
        # elif modal=='rgb':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        # else:
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1::4]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))

        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map
    

class FourFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        mix_feature_maps = [(rgb + ps) / 2 for rgb, ps in zip(rgb_feature_maps, ps_feature_maps)]
        # 防止数值爆炸的后处理
        mix_feature_maps = [torch.clamp(fm, min=0, max=1) for fm in rgb_feature_maps]
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        ps_patch_size = int(math.sqrt(ps_patch.shape[0]))
        ps_patch2 = self.resize2(ps_patch.permute(1, 0).reshape(-1, ps_patch_size, ps_patch_size))
        ps_patch2 = ps_patch2.reshape(ps_patch.shape[1], -1).T

        mix_patch = torch.cat(mix_feature_maps, 1)
        mix_patch = mix_patch.reshape(mix_patch.shape[1], -1).T

        mix_patch_size = int(math.sqrt(mix_patch.shape[0]))
        mix_patch2 = self.resize2(mix_patch.permute(1, 0).reshape(-1, mix_patch_size, mix_patch_size))
        mix_patch2 = mix_patch2.reshape(mix_patch.shape[1], -1).T

        self.patch_rgb_lib.append(rgb_patch)
        self.patch_ps_lib.append(ps_patch)

        if self.args.asy_memory_bank is None or len(self.patch_xyz_lib) < self.args.asy_memory_bank:

            xyz_patch = torch.cat(xyz_feature_maps, 1)
            xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
            xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
            xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
            xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
            xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

            xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
            xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

            if FUSION_BLOCK:
                with torch.no_grad():
                    fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), mix_patch2.unsqueeze(0))
                fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
            else:
                fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)

            self.patch_xyz_lib.append(xyz_patch)
            self.patch_fusion_lib.append(fusion_patch)
    

        if class_name is not None:
            torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        
    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        mix_feature_maps = [(rgb + ps) / 2 for rgb, ps in zip(rgb_feature_maps, ps_feature_maps)]
        # 防止数值爆炸的后处理
        mix_feature_maps = [torch.clamp(fm, min=0, max=1) for fm in rgb_feature_maps]
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T
        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        ps_patch_size = int(math.sqrt(ps_patch.shape[0]))
        ps_patch2 = self.resize2(ps_patch.permute(1, 0).reshape(-1, ps_patch_size, ps_patch_size))
        ps_patch2 = ps_patch2.reshape(ps_patch.shape[1], -1).T

        mix_patch = torch.cat(mix_feature_maps, 1)
        mix_patch = mix_patch.reshape(mix_patch.shape[1], -1).T

        mix_patch_size = int(math.sqrt(mix_patch.shape[0]))
        mix_patch2 = self.resize2(mix_patch.permute(1, 0).reshape(-1, mix_patch_size, mix_patch_size))
        mix_patch2 = mix_patch2.reshape(mix_patch.shape[1], -1).T
        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), mix_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    

        self.compute_s_s_map(xyz_patch, rgb_patch, ps_patch,fusion_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        mix_feature_maps = [(rgb + ps) / 2 for rgb, ps in zip(rgb_feature_maps, ps_feature_maps)]
        # 防止数值爆炸的后处理
        mix_feature_maps = [torch.clamp(fm, min=0, max=1) for fm in rgb_feature_maps]
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T
        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        ps_patch_size = int(math.sqrt(ps_patch.shape[0]))
        ps_patch2 = self.resize2(ps_patch.permute(1, 0).reshape(-1, ps_patch_size, ps_patch_size))
        ps_patch2 = ps_patch2.reshape(ps_patch.shape[1], -1).T

        mix_patch = torch.cat(mix_feature_maps, 1)
        mix_patch = mix_patch.reshape(mix_patch.shape[1], -1).T

        mix_patch_size = int(math.sqrt(mix_patch.shape[0]))
        mix_patch2 = self.resize2(mix_patch.permute(1, 0).reshape(-1, mix_patch_size, mix_patch_size))
        mix_patch2 = mix_patch2.reshape(mix_patch.shape[1], -1).T
        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), mix_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    
        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        # 3 memory bank results
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')
        

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.ps_s_lambda*s_ps, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.ps_smap_lambda*s_map_ps,self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(4, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_ps_lib = torch.cat(self.patch_ps_lib, 0)
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib)
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.ps_mean = torch.mean(self.patch_ps_lib)
        self.ps_std = torch.std(self.patch_ps_lib)
        self.fusion_mean = torch.mean(self.patch_fusion_lib)
        self.fusion_std = torch.std(self.patch_fusion_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_ps_lib = (self.patch_ps_lib - self.ps_mean)/self.ps_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean)/self.fusion_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_ps_lib,
                                                            n=int(self.f_coreset * self.patch_ps_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_ps_lib = self.patch_ps_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_fusion_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]


        self.patch_xyz_lib = self.patch_xyz_lib[torch.nonzero(torch.all(self.patch_xyz_lib!=0, dim=1))[:,0]]
        self.patch_xyz_lib = torch.cat((self.patch_xyz_lib, torch.zeros(1, self.patch_xyz_lib.shape[1])), 0)


    def compute_s_s_map(self, xyz_patch, rgb_patch,ps_patch, fusion_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

  
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')
        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.ps_s_lambda*s_ps, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.ps_smap_lambda*s_map_ps, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(4, -1).permute(1, 0)
 
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
  
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        elif modal=='ps':
            m_star = self.patch_ps_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_ps_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        # equation 7 from the paper
        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        elif modal=='ps':
            m_star_knn = torch.linalg.norm(m_test - self.patch_ps_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1)

        # sparse reweight
        # if modal=='rgb':
        #     _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # else:
        #     _, nn_idx = torch.topk(w_dist, k=4*self.n_reweight, largest=False)  # pt.2

        # if modal=='xyz':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1::4]], dim=1) 
        # elif modal=='rgb':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        # else:
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1::4]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))

        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map
    
class TripleFeatures_PS_EX(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        ps = sample[2]
        rgb = sample[1]
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(rgb,unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(ps,unorganized_pc_no_zeros.contiguous())
        print(len(ps_feature_maps))
        print(ps_feature_maps[0].shape)
        exchange_module = CombinedExchange(channel_p=50, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        ps_patch_size = int(math.sqrt(ps_patch.shape[0]))
        ps_patch2 =  self.resize2(ps_patch.permute(1, 0).reshape(-1, ps_patch_size, ps_patch_size))
        ps_patch2 = ps_patch2.reshape(ps_patch.shape[1], -1).T

        self.patch_rgb_lib.append(rgb_patch)
        self.patch_ps_lib.append(ps_patch)

        if self.args.asy_memory_bank is None or len(self.patch_xyz_lib) < self.args.asy_memory_bank:

            xyz_patch = torch.cat(xyz_feature_maps, 1)
            xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
            xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
            xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
            xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
            xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

            xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
            xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

            # if FUSION_BLOCK:
            #     with torch.no_grad():
            #         print(xyz_patch2.shape)
            #         print(rgb_patch2.shape)
            #         fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            #     fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
            # else:
            #     fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
            patch = torch.cat([xyz_patch2, rgb_patch2,ps_patch2], dim=1)

            if class_name is not None:
                torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1
            self.patch_xyz_lib.append(xyz_patch)
            #self.patch_fusion_lib.append(fusion_patch)
    


        
    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        ps = sample[2]
        rgb = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(rgb,unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(ps,unorganized_pc_no_zeros.contiguous())
        print(ps_feature_maps[0].shape)
        exchange_module = CombinedExchange(channel_p=50, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T
        ps_patch_size = int(math.sqrt(ps_patch.shape[0]))
        ps_patch2 = self.resize2(ps_patch.permute(1, 0).reshape(-1, ps_patch_size, ps_patch_size))
        ps_patch2 = ps_patch2.reshape(ps_patch.shape[1], -1).T

        # if FUSION_BLOCK:
        #     with torch.no_grad():
        #         fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
        #     fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        # else:
        #     fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    

        self.compute_s_s_map(xyz_patch, rgb_patch, ps_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        ps = sample[2]
        rgb = sample[1]
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(rgb,unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(ps,unorganized_pc_no_zeros.contiguous())
        exchange_module = CombinedExchange(channel_p=50, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T
        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T
        ps_patch_size = int(math.sqrt(ps_patch.shape[0]))
        ps_patch2 = self.resize2(ps_patch.permute(1, 0).reshape(-1, ps_patch_size, ps_patch_size))
        ps_patch2 = ps_patch2.reshape(ps_patch.shape[1], -1).T

        # if FUSION_BLOCK:
        #     with torch.no_grad():
        #         fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
        #     fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        # else:
        #     fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    
        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        ps_feat_size =  (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))

        # 3 memory bank results
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')
        

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.ps_s_lambda*s_ps]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.ps_smap_lambda*s_map_ps], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_ps_lib = torch.cat(self.patch_ps_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.ps_mean = torch.mean(self.patch_xyz_lib)
        self.ps_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_ps_lib = (self.patch_ps_lib - self.ps_mean)/self.ps_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_ps_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_ps_lib = self.patch_ps_lib[self.coreset_idx]


        self.patch_xyz_lib = self.patch_xyz_lib[torch.nonzero(torch.all(self.patch_xyz_lib!=0, dim=1))[:,0]]
        self.patch_xyz_lib = torch.cat((self.patch_xyz_lib, torch.zeros(1, self.patch_xyz_lib.shape[1])), 0)


    def compute_s_s_map(self, xyz_patch, rgb_patch, ps_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        ps_feat_size =  (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))

  
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.ps_s_lambda*s_ps]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.ps_smap_lambda*s_map_ps], dim=0).squeeze().reshape(3, -1).permute(1, 0)
 
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
  
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_ps_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_ps_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        # equation 7 from the paper
        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_ps_lib[nn_idx[0, 1:]], dim=1)

        # sparse reweight
        # if modal=='rgb':
        #     _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # else:
        #     _, nn_idx = torch.topk(w_dist, k=4*self.n_reweight, largest=False)  # pt.2

        # if modal=='xyz':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1::4]], dim=1) 
        # elif modal=='rgb':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        # else:
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1::4]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))

        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map

class PSRGBPointFeatures_add_EX(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        exchange_module = CombinedExchange(channel_p=50, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        ps_patch_resize = ps_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        patch = torch.cat([xyz_patch, rgb_patch_resize,ps_patch_resize], dim=1)

        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)
        self.patch_ps_lib.append(ps_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        exchange_module = CombinedExchange(channel_p=50, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        self.compute_s_s_map(xyz_patch, rgb_patch, ps_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        exchange_module = CombinedExchange(channel_p=50, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T
    
        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')

        s = torch.tensor([[s_xyz, s_rgb,s_ps]])
        s_map = torch.cat([s_map_xyz, s_map_rgb,s_map_ps], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_ps_lib = torch.cat(self.patch_ps_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib)
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.ps_mean = torch.mean(self.patch_ps_lib)
        self.ps_std = torch.std(self.patch_ps_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_ps_lib = (self.patch_ps_lib - self.ps_mean)/self.ps_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_ps_lib,
                                                            n=int(self.f_coreset * self.patch_ps_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_ps_lib = self.patch_ps_lib[self.coreset_idx]

    def compute_s_s_map(self, xyz_patch, rgb_patch, ps_patch,feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')

        s = s_xyz + s_rgb + s_ps
        s_map = s_map_xyz + s_map_rgb + s_map_ps
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_ps_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_ps_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_ps_lib[nn_idx[0, 1:]], dim=1)

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map
    
class PSRGBPointFeatures_add_EX_VS(Features):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_name = None
        self.save_dir = None
        self.image_count = 0
        
    def save_images(self, sample, mask, s_map):
        img, depth_map, ps = sample
        
        # 处理RGB图像
        # 处理RGB图像
        if isinstance(img, torch.Tensor):
        # 检查维度
            if img.dim() == 4:  # 如果是4维张量 (批次, 通道, 高度, 宽度)
                img = img.squeeze(0)  # 移除批次维度
        
        # 反标准化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
            img = img * std + mean
        
        # 转换为numpy数组并调整为0-255范围
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = np.array(img)
        
        # 处理mask
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().squeeze()
            # 由于gt_transform包含ToTensor()，值范围已经是0-1，所以我们需要将其转换回0-255
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = np.array(mask).squeeze()
        
        # 处理s_map
        s_map_np = s_map.detach().cpu().squeeze().numpy()
        
        
        # 归一化s_map以便可视化
        s_map_normalized = ((s_map_np - s_map_np.min()) / (s_map_np.max() - s_map_np.min()) * 255).astype(np.uint8)
        
        # 创建热力图
        heatmap = cv2.applyColorMap(s_map_normalized, cv2.COLORMAP_JET)
        
        # 保存图片
        new_dir = os.path.join(self.save_dir, str(self.image_count))
        os.makedirs(new_dir, exist_ok=True)
        
        # 保存RGB图像
        cv2.imwrite(os.path.join(new_dir, 'rgb.png'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        
        # 保存mask
        if np.any(mask_np):
            cv2.imwrite(os.path.join(new_dir, 'mask.png'), mask_np)
        else:
            print("警告：mask为空或全黑，未保存mask.png")
        
        # 保存分割图（热力图）
        cv2.imwrite(os.path.join(new_dir, 'segmentation.png'), heatmap)
        # 保存原始的s_map
        np.save(os.path.join(new_dir, 'original_s_map.npy'), s_map_np)
        self.image_count += 1

    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.cls_name = class_name
        self.save_dir = os.path.join('/workspace/data1/datasets/VS/ours', self.cls_name)
        os.makedirs(self.save_dir, exist_ok=True)
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        exchange_module = CombinedExchange(channel_p=50, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        ps_patch_resize = ps_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        patch = torch.cat([xyz_patch, rgb_patch_resize,ps_patch_resize], dim=1)

        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)
        self.patch_ps_lib.append(ps_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        exchange_module = CombinedExchange(channel_p=50, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        s_map = self.compute_s_s_map(xyz_patch, rgb_patch, ps_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)
        # 只有当label为1时才进行计算和保存
        if label == 1:
            # 计算pixel得分
            pixel_preds = s_map.flatten().detach().cpu().numpy()
            pixel_labels = mask.flatten().numpy()
            pixel_score = roc_auc_score(pixel_labels, pixel_preds)

            # 计算AU-PRO
            gt = mask.detach().cpu().squeeze().numpy()
            prediction = s_map.detach().cpu().squeeze().numpy()
            au_pro, _ = calculate_au_pro([gt], [prediction])

            # 如果满足条件，保存图片
            if pixel_score > 0.95 and au_pro > 0.9:
                self.save_images(sample, mask, s_map)
            
    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        exchange_module = CombinedExchange(channel_p=50, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T
    
        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')

        s = torch.tensor([[s_xyz, s_rgb,s_ps]])
        s_map = torch.cat([s_map_xyz, s_map_rgb,s_map_ps], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_ps_lib = torch.cat(self.patch_ps_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib)
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.ps_mean = torch.mean(self.patch_ps_lib)
        self.ps_std = torch.std(self.patch_ps_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_ps_lib = (self.patch_ps_lib - self.ps_mean)/self.ps_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_ps_lib,
                                                            n=int(self.f_coreset * self.patch_ps_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_ps_lib = self.patch_ps_lib[self.coreset_idx]

    def compute_s_s_map(self, xyz_patch, rgb_patch, ps_patch,feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')

        s = s_xyz + s_rgb + s_ps
        s_map = s_map_xyz + s_map_rgb + s_map_ps
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
        
        return s_map

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_ps_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_ps_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_ps_lib[nn_idx[0, 1:]], dim=1)

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map


class DoubleRGBPointFeatures_uninter_full(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)

        xyz_patch = xyz_patch.view(1, 1152, 32, 32)
        xyz_patch = self.resize(self.average(xyz_patch))
        xyz_patch = xyz_patch.reshape(xyz_patch.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784 * 4, -1)

        patch = torch.cat([xyz_patch, rgb_patch_resize], dim=1)

        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name + str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = torch.cat(xyz_feature_maps, 1)

        xyz_patch = xyz_patch.view(1, 1152, 32, 32)
        xyz_patch = self.resize(self.average(xyz_patch))
        xyz_patch = xyz_patch.reshape(xyz_patch.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.compute_s_s_map(xyz_patch, rgb_patch, xyz_patch[0].shape[-2:], mask, label, center,
                             neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = torch.cat(xyz_feature_maps, 1)

        xyz_patch = xyz_patch.view(1, 1152, 32, 32)
        xyz_patch = self.resize(self.average(xyz_patch))
        xyz_patch = xyz_patch.reshape(xyz_patch.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        # 2D dist
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda * s_xyz, self.args.rgb_s_lambda * s_rgb]])

        s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.rgb_smap_lambda * s_map_rgb],
                          dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, center, neighbour_idx,
                        nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda * s_xyz, self.args.rgb_s_lambda * s_rgb]])
        s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.rgb_smap_lambda * s_map_rgb],
                          dim=0).squeeze().reshape(2, -1).permute(1, 0)

        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        s_map = s_map.view(1, 224, 224)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)
        if torch.isnan(min_val).any():
            print("警告: 计算 min_val 时出现 NaN 值。")
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val) / 1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal == 'xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal == 'xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) / 1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1) / 1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        if torch.isnan(self.xyz_mean):
            print("警告: 计算 xyz_mean 时出现 NaN 值。")
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean) / self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]


class DoubleRGBPointFeatures_PS(Features):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_name = None
        self.save_dir = None
        self.image_count = 0
        
    def save_images(self, sample, mask, s_map, pixel_score, au_pro):
        # 定义分数阶梯
        score_ranges = {
            'excellent': (0.95, 1.0, 0.9, 1.0),  # (pixel_min, pixel_max, au_pro_min, au_pro_max)
            'good': (0.9, 0.95, 0.85, 0.9),
            'fair': (0.85, 0.9, 0.8, 0.85),
            'poor': (0.0, 0.85, 0.0, 0.8)
        }
        
        # 确定当前分数属于哪个范围
        score_level = None
        for level, (p_min, p_max, a_min, a_max) in score_ranges.items():
            if p_min <= pixel_score <= p_max and a_min <= au_pro <= a_max:
                score_level = level
                break
        
        if score_level is None:
            print(f"警告: 无法确定分数等级 (pixel_score={pixel_score:.3f}, au_pro={au_pro:.3f})")
            score_level = 'unknown'
            
        # 构建保存路径
        save_dir = os.path.join(self.save_dir, score_level, str(self.image_count))
        os.makedirs(save_dir, exist_ok=True)
        
        # 处理RGB图像
        if isinstance(img := sample[0], torch.Tensor):
            if img.dim() == 4:
                img = img.squeeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
            img = img * std + mean
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = np.array(img)
        
        # 处理mask
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().squeeze()
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = np.array(mask).squeeze()
        
        # 处理s_map
        s_map_np = s_map.detach().cpu().squeeze().numpy()
        s_map_normalized = ((s_map_np - s_map_np.min()) / (s_map_np.max() - s_map_np.min()) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(s_map_normalized, cv2.COLORMAP_JET)
        
        # 保存图片和分数信息
        cv2.imwrite(os.path.join(save_dir, 'rgb.png'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        if np.any(mask_np):
            cv2.imwrite(os.path.join(save_dir, 'mask.png'), mask_np)
        else:
            print(f"警告：mask为空或全黑，未保存mask.png")
        cv2.imwrite(os.path.join(save_dir, 'segmentation.png'), heatmap)
        np.save(os.path.join(save_dir, 'original_s_map.npy'), s_map_np)
        
        # 保存分数信息
        with open(os.path.join(save_dir, 'scores.txt'), 'w') as f:
            f.write(f"Pixel Score: {pixel_score:.4f}\n")
            f.write(f"AU-PRO Score: {au_pro:.4f}\n")
            f.write(f"Score Level: {score_level}\n")
        
        self.image_count += 1
    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.cls_name = class_name
        self.save_dir = os.path.join('/workspace/data1/datasets/VS/25D3D', self.cls_name)
        os.makedirs(self.save_dir, exist_ok=True)
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        patch = torch.cat([xyz_patch, rgb_patch_resize], dim=1)

        # if class_name is not None:
        #     torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
        #     self.ins_id += 1

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        s_map = self.compute_s_s_map(xyz_patch, rgb_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)
        if label == 1:
            # 计算pixel得分
            pixel_preds = s_map.flatten().detach().cpu().numpy()
            pixel_labels = mask.flatten().numpy()
            pixel_score = roc_auc_score(pixel_labels, pixel_preds)

            # 计算AU-PRO
            gt = mask.detach().cpu().squeeze().numpy()
            prediction = s_map.detach().cpu().squeeze().numpy()
            au_pro, _ = calculate_au_pro([gt], [prediction])

            # 如果满足条件，保存图片
            # if pixel_score > 0.95 and au_pro > 0.9:
            #     self.save_images(sample, mask, s_map)
            self.save_images(sample, mask, s_map, pixel_score, au_pro)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
    
        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb]])
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        s_map = s_map.view(1, 224, 224)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
        
        return s_map

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)
        if torch.isnan(min_val).any():
            print("警告: 计算 min_val 时出现 NaN 值。")
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        if torch.isnan(self.xyz_mean):
            print("警告: 计算 xyz_mean 时出现 NaN 值。")
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

class TripleFeatures_VS(Features):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_name = None
        self.save_dir = None
        self.image_count = 0
        
    def save_images(self, sample, mask, s_map, pixel_score, au_pro):
        # 定义分数阶梯
        score_ranges = {
            'excellent': (0.95, 1.0, 0.9, 1.0),  # (pixel_min, pixel_max, au_pro_min, au_pro_max)
            'good': (0.9, 0.95, 0.85, 0.9),
            'fair': (0.85, 0.9, 0.8, 0.85),
            'poor': (0.0, 0.85, 0.0, 0.8)
        }
        
        # 确定当前分数属于哪个范围
        score_level = None
        for level, (p_min, p_max, a_min, a_max) in score_ranges.items():
            if p_min <= pixel_score <= p_max and a_min <= au_pro <= a_max:
                score_level = level
                break
        
        if score_level is None:
            print(f"警告: 无法确定分数等级 (pixel_score={pixel_score:.3f}, au_pro={au_pro:.3f})")
            score_level = 'unknown'
            
        # 构建保存路径
        save_dir = os.path.join(self.save_dir, score_level, str(self.image_count))
        os.makedirs(save_dir, exist_ok=True)
        
        # 处理RGB图像
        if isinstance(img := sample[0], torch.Tensor):
            if img.dim() == 4:
                img = img.squeeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
            img = img * std + mean
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = np.array(img)
        
        # 处理mask
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().squeeze()
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = np.array(mask).squeeze()
        
        # 处理s_map
        s_map_np = s_map.detach().cpu().squeeze().numpy()
        s_map_normalized = ((s_map_np - s_map_np.min()) / (s_map_np.max() - s_map_np.min()) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(s_map_normalized, cv2.COLORMAP_JET)
        
        # 保存图片和分数信息
        cv2.imwrite(os.path.join(save_dir, 'rgb.png'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        if np.any(mask_np):
            cv2.imwrite(os.path.join(save_dir, 'mask.png'), mask_np)
        else:
            print(f"警告：mask为空或全黑，未保存mask.png")
        cv2.imwrite(os.path.join(save_dir, 'segmentation.png'), heatmap)
        np.save(os.path.join(save_dir, 'original_s_map.npy'), s_map_np)
        
        # 保存分数信息
        with open(os.path.join(save_dir, 'scores.txt'), 'w') as f:
            f.write(f"Pixel Score: {pixel_score:.4f}\n")
            f.write(f"AU-PRO Score: {au_pro:.4f}\n")
            f.write(f"Score Level: {score_level}\n")
        
        self.image_count += 1


    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.cls_name = class_name
        self.save_dir = os.path.join('/workspace/data1/datasets/VS/M3DM', self.cls_name)
        os.makedirs(self.save_dir, exist_ok=True)
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        self.patch_rgb_lib.append(rgb_patch)

        if self.args.asy_memory_bank is None or len(self.patch_xyz_lib) < self.args.asy_memory_bank:

            xyz_patch = torch.cat(xyz_feature_maps, 1)
            xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
            xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
            xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
            xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
            xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

            xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
            xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

            if FUSION_BLOCK:
                with torch.no_grad():
                    fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
                fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
            else:
                fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)

            self.patch_xyz_lib.append(xyz_patch)
            self.patch_fusion_lib.append(fusion_patch)
    

        # if class_name is not None:
        #     torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
        #     self.ins_id += 1

        
    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    

        s_map = self.compute_s_s_map(xyz_patch, rgb_patch, fusion_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)
        
        
        # 只有当label为1时才进行计算和保存
        if label == 1:
            # 计算pixel得分
            pixel_preds = s_map.flatten().detach().cpu().numpy()
            pixel_labels = mask.flatten().numpy()
            pixel_score = roc_auc_score(pixel_labels, pixel_preds)

            # 计算AU-PRO
            gt = mask.detach().cpu().squeeze().numpy()
            prediction = s_map.detach().cpu().squeeze().numpy()
            au_pro, _ = calculate_au_pro([gt], [prediction])

            # 如果满足条件，保存图片
            # if pixel_score > 0.95 and au_pro > 0.9:
            #     self.save_images(sample, mask, s_map)
            self.save_images(sample, mask, s_map, pixel_score, au_pro)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    
        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        # 3 memory bank results
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')
        

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.fusion_mean = torch.mean(self.patch_xyz_lib)
        self.fusion_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean)/self.fusion_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]


        self.patch_xyz_lib = self.patch_xyz_lib[torch.nonzero(torch.all(self.patch_xyz_lib!=0, dim=1))[:,0]]
        self.patch_xyz_lib = torch.cat((self.patch_xyz_lib, torch.zeros(1, self.patch_xyz_lib.shape[1])), 0)


    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

  
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)
 
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
  
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
        
        return s_map

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        # equation 7 from the paper
        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1)

        # sparse reweight
        # if modal=='rgb':
        #     _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # else:
        #     _, nn_idx = torch.topk(w_dist, k=4*self.n_reweight, largest=False)  # pt.2

        # if modal=='xyz':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1::4]], dim=1) 
        # elif modal=='rgb':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        # else:
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1::4]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))

        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map

class RGBFeatures_VS(Features):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_name = None
        self.save_dir = None
        self.image_count = 0
        
    def save_images(self, sample, mask, s_map, pixel_score, au_pro):
        # 定义分数阶梯
        score_ranges = {
            'excellent': (0.95, 1.0, 0.9, 1.0),  # (pixel_min, pixel_max, au_pro_min, au_pro_max)
            'good': (0.9, 0.95, 0.85, 0.9),
            'fair': (0.85, 0.9, 0.8, 0.85),
            'poor': (0.0, 0.85, 0.0, 0.8)
        }
        
        # 确定当前分数属于哪个范围
        score_level = None
        for level, (p_min, p_max, a_min, a_max) in score_ranges.items():
            if p_min <= pixel_score <= p_max and a_min <= au_pro <= a_max:
                score_level = level
                break
        
        if score_level is None:
            print(f"警告: 无法确定分数等级 (pixel_score={pixel_score:.3f}, au_pro={au_pro:.3f})")
            score_level = 'unknown'
            
        # 构建保存路径
        save_dir = os.path.join(self.save_dir, score_level, str(self.image_count))
        os.makedirs(save_dir, exist_ok=True)
        
        # 处理RGB图像
        if isinstance(img := sample[0], torch.Tensor):
            if img.dim() == 4:
                img = img.squeeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
            img = img * std + mean
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = np.array(img)
        # 处理PS图像
        if isinstance(ps := sample[2], torch.Tensor):
            if ps.dim() == 4:
                ps = ps.squeeze(0)
            ps_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(ps.device)
            ps_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(ps.device)
            ps = ps * ps_std + ps_mean
            ps_np = (ps.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            ps_np = np.array(ps)
        # 处理mask
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().squeeze()
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = np.array(mask).squeeze()
        
        # 处理s_map
        s_map_np = s_map.detach().cpu().squeeze().numpy()
        s_map_normalized = ((s_map_np - s_map_np.min()) / (s_map_np.max() - s_map_np.min()) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(s_map_normalized, cv2.COLORMAP_JET)
        
        # 保存图片和分数信息
        cv2.imwrite(os.path.join(save_dir, 'rgb.png'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, 'ps.png'), cv2.cvtColor(ps_np, cv2.COLOR_RGB2BGR))
        if np.any(mask_np):
            cv2.imwrite(os.path.join(save_dir, 'mask.png'), mask_np)
        else:
            print(f"警告：mask为空或全黑，未保存mask.png")
        cv2.imwrite(os.path.join(save_dir, 'segmentation.png'), heatmap)
        np.save(os.path.join(save_dir, 'original_s_map.npy'), s_map_np)
        
        # 保存分数信息
        with open(os.path.join(save_dir, 'scores.txt'), 'w') as f:
            f.write(f"Pixel Score: {pixel_score:.4f}\n")
            f.write(f"AU-PRO Score: {au_pro:.4f}\n")
            f.write(f"Score Level: {score_level}\n")
        
        self.image_count += 1

    def add_sample_to_mem_bank(self, sample,class_name = None):
        self.cls_name = class_name
        self.save_dir = os.path.join('/workspace/data1/datasets/VS/new2D', self.cls_name)
        os.makedirs(self.save_dir, exist_ok=True)
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, _, _, center_idx, _ = self(sample[0],unorganized_pc_no_zeros.contiguous())
        # 检查特征是否包含 NaN
        if any(torch.isnan(fm).any() for fm in rgb_feature_maps + xyz_feature_maps):
            print("警告: 提取的特征包含 NaN 值。")

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.patch_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(sample[0], unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        s_map = self.compute_s_s_map(rgb_patch, rgb_feature_maps[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)
        # 只有当label为1时才进行计算和保存
        if label == 1:
            # 计算pixel得分
            pixel_preds = s_map.flatten().detach().cpu().numpy()
            pixel_labels = mask.flatten().numpy()
            pixel_score = roc_auc_score(pixel_labels, pixel_preds)

            # 计算AU-PRO
            gt = mask.detach().cpu().squeeze().numpy()
            prediction = s_map.detach().cpu().squeeze().numpy()
            au_pro, _ = calculate_au_pro([gt], [prediction])

            # 如果满足条件，保存图片
            # if pixel_score > 0.95 and au_pro > 0.9:
            #     self.save_images(sample, mask, s_map)
            self.save_images(sample, mask, s_map, pixel_score, au_pro)

    def run_coreset(self):

        self.patch_lib = torch.cat(self.patch_lib, 0)
        self.mean = torch.mean(self.patch_lib)
        self.std = torch.std(self.patch_lib)
        self.patch_lib = (self.patch_lib - self.mean)/self.std

        # self.patch_lib = self.rgb_layernorm(self.patch_lib)

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]


    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx, nonzero_patch_indices = None):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        patch = (patch - self.mean)/self.std

        # self.patch_lib = self.rgb_layernorm(self.patch_lib)
        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
        
        return s_map

class PointFeatures_VS(Features):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_name = None
        self.save_dir = None
        self.image_count = 0
        
    def save_images(self, sample, mask, s_map, pixel_score, au_pro):
        # 定义分数阶梯
        score_ranges = {
            'excellent': (0.95, 1.0, 0.9, 1.0),  # (pixel_min, pixel_max, au_pro_min, au_pro_max)
            'good': (0.9, 0.95, 0.85, 0.9),
            'fair': (0.85, 0.9, 0.8, 0.85),
            'poor': (0.0, 0.85, 0.0, 0.8)
        }
        
        # 确定当前分数属于哪个范围
        score_level = None
        for level, (p_min, p_max, a_min, a_max) in score_ranges.items():
            if p_min <= pixel_score <= p_max and a_min <= au_pro <= a_max:
                score_level = level
                break
        
        if score_level is None:
            print(f"警告: 无法确定分数等级 (pixel_score={pixel_score:.3f}, au_pro={au_pro:.3f})")
            score_level = 'unknown'
            
        # 构建保存路径
        save_dir = os.path.join(self.save_dir, score_level, str(self.image_count))
        os.makedirs(save_dir, exist_ok=True)
        
        # 处理RGB图像
        if isinstance(img := sample[0], torch.Tensor):
            if img.dim() == 4:
                img = img.squeeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
            img = img * std + mean
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = np.array(img)
        
        # 处理mask
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().squeeze()
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = np.array(mask).squeeze()
        
        # 处理s_map
        s_map_np = s_map.detach().cpu().squeeze().numpy()
        s_map_normalized = ((s_map_np - s_map_np.min()) / (s_map_np.max() - s_map_np.min()) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(s_map_normalized, cv2.COLORMAP_JET)
        
        # 保存图片和分数信息
        cv2.imwrite(os.path.join(save_dir, 'rgb.png'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        if np.any(mask_np):
            cv2.imwrite(os.path.join(save_dir, 'mask.png'), mask_np)
        else:
            print(f"警告：mask为空或全黑，未保存mask.png")
        cv2.imwrite(os.path.join(save_dir, 'segmentation.png'), heatmap)
        np.save(os.path.join(save_dir, 'original_s_map.npy'), s_map_np)
        
        # 保存分数信息
        with open(os.path.join(save_dir, 'scores.txt'), 'w') as f:
            f.write(f"Pixel Score: {pixel_score:.4f}\n")
            f.write(f"AU-PRO Score: {au_pro:.4f}\n")
            f.write(f"Score Level: {score_level}\n")
        
        self.image_count += 1

    def add_sample_to_mem_bank(self, sample,class_name = None):
        self.cls_name = class_name
        self.save_dir = os.path.join('/workspace/data1/datasets/VS/3D', self.cls_name)
        os.makedirs(self.save_dir, exist_ok=True)
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
 
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        self.patch_lib.append(xyz_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        s_map = self.compute_s_s_map(xyz_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)
        # 只有当label为1时才进行计算和保存
        if label == 1:
            # 计算pixel得分
            pixel_preds = s_map.flatten().detach().cpu().numpy()
            pixel_labels = mask.flatten().numpy()
            pixel_score = roc_auc_score(pixel_labels, pixel_preds)

            # 计算AU-PRO
            gt = mask.detach().cpu().squeeze().numpy()
            prediction = s_map.detach().cpu().squeeze().numpy()
            au_pro, _ = calculate_au_pro([gt], [prediction])

            # 如果满足条件，保存图片
            # if pixel_score > 0.95 and au_pro > 0.9:
            #     self.save_images(sample, mask, s_map)
            self.save_images(sample, mask, s_map, pixel_score, au_pro)

    def run_coreset(self):

        self.patch_lib = torch.cat(self.patch_lib, 0)

        if self.args.rm_zero_for_project:
            self.patch_lib = self.patch_lib[torch.nonzero(torch.all(self.patch_lib!=0, dim=1))[:,0]]

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]
            
        if self.args.rm_zero_for_project:

            self.patch_lib = self.patch_lib[torch.nonzero(torch.all(self.patch_lib!=0, dim=1))[:,0]]
            self.patch_lib = torch.cat((self.patch_lib, torch.zeros(1, self.patch_lib.shape[1])), 0)


    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx, nonzero_patch_indices = None):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''


        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)
        # 处理s中的nan值
        if torch.isnan(s):
            s = torch.tensor(0.0)  # 将nan替换为0
            
        # 处理s_map中的nan值
        s_map = torch.nan_to_num(s_map, 0.0)  # 将所有nan替换为0

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
        
        return s_map
class DoubleRGBPointFeatures_VS(Features):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_name = None
        self.save_dir = None
        self.image_count = 0
        
    def save_images(self, sample, mask, s_map, pixel_score, au_pro):
        # 定义分数阶梯
        score_ranges = {
            'excellent': (0.95, 1.0, 0.9, 1.0),  # (pixel_min, pixel_max, au_pro_min, au_pro_max)
            'good': (0.9, 0.95, 0.85, 0.9),
            'fair': (0.85, 0.9, 0.8, 0.85),
            'poor': (0.0, 0.85, 0.0, 0.8)
        }
        
        # 确定当前分数属于哪个范围
        score_level = None
        for level, (p_min, p_max, a_min, a_max) in score_ranges.items():
            if p_min <= pixel_score <= p_max and a_min <= au_pro <= a_max:
                score_level = level
                break
        
        if score_level is None:
            print(f"警告: 无法确定分数等级 (pixel_score={pixel_score:.3f}, au_pro={au_pro:.3f})")
            score_level = 'unknown'
            
        # 构建保存路径
        save_dir = os.path.join(self.save_dir, score_level, str(self.image_count))
        os.makedirs(save_dir, exist_ok=True)
        
        # 处理RGB图像
        if isinstance(img := sample[0], torch.Tensor):
            if img.dim() == 4:
                img = img.squeeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
            img = img * std + mean
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = np.array(img)
        # 处理PS图像
        if isinstance(ps := sample[2], torch.Tensor):
            if ps.dim() == 4:
                ps = ps.squeeze(0)
            ps_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(ps.device)
            ps_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(ps.device)
            ps = ps * ps_std + ps_mean
            ps_np = (ps.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            ps_np = np.array(ps)
        # 处理mask
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().squeeze()
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = np.array(mask).squeeze()
        
        # 处理s_map
        s_map_np = s_map.detach().cpu().squeeze().numpy()
        s_map_normalized = ((s_map_np - s_map_np.min()) / (s_map_np.max() - s_map_np.min()) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(s_map_normalized, cv2.COLORMAP_JET)
        
        # 保存图片和分数信息
        cv2.imwrite(os.path.join(save_dir, 'rgb.png'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, 'ps.png'), cv2.cvtColor(ps_np, cv2.COLOR_RGB2BGR))
        if np.any(mask_np):
            cv2.imwrite(os.path.join(save_dir, 'mask.png'), mask_np)
        else:
            print(f"警告：mask为空或全黑，未保存mask.png")
        cv2.imwrite(os.path.join(save_dir, 'segmentation.png'), heatmap)
        np.save(os.path.join(save_dir, 'original_s_map.npy'), s_map_np)
        
        # 保存分数信息
        with open(os.path.join(save_dir, 'scores.txt'), 'w') as f:
            f.write(f"Pixel Score: {pixel_score:.4f}\n")
            f.write(f"AU-PRO Score: {au_pro:.4f}\n")
            f.write(f"Score Level: {score_level}\n")
        
        self.image_count += 1

    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.cls_name = class_name
        self.save_dir = os.path.join('/workspace/data1/datasets/VS/patchcore2d3d', self.cls_name)
        os.makedirs(self.save_dir, exist_ok=True)
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        patch = torch.cat([xyz_patch, rgb_patch_resize], dim=1)

        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        s_map = self.compute_s_s_map(xyz_patch, rgb_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)
        # 只有当label为1时才进行计算和保存
        if label == 1:
            # 计算pixel得分
            pixel_preds = s_map.flatten().detach().cpu().numpy()
            pixel_labels = mask.flatten().numpy()
            pixel_score = roc_auc_score(pixel_labels, pixel_preds)

            # 计算AU-PRO
            gt = mask.detach().cpu().squeeze().numpy()
            prediction = s_map.detach().cpu().squeeze().numpy()
            au_pro, _ = calculate_au_pro([gt], [prediction])

            # 如果满足条件，保存图片
            # if pixel_score > 0.95 and au_pro > 0.9:
            #     self.save_images(sample, mask, s_map)
            self.save_images(sample, mask, s_map, pixel_score, au_pro)


    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
    
        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb]])
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        s_map = s_map.view(1, 224, 224)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
        
        return s_map

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)
        if torch.isnan(min_val).any():
            print("警告: 计算 min_val 时出现 NaN 值。")
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        if torch.isnan(self.xyz_mean):
            print("警告: 计算 xyz_mean 时出现 NaN 值。")
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

class PS_VS(Features):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_name = None
        self.save_dir = None
        self.image_count = 0
        
    def save_images(self, sample, mask, s_map, pixel_score, au_pro):
        # 定义分数阶梯
        score_ranges = {
            'excellent': (0.95, 1.0, 0.9, 1.0),  # (pixel_min, pixel_max, au_pro_min, au_pro_max)
            'good': (0.9, 0.95, 0.85, 0.9),
            'fair': (0.85, 0.9, 0.8, 0.85),
            'poor': (0.0, 0.85, 0.0, 0.8)
        }
        
        # 确定当前分数属于哪个范围
        score_level = None
        for level, (p_min, p_max, a_min, a_max) in score_ranges.items():
            if p_min <= pixel_score <= p_max and a_min <= au_pro <= a_max:
                score_level = level
                break
        
        if score_level is None:
            print(f"警告: 无法确定分数等级 (pixel_score={pixel_score:.3f}, au_pro={au_pro:.3f})")
            score_level = 'unknown'
            
        # 构建保存路径
        save_dir = os.path.join(self.save_dir, score_level, str(self.image_count))
        os.makedirs(save_dir, exist_ok=True)
        
        # 处理RGB图像
        if isinstance(img := sample[0], torch.Tensor):
            if img.dim() == 4:
                img = img.squeeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
            img = img * std + mean
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = np.array(img)
        
        
        # 处理mask
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().squeeze()
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = np.array(mask).squeeze()
        
        # 处理s_map
        s_map_np = s_map.detach().cpu().squeeze().numpy()
        s_map_normalized = ((s_map_np - s_map_np.min()) / (s_map_np.max() - s_map_np.min()) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(s_map_normalized, cv2.COLORMAP_JET)
        
        # 保存图片和分数信息
        cv2.imwrite(os.path.join(save_dir, 'rgb.png'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        if np.any(mask_np):
            cv2.imwrite(os.path.join(save_dir, 'mask.png'), mask_np)
        else:
            print(f"警告：mask为空或全黑，未保存mask.png")
        cv2.imwrite(os.path.join(save_dir, 'segmentation.png'), heatmap)
        np.save(os.path.join(save_dir, 'original_s_map.npy'), s_map_np)
        
        # 保存分数信息
        with open(os.path.join(save_dir, 'scores.txt'), 'w') as f:
            f.write(f"Pixel Score: {pixel_score:.4f}\n")
            f.write(f"AU-PRO Score: {au_pro:.4f}\n")
            f.write(f"Score Level: {score_level}\n")
        
        self.image_count += 1

    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.cls_name = class_name
        self.save_dir = os.path.join('/workspace/data1/datasets/VS/PS2D', self.cls_name)
        os.makedirs(self.save_dir, exist_ok=True)
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        ps_patch = torch.cat(ps_feature_maps, 1) 
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        ps_patch_resize = ps_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        patch = torch.cat([rgb_patch_resize,ps_patch_resize], dim=1)

        # if class_name is not None:
        #     torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
        #     self.ins_id += 1

        self.patch_ps_lib.append(ps_patch)
        self.patch_rgb_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        ps_patch = torch.cat(ps_feature_maps, 1) 
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        s_map = self.compute_s_s_map(ps_patch, rgb_patch, mask, label)
        # 只有当label为1时才进行计算和保存
        if label == 1:
            # 计算pixel得分
            pixel_preds = s_map.flatten().detach().cpu().numpy()
            pixel_labels = mask.flatten().numpy()
            pixel_score = roc_auc_score(pixel_labels, pixel_preds)

            # 计算AU-PRO
            gt = mask.detach().cpu().squeeze().numpy()
            prediction = s_map.detach().cpu().squeeze().numpy()
            au_pro, _ = calculate_au_pro([gt], [prediction])

            # 如果满足条件，保存图片
            # if pixel_score > 0.95 and au_pro > 0.9:
            #     self.save_images(sample, mask, s_map)
            self.save_images(sample, mask, s_map, pixel_score, au_pro)


    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())


        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        ps_patch = torch.cat(ps_feature_maps, 1) 
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T
    
        # 2D dist 
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))

        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.ps_s_lambda*s_ps, self.args.rgb_s_lambda*s_rgb]])
 
        s_map = torch.cat([self.args.ps_smap_lambda*s_map_ps, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def compute_s_s_map(self, ps_patch, rgb_patch,mask, label):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.ps_s_lambda*s_ps, self.args.rgb_s_lambda*s_rgb]])
        s_map = torch.cat([self.args.ps_smap_lambda*s_map_ps, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        s_map = s_map.view(1, 224, 224)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
        
        return s_map

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='ps'):

        min_val, min_idx = torch.min(dist, dim=1)
        if torch.isnan(min_val).any():
            print("警告: 计算 min_val 时出现 NaN 值。")
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='ps':
            m_star = self.patch_ps_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_ps_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='ps':
            m_star_knn = torch.linalg.norm(m_test - self.patch_ps_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_ps_lib = torch.cat(self.patch_ps_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.ps_mean = torch.mean(self.patch_ps_lib)
        if torch.isnan(self.ps_mean):
            print("警告: 计算 ps_mean 时出现 NaN 值。")
        self.ps_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_ps_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_ps_lib = (self.patch_ps_lib - self.ps_mean)/self.ps_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_ps_lib,
                                                            n=int(self.f_coreset * self.patch_ps_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_ps_lib = self.patch_ps_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_ps_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
FUSION_BLOCK = True
class OURS_EX_VS(Features):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_name = None
        self.save_dir = None
        self.image_count = 0
        
    def save_images(self, sample, mask, s_map, pixel_score, au_pro):
        # 定义分数阶梯
        score_ranges = {
            'excellent': (0.95, 1.0, 0.9, 1.0),  # (pixel_min, pixel_max, au_pro_min, au_pro_max)
            'good': (0.9, 0.95, 0.85, 0.9),
            'fair': (0.85, 0.9, 0.8, 0.85),
            'poor': (0.0, 0.85, 0.0, 0.8)
        }
        
        # 确定当前分数属于哪个范围
        score_level = None
        for level, (p_min, p_max, a_min, a_max) in score_ranges.items():
            if p_min <= pixel_score <= p_max and a_min <= au_pro <= a_max:
                score_level = level
                break
        
        if score_level is None:
            print(f"警告: 无法确定分数等级 (pixel_score={pixel_score:.3f}, au_pro={au_pro:.3f})")
            score_level = 'unknown'
            
        # 构建保存路径
        save_dir = os.path.join(self.save_dir, score_level, str(self.image_count))
        os.makedirs(save_dir, exist_ok=True)
        
        # 处理RGB图像
        if isinstance(img := sample[0], torch.Tensor):
            if img.dim() == 4:
                img = img.squeeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
            img = img * std + mean
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = np.array(img)

        # 处理PS图像
        if isinstance(ps := sample[2], torch.Tensor):
            if ps.dim() == 4:
                ps = ps.squeeze(0)
            ps_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(ps.device)
            ps_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(ps.device)
            ps = ps * ps_std + ps_mean
            ps_np = (ps.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            ps_np = np.array(ps)
        
        
        # 处理mask
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().squeeze()
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = np.array(mask).squeeze()
        
        # 处理s_map
        s_map_np = s_map.detach().cpu().squeeze().numpy()
        s_map_normalized = ((s_map_np - s_map_np.min()) / (s_map_np.max() - s_map_np.min()) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(s_map_normalized, cv2.COLORMAP_JET)
        
        # 保存图片和分数信息
        cv2.imwrite(os.path.join(save_dir, f'rgb_{pixel_score:.3f}_{au_pro:.3f}.png'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f'ps_{pixel_score:.3f}_{au_pro:.3f}.png'), cv2.cvtColor(ps_np, cv2.COLOR_RGB2BGR))
        if np.any(mask_np):
            cv2.imwrite(os.path.join(save_dir, f'mask_{pixel_score:.3f}_{au_pro:.3f}.png'), mask_np)
        else:
            print(f"警告：mask为空或全黑，未保存mask.png")
        cv2.imwrite(os.path.join(save_dir, f'segmentation_{pixel_score:.3f}_{au_pro:.3f}.png'), heatmap)
        np.save(os.path.join(save_dir, f'original_s_map_{pixel_score:.3f}_{au_pro:.3f}.npy'), s_map_np)
        
        # 保存分数信息
        with open(os.path.join(save_dir, 'scores.txt'), 'w') as f:
            f.write(f"Pixel Score: {pixel_score:.4f}\n")
            f.write(f"AU-PRO Score: {au_pro:.4f}\n")
            f.write(f"Score Level: {score_level}\n")
        
        self.image_count += 1

    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.cls_name = class_name
        self.save_dir = os.path.join('/workspace/data1/datasets/VS/True', self.cls_name)
        os.makedirs(self.save_dir, exist_ok=True)
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        fusion_rgb_feature_maps = rgb_feature_maps
        exchange_module = CombinedExchange(channel_p=100, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        ps_patch_resize = ps_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        fusion_rgb_patch = torch.cat(fusion_rgb_feature_maps, 1)
        fusion_rgb_patch = fusion_rgb_patch.reshape(fusion_rgb_patch.shape[1], -1).T
        
        fusion_rgb_patch_resize = fusion_rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        fusion_rgb_patch_size = int(math.sqrt(fusion_rgb_patch.shape[0]))
        fusion_rgb_patch2 =  self.resize2(fusion_rgb_patch.permute(1, 0).reshape(-1, fusion_rgb_patch_size, fusion_rgb_patch_size))
        fusion_rgb_patch2 = fusion_rgb_patch2.reshape(fusion_rgb_patch.shape[1], -1).T
        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), fusion_rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, fusion_rgb_patch2], dim=1)
        # patch = torch.cat([xyz_patch, rgb_patch_resize,ps_patch_resize], dim=1)

        # # if class_name is not None:
        # #     torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
        # #     self.ins_id += 1

        self.patch_fusion_lib.append(fusion_patch)
        self.patch_rgb_lib.append(rgb_patch)
        self.patch_ps_lib.append(ps_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        fusion_rgb_feature_maps = rgb_feature_maps
        exchange_module = CombinedExchange(channel_p=50, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T


        fusion_rgb_patch = torch.cat(fusion_rgb_feature_maps, 1)
        fusion_rgb_patch = fusion_rgb_patch.reshape(fusion_rgb_patch.shape[1], -1).T
        

        fusion_rgb_patch_size = int(math.sqrt(fusion_rgb_patch.shape[0]))
        fusion_rgb_patch2 =  self.resize2(fusion_rgb_patch.permute(1, 0).reshape(-1, fusion_rgb_patch_size, fusion_rgb_patch_size))
        fusion_rgb_patch2 = fusion_rgb_patch2.reshape(fusion_rgb_patch.shape[1], -1).T
        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), fusion_rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, fusion_rgb_patch2], dim=1)

        s_map = self.compute_s_s_map(fusion_patch, rgb_patch, ps_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)
        # 只有当label为1时才进行计算和保存
        if label == 1:
            # 计算pixel得分
            pixel_preds = s_map.flatten().detach().cpu().numpy()
            pixel_labels = mask.flatten().numpy()
            pixel_score = roc_auc_score(pixel_labels, pixel_preds)

            # 计算AU-PRO
            gt = mask.detach().cpu().squeeze().numpy()
            prediction = s_map.detach().cpu().squeeze().numpy()
            au_pro, _ = calculate_au_pro([gt], [prediction])

            # 如果满足条件，保存图片
            self.save_images(sample, mask, s_map, pixel_score, au_pro)
            
    def add_sample_to_late_fusion_mem_bank(self, sample):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        fusion_rgb_feature_maps = rgb_feature_maps
        exchange_module = CombinedExchange(channel_p=50, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T


        fusion_rgb_patch = torch.cat(fusion_rgb_feature_maps, 1)
        fusion_rgb_patch = fusion_rgb_patch.reshape(fusion_rgb_patch.shape[1], -1).T
        

        fusion_rgb_patch_size = int(math.sqrt(fusion_rgb_patch.shape[0]))
        fusion_rgb_patch2 =  self.resize2(fusion_rgb_patch.permute(1, 0).reshape(-1, fusion_rgb_patch_size, fusion_rgb_patch_size))
        fusion_rgb_patch2 = fusion_rgb_patch2.reshape(fusion_rgb_patch.shape[1], -1).T
        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), fusion_rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, fusion_rgb_patch2], dim=1)
    
        # 2D dist 
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')

        s = torch.tensor([[s_fusion, s_rgb,s_ps]])
        s_map = torch.cat([s_map_fusion, s_map_rgb,s_map_ps], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_ps_lib = torch.cat(self.patch_ps_lib, 0)

        self.fusion_mean = torch.mean(self.patch_fusion_lib)
        self.fusion_std = torch.std(self.patch_fusion_lib)
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.ps_mean = torch.mean(self.patch_ps_lib)
        self.ps_std = torch.std(self.patch_ps_lib)

        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean)/self.fusion_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_ps_lib = (self.patch_ps_lib - self.ps_mean)/self.ps_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_fusion_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_ps_lib,
                                                            n=int(self.f_coreset * self.patch_ps_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_ps_lib = self.patch_ps_lib[self.coreset_idx]

    def compute_s_s_map(self, fusion_patch, rgb_patch, ps_patch,feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')

        s = s_fusion + s_rgb + s_ps
        s_map = s_map_fusion + s_map_rgb + s_map_ps
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
        
        return s_map

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='fusion'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='fusion':
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_ps_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_ps_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='fusion':
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_ps_lib[nn_idx[0, 1:]], dim=1)

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map
    
FUSION_BLOCK = True
class NEW_OURS_EX_VS(Features):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_name = None
        self.save_dir = None
        self.image_count = 0
        
    def save_images(self, sample, mask, s_map, pixel_score, au_pro):
        # 定义分数阶梯
        score_ranges = {
            'excellent': (0.95, 1.0, 0.9, 1.0),  # (pixel_min, pixel_max, au_pro_min, au_pro_max)
            'good': (0.9, 0.95, 0.85, 0.9),
            'fair': (0.85, 0.9, 0.8, 0.85),
            'poor': (0.0, 0.85, 0.0, 0.8)
        }
        
        # 确定当前分数属于哪个范围
        score_level = None
        for level, (p_min, p_max, a_min, a_max) in score_ranges.items():
            if p_min <= pixel_score <= p_max and a_min <= au_pro <= a_max:
                score_level = level
                break
        
        if score_level is None:
            print(f"警告: 无法确定分数等级 (pixel_score={pixel_score:.3f}, au_pro={au_pro:.3f})")
            score_level = 'unknown'
            
        # 构建保存路径
        save_dir = os.path.join(self.save_dir, score_level, str(self.image_count))
        os.makedirs(save_dir, exist_ok=True)
        
        # 处理RGB图像
        if isinstance(img := sample[0], torch.Tensor):
            if img.dim() == 4:
                img = img.squeeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
            img = img * std + mean
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = np.array(img)

        # 处理PS图像
        if isinstance(ps := sample[2], torch.Tensor):
            if ps.dim() == 4:
                ps = ps.squeeze(0)
            ps_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(ps.device)
            ps_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(ps.device)
            ps = ps * ps_std + ps_mean
            ps_np = (ps.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            ps_np = np.array(ps)
        
        
        # 处理mask
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().squeeze()
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = np.array(mask).squeeze()
        
        # 处理s_map
        s_map_np = s_map.detach().cpu().squeeze().numpy()
        s_map_normalized = ((s_map_np - s_map_np.min()) / (s_map_np.max() - s_map_np.min()) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(s_map_normalized, cv2.COLORMAP_JET)
        
        # 保存图片和分数信息
        cv2.imwrite(os.path.join(save_dir, f'rgb_{pixel_score:.3f}_{au_pro:.3f}.png'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f'ps_{pixel_score:.3f}_{au_pro:.3f}.png'), cv2.cvtColor(ps_np, cv2.COLOR_RGB2BGR))
        if np.any(mask_np):
            cv2.imwrite(os.path.join(save_dir, f'mask_{pixel_score:.3f}_{au_pro:.3f}.png'), mask_np)
        else:
            print(f"警告：mask为空或全黑，未保存mask.png")
        cv2.imwrite(os.path.join(save_dir, f'segmentation_{pixel_score:.3f}_{au_pro:.3f}.png'), heatmap)
        np.save(os.path.join(save_dir, f'original_s_map_{pixel_score:.3f}_{au_pro:.3f}.npy'), s_map_np)
        
        # 保存分数信息
        with open(os.path.join(save_dir, 'scores.txt'), 'w') as f:
            f.write(f"Pixel Score: {pixel_score:.4f}\n")
            f.write(f"AU-PRO Score: {au_pro:.4f}\n")
            f.write(f"Score Level: {score_level}\n")
        
        self.image_count += 1

    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.cls_name = class_name
        self.save_dir = os.path.join('/workspace/data1/datasets/VS/new_True', self.cls_name)
        os.makedirs(self.save_dir, exist_ok=True)
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        exchange_module = CombinedExchange(channel_p=self.ex_factor, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T

        ps_patch_resize = ps_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)
        ps_patch_size = int(math.sqrt(ps_patch.shape[0]))
        ps_patch2 =  self.resize2(ps_patch.permute(1, 0).reshape(-1, ps_patch_size, ps_patch_size))
        ps_patch2 = ps_patch2.reshape(ps_patch.shape[1], -1).T
        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), ps_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, ps_patch2], dim=1)
        # patch = torch.cat([xyz_patch, rgb_patch_resize,ps_patch_resize], dim=1)
        # patch = torch.cat([xyz_patch,ps_patch_resize], dim=1)

        # if class_name is not None:
        #     torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
        #     self.ins_id += 1

        self.patch_fusion_lib.append(fusion_patch)
        self.patch_rgb_lib.append(rgb_patch)
        self.patch_ps_lib.append(ps_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        exchange_module = CombinedExchange(channel_p=self.ex_factor, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T


        ps_patch_size = int(math.sqrt(ps_patch.shape[0]))
        ps_patch2 =  self.resize2(ps_patch.permute(1, 0).reshape(-1, ps_patch_size, ps_patch_size))
        ps_patch2 = ps_patch2.reshape(ps_patch.shape[1], -1).T
        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), ps_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, ps_patch2], dim=1)

        s_map = self.compute_s_s_map(fusion_patch, rgb_patch, ps_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)
        # # 只有当label为1时才进行计算和保存
        # if label == 1:
        #     # 计算pixel得分
        #     pixel_preds = s_map.flatten().detach().cpu().numpy()
        #     pixel_labels = mask.flatten().numpy()
        #     pixel_score = roc_auc_score(pixel_labels, pixel_preds)

        #     # 计算AU-PRO
        #     gt = mask.detach().cpu().squeeze().numpy()
        #     prediction = s_map.detach().cpu().squeeze().numpy()
        #     au_pro, _ = calculate_au_pro([gt], [prediction])

        #     # 如果满足条件，保存图片
        #     self.save_images(sample, mask, s_map, pixel_score, au_pro)
            
    def add_sample_to_late_fusion_mem_bank(self, sample):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        ps_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],unorganized_pc_no_zeros.contiguous())
        exchange_module = CombinedExchange(channel_p=self.ex_factor, window_size=2)
        for i in range(len(ps_feature_maps)):
            ps_feature_maps[i], rgb_feature_maps[i] = exchange_module(ps_feature_maps[i], rgb_feature_maps[i])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        

        ps_patch = torch.cat(ps_feature_maps, 1)
        ps_patch = ps_patch.reshape(ps_patch.shape[1], -1).T


        ps_patch_size = int(math.sqrt(ps_patch.shape[0]))
        ps_patch2 =  self.resize2(ps_patch.permute(1, 0).reshape(-1, ps_patch_size, ps_patch_size))
        ps_patch2 = ps_patch2.reshape(ps_patch.shape[1], -1).T
        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), ps_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, ps_patch2], dim=1)
    
        # 2D dist 
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')

        s = torch.tensor([[s_fusion, s_rgb,s_ps]])
        s_map = torch.cat([s_map_fusion, s_map_rgb,s_map_ps], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_ps_lib = torch.cat(self.patch_ps_lib, 0)

        self.fusion_mean = torch.mean(self.patch_fusion_lib)
        self.fusion_std = torch.std(self.patch_fusion_lib)
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.ps_mean = torch.mean(self.patch_ps_lib)
        self.ps_std = torch.std(self.patch_ps_lib)

        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean)/self.fusion_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_ps_lib = (self.patch_ps_lib - self.ps_mean)/self.ps_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_fusion_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_ps_lib,
                                                            n=int(self.f_coreset * self.patch_ps_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_ps_lib = self.patch_ps_lib[self.coreset_idx]

    def compute_s_s_map(self, fusion_patch, rgb_patch, ps_patch,feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        ps_patch = (ps_patch - self.ps_mean)/self.ps_std
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_ps = torch.cdist(ps_patch, self.patch_ps_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))
        ps_feat_size = (int(math.sqrt(ps_patch.shape[0])), int(math.sqrt(ps_patch.shape[0])))
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_ps, s_map_ps = self.compute_single_s_s_map(ps_patch, dist_ps, ps_feat_size, modal='ps')

        s = s_fusion + s_rgb + s_ps
        s_map = s_map_fusion + s_map_rgb + s_map_ps
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
        
        return s_map

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='fusion'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='fusion':
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_ps_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_ps_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='fusion':
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_ps_lib[nn_idx[0, 1:]], dim=1)

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map


class TripleFeatures_uninter(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        self.patch_rgb_lib.append(rgb_patch)


        xyz_patch = torch.cat(xyz_feature_maps, 1)

        xyz_patch = xyz_patch.view(1, 1152, 32, 32)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch))
        xyz_patch2 = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T



        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_fusion_lib.append(fusion_patch)

        if class_name is not None:
            torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name + str(self.ins_id) + '.pt'))
            self.ins_id += 1

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)

        xyz_patch = xyz_patch.view(1, 1152, 32, 32)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch))
        xyz_patch2 = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)

        self.compute_s_s_map(xyz_patch, rgb_patch, fusion_patch, xyz_patch_full_resized[0].shape[-2:], mask, label,
                             center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        organized_pc = sample[1]
        if torch.isnan(organized_pc).any() or (organized_pc == 0).all(dim=1).any():
            print("警告: 输入数据包含 NaN 或无效值。")
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)

        xyz_patch = xyz_patch.view(1, 1152, 32, 32)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch))
        xyz_patch2 = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)

        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)

        # 3D dist
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean) / self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        # 3 memory bank results
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size,
                                                             modal='fusion')

        s = torch.tensor(
            [[self.args.xyz_s_lambda * s_xyz, self.args.rgb_s_lambda * s_rgb, self.args.fusion_s_lambda * s_fusion]])

        s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.rgb_smap_lambda * s_map_rgb,
                           self.args.fusion_smap_lambda * s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.fusion_mean = torch.mean(self.patch_xyz_lib)
        self.fusion_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean) / self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean) / self.fusion_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]

        self.patch_xyz_lib = self.patch_xyz_lib[torch.nonzero(torch.all(self.patch_xyz_lib != 0, dim=1))[:, 0]]
        self.patch_xyz_lib = torch.cat((self.patch_xyz_lib, torch.zeros(1, self.patch_xyz_lib.shape[1])), 0)

    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, feature_map_dims, mask, label, center, neighbour_idx,
                        nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 3D dist
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean) / self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size,
                                                             modal='fusion')

        s = torch.tensor(
            [[self.args.xyz_s_lambda * s_xyz, self.args.rgb_s_lambda * s_rgb, self.args.fusion_s_lambda * s_fusion]])

        s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.rgb_smap_lambda * s_map_rgb,
                           self.args.fusion_smap_lambda * s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))

        s_map = s_map.view(1, self.image_size, self.image_size)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        # equation 7 from the paper
        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1)

        # sparse reweight
        # if modal=='rgb':
        #     _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # else:
        #     _, nn_idx = torch.topk(w_dist, k=4*self.n_reweight, largest=False)  # pt.2

        # if modal=='xyz':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1::4]], dim=1)
        # elif modal=='rgb':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        # else:
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1::4]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))

        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map
import torch.nn.functional as F
class TripleFeatures_Shape(Features):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_xyz_lib = []  # 3D形状特征库
        self.patch_rgb_lib = []  # RGB特征库
        self.patch_fusion_lib = [] # 融合特征库
        self.k_neighbors = 5 # KNN邻居数量
        self.temperature = 0.07  # 添加 temperature 参数

    def shape_guided_reconstruction(self, xyz_feat, rgb_feat):
        """形状引导的RGB特征重建"""
        if len(self.patch_xyz_lib) > 0:
            # 获取特征库
            if isinstance(self.patch_xyz_lib, torch.Tensor):
                xyz_lib = self.patch_xyz_lib
                rgb_lib = self.patch_rgb_lib
            else:
                xyz_lib = torch.cat(self.patch_xyz_lib, dim=0)
                rgb_lib = torch.cat(self.patch_rgb_lib, dim=0)
            
            # 标准化特征
            xyz_feat = (xyz_feat - self.xyz_mean) / self.xyz_std
            xyz_lib = (xyz_lib - self.xyz_mean) / self.xyz_std
            
            # 计算形状相似度
            dist_xyz = torch.cdist(xyz_feat, xyz_lib)
            
            # KNN搜索
            k = min(self.k_neighbors, len(rgb_lib))  # 使用实际的rgb_lib长度
            
            # 确保dist_xyz的第二维度(候选集)不超过rgb_lib的大小
            if dist_xyz.shape[1] > len(rgb_lib):
                dist_xyz = dist_xyz[:, :len(rgb_lib)]
                
            _, knn_idx = torch.topk(-dist_xyz, k=k, dim=1)  # 使用负距离找最近的
            
            # 添加安全检查
            if knn_idx.max() >= len(rgb_lib):
                print(f"警告: KNN索引 {knn_idx.max()} 超出范围 {len(rgb_lib)}")
                knn_idx = torch.clamp(knn_idx, 0, len(rgb_lib) - 1)
            
            # 获取对应的RGB特征
            knn_rgb = rgb_lib[knn_idx]
            
            # 计算注意力权重并重建
            weights = F.softmax(-dist_xyz[:, :k]/self.temperature, dim=1)
            guided_rgb = torch.sum(weights.unsqueeze(-1) * knn_rgb, dim=1)
            
            # 残差连接
            guided_rgb = guided_rgb + rgb_feat
        else:
            guided_rgb = rgb_feat
        
        return guided_rgb
    
    def add_sample_to_mem_bank(self, sample, class_name=None):
        # 1. 提取点云和RGB特征
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        # 2. 处理RGB特征
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T


        # 3. 形状引导的特征重建
        if self.args.asy_memory_bank is None or len(self.patch_xyz_lib) < self.args.asy_memory_bank:
            xyz_patch = torch.cat(xyz_feature_maps, 1)
            xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
            xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
            xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
            xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
            xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

            xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
            xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T
            
            # 使用形状特征引导RGB特征
            guided_rgb = self.shape_guided_reconstruction(xyz_patch2, rgb_patch2)
            guided_rgb_patch = guided_rgb
            guided_rgb_patch = guided_rgb_patch.reshape(guided_rgb_patch.shape[1], -1).T
            
            # if FUSION_BLOCK:
            #     fusion_patch = self.fusion_block(xyz_patch2, guided_rgb)
            # else:
            #     fusion_patch = torch.cat([xyz_patch2, guided_rgb], dim=1)

            self.patch_xyz_lib.append(xyz_patch)
            self.patch_rgb_lib.append(rgb_patch)
            self.patch_fusion_lib.append(guided_rgb_patch)

        if class_name is not None:
            torch.save(guided_rgb_patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        
    def predict(self, sample, mask, label):
        # 1. 提取点云和RGB特征
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        # 2. 处理RGB特征
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T


        # 3. 形状引导的特征重建
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T
            
        # 使用形状特征引导RGB特征
        guided_rgb = self.shape_guided_reconstruction(xyz_patch2, rgb_patch2)
        guided_rgb_patch = guided_rgb
        guided_rgb_patch = guided_rgb_patch.reshape(guided_rgb_patch.shape[1], -1).T
            

            # if FUSION_BLOCK:
            #     with torch.no_grad():
            #         fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            #     fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
            # else:
            #     fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    

        self.compute_s_s_map(xyz_patch, rgb_patch, guided_rgb_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):
        # 1. 提取点云和RGB特征
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        # 2. 处理RGB特征
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T


        # 3. 形状引导的特征重建
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T
            
        # 使用形状特征引导RGB特征
        guided_rgb = self.shape_guided_reconstruction(xyz_patch2, rgb_patch2)
        guided_rgb_patch = guided_rgb
        guided_rgb_patch = guided_rgb_patch.reshape(guided_rgb_patch.shape[1], -1).T

        # if FUSION_BLOCK:
        #     with torch.no_grad():
        #         fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
        #     fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        # else:
        #     fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    
        # 3D dist
        fusion_patch = guided_rgb_patch
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        # 3 memory bank results
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')
        

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.fusion_mean = torch.mean(self.patch_xyz_lib)
        self.fusion_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean)/self.fusion_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]


        self.patch_xyz_lib = self.patch_xyz_lib[torch.nonzero(torch.all(self.patch_xyz_lib!=0, dim=1))[:,0]]
        self.patch_xyz_lib = torch.cat((self.patch_xyz_lib, torch.zeros(1, self.patch_xyz_lib.shape[1])), 0)


    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

  
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)
 
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
  
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        # equation 7 from the paper
        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1)

        # sparse reweight
        # if modal=='rgb':
        #     _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # else:
        #     _, nn_idx = torch.topk(w_dist, k=4*self.n_reweight, largest=False)  # pt.2

        # if modal=='xyz':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1::4]], dim=1) 
        # elif modal=='rgb':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        # else:
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1::4]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))

        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map
