import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN

class Model(torch.nn.Module):

    def __init__(self, device, rgb_backbone_name='vit_base_patch8_224_dino', out_indices=None,
                 checkpoint_path='/fuxi_team14_intern/m3dm/checkpoints/dino_vitbase8_pretrain.pth',
                 pool_last=False, xyz_backbone_name='Point_MAE', group_size=128, num_group=1024):
        super().__init__()
        # 'vit_base_patch8_224_dino'
        # Determine if to output features.
        self.device = device

        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})

        ## RGB backbone
        self.rgb_backbone = timm.create_model(model_name=rgb_backbone_name, pretrained=False,
                                              checkpoint_path=checkpoint_path,
                                              **kwargs)

        ## XYZ backbone

        if xyz_backbone_name == 'Point_MAE':
            self.xyz_backbone = PointTransformer(group_size=group_size, num_group=num_group)
            self.xyz_backbone.load_model_from_ckpt("/fuxi_team14_intern/m3dm/checkpoints/pointmae_pretrain.pth")
        elif xyz_backbone_name == 'Point-BERT':
            self.xyz_backbone=PointTransformer(group_size=group_size, num_group=num_group, encoder_dims=256)
            self.xyz_backbone.load_model_from_pb_ckpt("/fuxi_team14_intern/m3dm/checkpoints/Point-BERT.pth")
        elif xyz_backbone_name == 'FPFH':
            self.xyz_backbone=FPFH(group_size=group_size, num_group=num_group,voxel_size=0.05)
            #self.xyz_backbone.load_model_from_pb_ckpt("/workspace/data2/checkpoints/Point-BERT.pth")




    def forward_rgb_features(self, x):
        x = self.rgb_backbone.patch_embed(x)
        x = self.rgb_backbone._pos_embed(x)
        x = self.rgb_backbone.norm_pre(x)
        if self.rgb_backbone.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.rgb_backbone.blocks(x)
        x = self.rgb_backbone.norm(x)

        feat = x[:,1:].permute(0, 2, 1).view(1, -1, 28, 28)
        return feat


    def forward(self, rgb, xyz):
        
        rgb_features = self.forward_rgb_features(rgb)

        xyz_features, center, ori_idx, center_idx = self.xyz_backbone(xyz)

        xyz_features.permute(0, 2, 1)

        return rgb_features, xyz_features, center, ori_idx, center_idx



def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center, center_idx = fps(xyz.contiguous(), self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class PointTransformer(nn.Module):
    def __init__(self, group_size=128, num_group=1024, encoder_dims=384):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6

        self.group_size = group_size
        self.num_group = num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = encoder_dims
        if self.encoder_dims != self.trans_dim:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            #if incompatible.missing_keys:
            #    print('missing_keys')
            #    print(
            #            incompatible.missing_keys
            #        )
            #if incompatible.unexpected_keys:
            #    print('unexpected_keys')
            #    print(
            #            incompatible.unexpected_keys

            #        )

            # print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def load_model_from_pb_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys')
            print(
                    incompatible.missing_keys
                )
        if incompatible.unexpected_keys:
            print('unexpected_keys')
            print(
                    incompatible.unexpected_keys

                )
                
        print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')


    def forward(self, pts):
        if self.encoder_dims != self.trans_dim:
            B,C,N = pts.shape
            pts = pts.transpose(-1, -2) # B N 3
            # divide the point clo  ud in the same form. This is important
            neighborhood,  center, ori_idx, center_idx = self.group_divider(pts)
            # # generate mask
            # bool_masked_pos = self._mask_center(center, no_mask = False) # B G
            # encoder the input cloud blocks
            group_input_tokens = self.encoder(neighborhood)  #  B G N
            group_input_tokens = self.reduce_dim(group_input_tokens)
            # prepare cls
            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
            cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
            # add pos embedding
            pos = self.pos_embed(center)
            # final input
            x = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)
            # transformer
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x)[:,1:].transpose(-1, -2).contiguous() for x in feature_list]
            x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
            return x, center, ori_idx, center_idx 
        else:
            B, C, N = pts.shape
            pts = pts.transpose(-1, -2)  # B N 3
            # divide the point clo  ud in the same form. This is important
            neighborhood, center, ori_idx, center_idx = self.group_divider(pts)

            group_input_tokens = self.encoder(neighborhood)  # B G N

            pos = self.pos_embed(center)
            # final input
            x = group_input_tokens
            # transformer
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
            x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
            return x, center, ori_idx, center_idx
        
# class FPFH(nn.Module):
#     def __init__(self, group_size=32, num_group=512, voxel_size=0.05):
#         super(FPFH, self).__init__()
#         self.group_size = group_size
#         self.num_group = num_group
#         self.voxel_size = voxel_size
#         self.resize = nn.AdaptiveAvgPool2d((28, 28))
#         self.average = nn.AvgPool2d(2, 2)

#     def organized_pc_to_unorganized_pc(self, organized_pc):
#         return organized_pc.reshape(-1, 3)

#     def get_fpfh_features(self, organized_pc):
#         organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
#         unorganized_pc = self.organized_pc_to_unorganized_pc(organized_pc_np)
        
#         nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
#         unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
        
#         o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))

#         radius_normal = self.voxel_size * 2
#         o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

#         radius_feature = self.voxel_size * 5
#         pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#             o3d_pc, 
#             o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
#         )
#         fpfh = pcd_fpfh.data.T

#         full_fpfh = np.zeros((unorganized_pc.shape[0], fpfh.shape[1]), dtype=fpfh.dtype)
#         full_fpfh[nonzero_indices, :] = fpfh
#         full_fpfh_reshaped = full_fpfh.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1]))
#         full_fpfh_tensor = torch.tensor(full_fpfh_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
        
#         return full_fpfh_tensor

#     def forward(self, xyz):
#         batch_size, _, height, width = xyz.shape

#         # Compute FPFH features
#         xyz_features = self.get_fpfh_features(xyz)

#         # Resize and average
#         xyz_features_resized = self.resize(self.average(xyz_features))

#         # Randomly sample center points
#         center_idx = torch.randperm(height * width)[:self.num_group]
#         center = xyz.view(batch_size, 3, -1).permute(0, 2, 1)[:, center_idx, :]

#         # Create original indices
#         ori_idx = torch.arange(height * width).view(1, height, width).expand(batch_size, -1, -1)

#         return xyz_features_resized, center, ori_idx, center_idx

#     def add_sample_to_mem_bank(self, sample):
#         fpfh_feature_maps = self.get_fpfh_features(sample[1])
#         fpfh_feature_maps_resized = self.resize(self.average(fpfh_feature_maps))
#         fpfh_patch = fpfh_feature_maps_resized.reshape(fpfh_feature_maps_resized.shape[1], -1).T
#         return fpfh_patch

#     def predict(self, sample):
#         depth_feature_maps = self.get_fpfh_features(sample[1])
#         depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
#         patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T
#         return patch, depth_feature_maps_resized.shape[-2:]
import numpy as np
import open3d as o3d
class FPFH(nn.Module):
    def __init__(self, group_size=32, num_group=512, voxel_size=0.05):
        super(FPFH, self).__init__()
        self.group_size = group_size
        self.num_group = num_group
        self.voxel_size = voxel_size

    def get_fpfh_features(self, unorganized_pc):
        # 确保 unorganized_pc 是 CPU 上的 numpy 数组
        if isinstance(unorganized_pc, torch.Tensor):
            unorganized_pc = unorganized_pc.cpu().numpy()
        
        # 确保形状是 (N, 3)
        if unorganized_pc.shape[0] == 3:
            unorganized_pc = unorganized_pc.T
        
        # 移除零点
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
        
        # 确保数据类型是 float64
        unorganized_pc_no_zeros = unorganized_pc_no_zeros.astype(np.float64)
        
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(unorganized_pc_no_zeros)

        radius_normal = self.voxel_size * 2
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            o3d_pc, 
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        fpfh = pcd_fpfh.data  # 形状为 (33, M)，M 是非零点的数量

        # 将 FPFH 特征转换为 torch.Tensor
        fpfh_tensor = torch.tensor(fpfh, dtype=torch.float32)
        
        return fpfh_tensor

    def forward(self, xyz):
        # 假设 xyz 是形状为 (B, N, 3) 的 torch.Tensor，其中 B 是批量大小，N 是点的数量
        xyz = xyz.permute(0, 2, 1)
        batch_size,num_points, _ = xyz.shape

        # 计算 FPFH 特征
        fpfh_features = []
        for i in range(batch_size):
            fpfh = self.get_fpfh_features(xyz[i])
            fpfh_features.append(fpfh)
        
        fpfh_features = torch.stack(fpfh_features)

        # 随机采样中心点
        center_idx = torch.randperm(num_points)[:self.num_group]
        center = xyz[:, center_idx, :]

        ori_idx = torch.arange(num_points)

        return fpfh_features, center, ori_idx,center_idx

    def add_sample_to_mem_bank(self, sample):
        print(sample.shape)
        # 假设 sample 是形状为 (N, 3) 的 torch.Tensor
        fpfh_features = self.get_fpfh_features(sample)
        return fpfh_features

    def predict(self, sample):
        # 假设 sample 是形状为 (N, 3) 的 torch.Tensor
        fpfh_features = self.get_fpfh_features(sample)
        return fpfh_features, None  # 返回 None 作为形状，因为无序点云没有固定的空间维度
