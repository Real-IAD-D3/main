import torch
import torch.nn as nn
import math

def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

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
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, 
#                  act_layer=nn.GELU, drop=0.2, use_ln=True):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
        
#         self.use_ln = use_ln
#         self.scale = min(1.0, (in_features ** -0.5))
        
#         # 第一个残差块
#         self.res_block1 = nn.Sequential(
#             nn.Linear(in_features, hidden_features),
#             nn.LayerNorm(hidden_features) if use_ln else nn.Identity(),
#             act_layer(),
#             nn.Dropout(drop),
#             nn.Linear(hidden_features, in_features),  # 注意输出维度要和输入相同
#             nn.LayerNorm(in_features) if use_ln else nn.Identity(),
#             nn.Dropout(drop)
#         )
        
#         # 第二个残差块
#         self.res_block2 = nn.Sequential(
#             nn.Linear(in_features, hidden_features),
#             nn.LayerNorm(hidden_features) if use_ln else nn.Identity(),
#             act_layer(),
#             nn.Dropout(drop),
#             nn.Linear(hidden_features, out_features),
#             nn.LayerNorm(out_features) if use_ln else nn.Identity(),
#             nn.Dropout(drop)
#         )
        
#         # 如果输入输出维度不同，需要一个投影层
#         self.proj = None
#         if in_features != out_features:
#             self.proj = nn.Linear(in_features, out_features)
        
#         self._init_weights()
        
#     def _init_weights(self):
#         gain = 0.001
#         # 初始化第一个残差块
#         for m in self.res_block1.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=gain)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#                 m.weight.data *= 0.1
        
#         # 初始化第二个残差块
#         for m in self.res_block2.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=gain)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#                 m.weight.data *= 0.1
        
#         # 初始化投影层
#         if self.proj is not None:
#             nn.init.xavier_uniform_(self.proj.weight, gain=gain)
#             nn.init.zeros_(self.proj.bias)
#             self.proj.weight.data *= 0.1

#     def forward(self, x):
#         # 第一个残差块
#         identity = x
#         x = self.res_block1(x * self.scale)
#         x = x * self.scale + identity
        
#         # 第二个残差块
#         identity = x if self.proj is None else self.proj(x)
#         x = self.res_block2(x * self.scale)
#         x = x * self.scale + identity
        
#         return x
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, 
#                  act_layer=nn.GELU, drop=0.1, use_ln=True):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
        
#         self.use_ln = use_ln
        
#         # 第一层
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         if use_ln:
#             self.ln1 = nn.LayerNorm(hidden_features)
#         self.act = act_layer()
#         self.drop1 = nn.Dropout(drop)
        
#         # 第二层
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         if use_ln:
#             self.ln2 = nn.LayerNorm(out_features)
#         self.drop2 = nn.Dropout(drop)
        
#         # 初始化参数
#         self._init_weights()
        
#     def _init_weights(self):
#         # 使用较小的初始值来防止梯度爆炸
#         nn.init.xavier_uniform_(self.fc1.weight, gain=0.01)
#         nn.init.zeros_(self.fc1.bias)
#         nn.init.xavier_uniform_(self.fc2.weight, gain=0.01)
#         nn.init.zeros_(self.fc2.bias)

#     def forward(self, x):
#         x = self.fc1(x)
#         if self.use_ln:
#             x = self.ln1(x)
#         x = self.act(x)
#         x = self.drop1(x)
        
#         x = self.fc2(x)
#         if self.use_ln:
#             x = self.ln2(x)
#         x = self.drop2(x)
        
#         return x
class FeatureFusionBlock(nn.Module):
    def __init__(self, xyz_dim, rgb_dim, mlp_ratio=4.):
        super().__init__()

        self.xyz_dim = xyz_dim
        self.rgb_dim = rgb_dim

        self.xyz_norm = nn.LayerNorm(xyz_dim)
        self.xyz_mlp = Mlp(in_features=xyz_dim, hidden_features=int(xyz_dim * mlp_ratio), act_layer=nn.GELU, drop=0.)
        initialize_weights(self.xyz_mlp)

        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.rgb_mlp = Mlp(in_features=rgb_dim, hidden_features=int(rgb_dim * mlp_ratio), act_layer=nn.GELU, drop=0.)
        initialize_weights(self.rgb_mlp)

        self.rgb_head = nn.Linear(rgb_dim, 256)
        self.xyz_head = nn.Linear(xyz_dim, 256)
        initialize_weights(self.rgb_head)
        initialize_weights(self.xyz_head)

        
        self.T = 1

    def feature_fusion(self, xyz_feature, rgb_feature):

        xyz_feature  = self.xyz_mlp(self.xyz_norm(xyz_feature))
        rgb_feature  = self.rgb_mlp(self.rgb_norm(rgb_feature))

        feature = torch.cat([xyz_feature, rgb_feature], dim=2)

        return feature

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, xyz_feature, rgb_feature):


        feature = self.feature_fusion(xyz_feature, rgb_feature)

        feature_xyz = feature[:,:, :self.xyz_dim]
        feature_rgb = feature[:,:, self.xyz_dim:]

        q = self.rgb_head(feature_rgb.view(-1, feature_rgb.shape[2]))
        k = self.xyz_head(feature_xyz.view(-1, feature_xyz.shape[2]))

        xyz_feature = xyz_feature.view(-1, xyz_feature.shape[2])
        rgb_feature = rgb_feature.view(-1, rgb_feature.shape[2])

        patch_no_zeros_indices = torch.nonzero(torch.all(xyz_feature != 0, dim=1))
        
        loss = self.contrastive_loss(q[patch_no_zeros_indices,:].squeeze(), k[patch_no_zeros_indices,:].squeeze())

        return loss

