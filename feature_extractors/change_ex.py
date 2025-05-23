import torch
import torch.nn as nn

class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p  # 1/p of the features will be exchanged.

    def forward(self, x0, x1):
        # x0, x1: the bi-temporal feature maps.
        N, C, H, W = x0.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).expand((N, -1))
        
        out_x0 = torch.zeros_like(x0)
        out_x1 = torch.zeros_like(x1)

        out_x0[~exchange_mask] = x0[~exchange_mask]
        out_x1[~exchange_mask] = x1[~exchange_mask]
        out_x0[exchange_mask] = x1[exchange_mask]
        out_x1[exchange_mask] = x0[exchange_mask]

        return out_x0, out_x1
    
class SpatialExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p  # 1/p of the features will be exchanged.

    def forward(self, x0, x1):
        # x0, x1: the bi-temporal feature maps.
        N, C, H, W = x0.shape
        # Create a mask based on width dimension
        exchange_mask = torch.arange(W, device=x0.device) % self.p == 0
        # Expand mask to match feature dimensions
        exchange_mask = exchange_mask.view(1, 1, 1, W).expand(N, C, H, -1)

        out_x0 = x0.clone()
        out_x1 = x1.clone()

        # Perform column-wise exchange
        out_x0[..., exchange_mask] = x1[..., exchange_mask]
        out_x1[..., exchange_mask] = x0[..., exchange_mask]

        return out_x0, out_x1

class CombinedExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.channel_exchange = ChannelExchange(p=p)
        self.spatial_exchange = SpatialExchange(p=p)

    def forward(self, x0, x1):
        # First perform channel exchange
        x0, x1 = self.channel_exchange(x0, x1)
        # Then perform spatial exchange
        x0, x1 = self.spatial_exchange(x0, x1)
        return x0, x1
