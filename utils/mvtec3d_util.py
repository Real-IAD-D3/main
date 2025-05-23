import tifffile as tiff
import torch
import numpy as np


def organized_pc_to_unorganized_pc(organized_pc):
    print(f"organized_pc shape: {organized_pc.shape}")
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])


def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    # nan_count = np.isnan(tiff_img).sum()
    # nan_percentage = nan_count / tiff_img.size * 100
    # print(f"NaN percentage: {nan_percentage:.2f}%")
    # 计算均值，忽略 NaN 值
    # mean_min = np.nanmin(tiff_img)

    # # 将 NaN 值替换为均值
    # tiff_img[np.isnan(tiff_img)] = mean_min
    # 将NaN值替换为0
    # tiff_img[np.isnan(tiff_img)] = 0
    #tiff_img = np.resize(tiff_img, (800, 800))
    # 获取第三通道非nan的最小值
    # min_z = np.nanmin(tiff_img[:, :, 2])
    
    # # 如果第三通道最小值小于等于0
    # if min_z <= 0:
    #     # 第三通道非nan的值减去最小值再加1
    #     mask = ~np.isnan(tiff_img[:, :, 2])
    #     tiff_img[mask, 2] = tiff_img[mask, 2] - min_z + 1
    # # 将NaN值置0
    #tiff_img[np.isnan(tiff_img)] = 0
    return tiff_img


def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=True):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).contiguous()
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode='nearest')
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0).contiguous()
    else:
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).contiguous().numpy()


def organized_pc_to_depth_map(organized_pc):
    return organized_pc[:, :, 2]
