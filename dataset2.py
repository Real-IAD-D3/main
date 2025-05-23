import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
import math
import cv2
import argparse
from matplotlib import pyplot as plt

def eyecandies_classes():
    return [
        'CandyCane',
        'ChocolateCookie',
        'ChocolatePraline',
        'Confetto',
        'GummyBear',
        'HazelnutTruffle',
        'LicoriceSandwich',
        'Lollipop',
        'Marshmallow',
        'PeppermintCandy',
    ]

def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]
def test_3d_classes():
    return [
             'audio_jack_socket',
            'common_mode_filter',
             'connector_housing-female',
             'crimp_st_cable_mount_box',
            'dc_power_connector',
             'fork_crimp_terminal',
              'headphone_jack_socket',
            'miniature_lifting_motor',
               'purple-clay-pot',
             'power_jack',
            
            'ethernet_connector',
               'ferrite_bead',
            'fuse_holder',
            'humidity_sensor',
            'knob-cap',
            'lattice_block_plug',
            'lego_pin_connector_plate',
            'lego_propeller',
               'limit-switch',
            'telephone_spring_switch',
        # "bagel",
        # "cable_gland",
        # "carrot",
        # "cookie",
        # "dowel",
        # "foam",
        # "peach",
        # "potato",
        # "rope",
        # "tire",
            ]

RGB_SIZE = 224

class BaseAnomalyDetectionDataset(Dataset):

    def __init__(self, split, class_name, img_size,downsampling, angle, small,dataset_path='datasets/eyecandies_preprocessed'):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split)
        self.downsampling = downsampling
        self.angle = angle
        self.small = small
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])
    def analyze_depth_importance(self,organized_pc):
        """分析深度图的重要性，返回行列的重要性标记"""
        depth = organized_pc[:,:,2]  # 获取深度通道

        # 计算每行的非零点占比
        row_importance = np.mean(depth != 0, axis=1)
        col_importance = np.mean(depth != 0, axis=0)

        # 使用中位数作为分界线
        row_median = np.median(row_importance[row_importance > 0])
        col_median = np.median(col_importance[col_importance > 0])

        # 标记重要性（True表示重要行/列）
        important_rows = row_importance >= row_median
        important_cols = col_importance >= col_median

        return important_rows, important_cols

    def smart_downsample(self, organized_pc, target_factor):
        """智能降采样，重要区域降采样更多，保持总体降采样因子不变"""
        important_rows, important_cols = self.analyze_depth_importance(organized_pc)

        # 获取重要和不重要的行列索引
        important_row_indices = np.where(important_rows)[0]
        unimportant_row_indices = np.where(~important_rows)[0]
        important_col_indices = np.where(important_cols)[0]
        unimportant_col_indices = np.where(~important_cols)[0]

        # 计算原始的重要和不重要区域的比例
        total_rows = len(important_row_indices) + len(unimportant_row_indices)
        total_cols = len(important_col_indices) + len(unimportant_col_indices)
        factor = int(math.sqrt(target_factor))
        # 目标总行列数
        target_total_rows = total_rows // factor
        target_total_cols = total_cols // factor

        # 设置重要区域的更高降采样率（比如降采样率为1/4）
        important_factor = factor*2

        # 计算重要区域的目标数量
        n_important_rows = len(important_row_indices) // important_factor
        n_important_cols = len(important_col_indices) // important_factor

        # 计算不重要区域需要保留的数量（确保总数符合目标）
        n_unimportant_rows = target_total_rows - n_important_rows
        n_unimportant_cols = target_total_cols - n_important_cols

        # 确保不重要区域的数量不会超过原始数量
        n_unimportant_rows = min(n_unimportant_rows, len(unimportant_row_indices))
        n_unimportant_cols = min(n_unimportant_cols, len(unimportant_col_indices))

        # 选择行
        selected_important_rows = np.linspace(0, len(important_row_indices)-1, n_important_rows, dtype=int)
        selected_important_rows = important_row_indices[selected_important_rows]

        selected_unimportant_rows = np.linspace(0, len(unimportant_row_indices)-1, n_unimportant_rows, dtype=int)
        selected_unimportant_rows = unimportant_row_indices[selected_unimportant_rows]

        # 选择列
        selected_important_cols = np.linspace(0, len(important_col_indices)-1, n_important_cols, dtype=int)
        selected_important_cols = important_col_indices[selected_important_cols]

        selected_unimportant_cols = np.linspace(0, len(unimportant_col_indices)-1, n_unimportant_cols, dtype=int)
        selected_unimportant_cols = unimportant_col_indices[selected_unimportant_cols]

        # 合并选择的行和列
        selected_rows = np.sort(np.concatenate([selected_important_rows, selected_unimportant_rows]))
        selected_cols = np.sort(np.concatenate([selected_important_cols, selected_unimportant_cols]))

        # 打印降采样信息
        print(f"原始大小: {organized_pc.shape[:2]}")
        print(f"重要行: {len(important_row_indices)} -> {len(selected_important_rows)} (1/{important_factor})")
        print(f"不重要行: {len(unimportant_row_indices)} -> {len(selected_unimportant_rows)}")
        print(f"重要列: {len(important_col_indices)} -> {len(selected_important_cols)} (1/{important_factor})")
        print(f"不重要列: {len(unimportant_col_indices)} -> {len(selected_unimportant_cols)}")
        print(f"最终大小: {len(selected_rows)}x{len(selected_cols)}")
        print(f"实际降采样因子: {(total_rows*total_cols)/(len(selected_rows)*len(selected_cols)):.2f}")

        return organized_pc[selected_rows][:, selected_cols]

    def get_matrix(self, image, angle):
        image = self.pillow_to_opencv(image)
        (h, w) = image.shape[:2]
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = self.calculate_destination_points(0, w, 0, h, angle)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        return M

    def calculate_destination_points(self, left, right, top, bottom, angle):
        # 计算中心点
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2

        # 计算角度的弧度值
        angle_rad = math.radians(angle)

        # 计算目标点
        dst_points = []
        for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]:
            new_x = center_x + (x - center_x) * math.cos(angle_rad) - (y - center_y) * math.sin(angle_rad)
            new_y = center_y + (x - center_x) * math.sin(angle_rad) + (y - center_y) * math.cos(angle_rad)
            dst_points.append([new_x, new_y])

        return np.float32(dst_points)

    def perspective_transform(self, image, matrix):
        """
        :param image_path: 输入图像路径
        :param angle: 旋转角度
        :param save_type: 保存的图片类型
        :return: 输出图像
        """
        # 读取图像
        image = self.pillow_to_opencv(image)
        (h, w) = image.shape[:2]
        # 执行透视变换
        transformed = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=image[0, 0].tolist())
        transformed = self.opencv_to_pillow(transformed)
        return transformed

    def opencv_to_pillow(self, opencv_image):
        """
        将 OpenCV 图像转换为 Pillow 图像。

        :param opencv_image: OpenCV 图像对象（BGR 格式）
        :return: Pillow 图像对象
        """
        # 检查是否为彩色图像并转换通道顺序
        if len(opencv_image.shape) == 3:  # 彩色图像
            opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(opencv_image_rgb)
        else:  # 灰度图像
            return Image.fromarray(opencv_image)

    def pillow_to_opencv(self, pil_image):
        """
        将 Pillow 图像转换为 OpenCV 图像。

        :param pil_image: Pillow 图像对象
        :return: OpenCV 图像对象（BGR 格式）
        """
        # 将 Pillow 图像转换为 numpy 数组
        opencv_image = np.array(pil_image)

        # 如果是 RGB 图像，转换为 BGR 格式
        if opencv_image.ndim == 3 and opencv_image.shape[2] == 3:  # 检查是否为 RGB 图像
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        return opencv_image

class PreTrainTensorDataset(Dataset):
    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.tensor_paths = os.listdir(self.root_path)


    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]

        tensor = torch.load(os.path.join(self.root_path, tensor_path))

        label = 0

        return tensor, label

class TrainDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size,downsampling,angle, small,   dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="train", class_name=class_name, img_size=img_size,downsampling=downsampling, angle=angle, small=small, dataset_path=dataset_path)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        # rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        rgb_paths = glob.glob(os.path.join(self.img_path, 'GOOD', 'rgb', '*', '*L05*RGB*.jpg'))
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        ps_paths = glob.glob(os.path.join(self.img_path, 'good', 'ps') + "/*.jpg")
        rgb_paths.sort()
        tiff_paths.sort()
        ps_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths, ps_paths))
        # sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        # img_tot_paths = img_tot_paths[0:10]
        # tot_labels = tot_labels[0:10]
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        ps_path = img_path[2]
        img = Image.open(rgb_path).convert('RGB')
        # add rotation
        # matrix = self.get_matrix(img, self.angle)
        # img = self.perspective_transform(img, matrix)

        img = self.rgb_transform(img)
        ps= Image.open(ps_path).convert('RGB')
        # # add rotation
        # ps = self.perspective_transform(ps, matrix)

        ps = self.rgb_transform(ps)
        if self.downsampling > 1:
            organized_pc = read_tiff_organized_pc(tiff_path)
            factor1 = int(math.floor(math.sqrt(self.downsampling)))
            factor2 = int(math.ceil(self.downsampling / factor1))
            organized_pc = organized_pc[::factor1, ::factor2]
            #organized_pc = self.smart_downsample(organized_pc, self.downsampling)
        else:
            organized_pc = read_tiff_organized_pc(tiff_path)

        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
        resized_organized_pc = resized_organized_pc.clone().detach().float()


        return (img, resized_organized_pc, resized_depth_map_3channel,ps), label


class TestDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size,downsampling,angle,small,dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="test", class_name=class_name, img_size=img_size,downsampling=downsampling,angle=angle, small=small,  dataset_path=dataset_path)
        self.gt_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        ps_tot_paths = []
        defect_types = os.listdir(self.img_path)
        print(defect_types)
        # 如果types不为NONE，只保留GOOD和指定的types
        for defect_type in defect_types:
            if defect_type == 'good':
                # rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb', '*', '*L05*RGB*.jpg'))
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                #gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                ps_paths = glob.glob(os.path.join(self.img_path, 'good', 'ps') + "/*.jpg")
                rgb_paths.sort()
                tiff_paths.sort()
                ps_paths.sort()
                # 只保留前5个样本
                if self.small:
                    rgb_paths = rgb_paths[:5]
                    tiff_paths = tiff_paths[:5]
                    ps_paths = ps_paths[:5]
                sample_paths = list(zip(rgb_paths, tiff_paths,ps_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                # rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb', '*', '*L05*RGB*.jpg'))
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                ps_paths = glob.glob(os.path.join(self.img_path, defect_type, 'ps') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                ps_paths.sort()
                # if self.small:
                #     # 检查每个gt mask中缺陷占比
                #     valid_indices = []
                #     for i, gt_path in enumerate(gt_paths):
                #         gt_mask = np.array(Image.open(gt_path))
                #         total_pixels = gt_mask.shape[0] * gt_mask.shape[1]
                #         threshold = int(total_pixels * 0.005)  # 计算1%像素数量
                #         defect_pixels = np.sum(gt_mask > 0)
                #         if defect_pixels <= threshold:  # 直接比较像素数量
                #             valid_indices.append(i)

                #     # 只保留缺陷占比<=1%的样本
                #     rgb_paths = [rgb_paths[i] for i in valid_indices]
                #     tiff_paths = [tiff_paths[i] for i in valid_indices]
                #     gt_paths = [gt_paths[i] for i in valid_indices]
                #     # ps_paths = [ps_paths[i] for i in valid_indices]
                sample_paths = list(zip(rgb_paths, tiff_paths,ps_paths))
                print(f"rgb_paths: {len(rgb_paths)}, tiff_paths: {len(tiff_paths)}, ps_paths: {len(ps_paths)}")
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))
        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        ps_path = img_path[2]
        img_original = Image.open(rgb_path).convert('RGB')
        # matrix = self.get_matrix(img_original, self.angle)
        # img_original = self.perspective_transform(img_original, matrix)

        # axes[0].imshow(cv2.cvtColor(self.pillow_to_opencv(img_original), cv2.COLOR_BGR2RGB))

        img = self.rgb_transform(img_original)
        ps_original = Image.open(ps_path).convert('RGB')
        # ps_original = self.perspective_transform(ps_original, matrix)
        ps = self.rgb_transform(ps_original)
        # organized_pc = read_tiff_organized_pc(tiff_path)
        if self.downsampling > 1:
            organized_pc = read_tiff_organized_pc(tiff_path)
            factor1 = int(math.floor(math.sqrt(self.downsampling)))
            factor2 = int(math.ceil(self.downsampling / factor1))
            organized_pc = organized_pc[::factor1, ::factor2]
            #organized_pc = self.smart_downsample(organized_pc, self.downsampling)
        else:
            organized_pc = read_tiff_organized_pc(tiff_path)

        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
        resized_organized_pc = resized_organized_pc.clone().detach().float()




        if gt == 0:
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
            # axes[1].imshow(self.pillow_to_opencv(gt), cmap='gray')
        else:
            gt = Image.open(gt).convert('L')
            # gt = self.perspective_transform(gt, matrix)
            # axes[1].imshow(self.pillow_to_opencv(gt), cmap='gray')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        # fig.show()
        # fig.savefig("test_image_3.png", dpi=300, bbox_inches='tight')

        return (img, resized_depth_map_3channel,ps), gt[:1], label, rgb_path

class ValidDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size,downsampling,angle,small,defect_name,dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="test", class_name=class_name, img_size=img_size,downsampling=downsampling,angle=angle, small=small,  dataset_path=dataset_path)
        self.gt_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.defect_name = defect_name
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        #ps_tot_paths = []
        defect_types = os.listdir(self.img_path)
        # print(defect_types)
        # 如果types不为NONE，只保留GOOD和指定的types
        for defect_type in defect_types:
            # print(defect_type)
            if defect_type == 'GOOD':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb', '*', '*L05*RGB*.jpg'))
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                ps_paths = glob.glob(os.path.join(self.img_path, 'GOOD', 'ps') + "/*.jpg")
                # print(os.path.join(self.img_path, defect_type, 'rgb', '*', '*L05*RGB*.jpg'))
                # print(rgb_paths)
                rgb_paths.sort()
                tiff_paths.sort()
                ps_paths.sort()
                # 只保留前5个样本
                if self.small:
                    rgb_paths = rgb_paths[:5]
                    tiff_paths = tiff_paths[:5]
                    ps_paths = ps_paths[:5]
                sample_paths = list(zip(rgb_paths, tiff_paths, ps_paths))
                sample_paths = sample_paths[:5]
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            elif defect_type in self.defect_name:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb', '*', '*L05*RGB*.jpg'))
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt','rgb') + "/*.png")
                ps_paths = glob.glob(os.path.join(self.img_path, defect_type, 'ps') + "/*.jpg")
                # print(os.path.join(self.img_path, defect_type, 'rgb', '*', '*L05*RGB*.jpg'))
                # print(rgb_paths)
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                ps_paths.sort()
                if self.small:
                    # 检查每个gt mask中缺陷占比
                    valid_indices = []
                    for i, gt_path in enumerate(gt_paths):
                        gt_mask = np.array(Image.open(gt_path))
                        total_pixels = gt_mask.shape[0] * gt_mask.shape[1]
                        threshold = int(total_pixels * 0.005)  # 计算1%像素数量
                        defect_pixels = np.sum(gt_mask > 0)
                        if defect_pixels <= threshold:  # 直接比较像素数量
                            valid_indices.append(i)

                    # 只保留缺陷占比<=1%的样本
                    rgb_paths = [rgb_paths[i] for i in valid_indices]
                    tiff_paths = [tiff_paths[i] for i in valid_indices]
                    gt_paths = [gt_paths[i] for i in valid_indices]
                    ps_paths = [ps_paths[i] for i in valid_indices]
                sample_paths = list(zip(rgb_paths, tiff_paths, ps_paths))
                print(f"rgb_paths: {len(rgb_paths)}, tiff_paths: {len(tiff_paths)}, ps_paths: {len(ps_paths)}")
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        ps_path = img_path[2]
        img_original = Image.open(rgb_path).convert('RGB')
        matrix = self.get_matrix(img_original, self.angle)
        img_original = self.perspective_transform(img_original, matrix)

        # axes[0].imshow(cv2.cvtColor(self.pillow_to_opencv(img_original), cv2.COLOR_BGR2RGB))

        img = self.rgb_transform(img_original)
        ps_original = Image.open(ps_path).convert('RGB')
        ps_original = self.perspective_transform(ps_original, matrix)
        ps = self.rgb_transform(ps_original)
        # organized_pc = read_tiff_organized_pc(tiff_path)
        if self.downsampling > 1:
            organized_pc = read_tiff_organized_pc(tiff_path)
            factor1 = int(math.floor(math.sqrt(self.downsampling)))
            factor2 = int(math.ceil(self.downsampling / factor1))
            organized_pc = organized_pc[::factor1, ::factor2]
            #organized_pc = self.smart_downsample(organized_pc, self.downsampling)
        else:
            organized_pc = read_tiff_organized_pc(tiff_path)

        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
        resized_organized_pc = resized_organized_pc.clone().detach().float()




        if gt == 0:
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
            # axes[1].imshow(self.pillow_to_opencv(gt), cmap='gray')
        else:
            gt = Image.open(gt).convert('L')
            gt = self.perspective_transform(gt, matrix)
            # axes[1].imshow(self.pillow_to_opencv(gt), cmap='gray')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        # fig.show()
        # fig.savefig("test_image_3.png", dpi=300, bbox_inches='tight')

        return (img, resized_depth_map_3channel, ps), gt[:1], label, rgb_path

from torch.utils.data import Subset
def redistribute_dataset(dataset, chunk_size=50):
    """
    将数据集每50个分成一组，并重新分配包含GOOD的组
    :param dataset: 原始数据集
    :param chunk_size: 每组的大小
    :return: 重新分配后的数据集列表
    """
    total_size = len(dataset)
    num_chunks = (total_size + chunk_size - 1) // chunk_size  # 向上取整

    # 初始分组
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_size)
        chunk_indices = list(range(start_idx, end_idx))
        chunks.append(Subset(dataset, chunk_indices))

    # 找出包含GOOD的组
    good_chunks = []
    normal_chunks = []

    for i, chunk in enumerate(chunks):
        has_good = False
        # 检查这个chunk中是否包含GOOD
        for idx in chunk.indices:
            if 'GOOD' in dataset.img_paths[idx][0]:
                has_good = True
                break

        if has_good:
            good_chunks.append(i)
        else:
            normal_chunks.append(i)

    # 如果没有GOOD组或没有正常组，直接返回原始分组
    if not good_chunks or not normal_chunks:
        return chunks

    # 重新分配GOOD组的数据
    for good_chunk_idx in good_chunks:
        chunk = chunks[good_chunk_idx]
        good_indices = []
        normal_indices = []

        # 分离GOOD和非GOOD数据
        for idx in chunk.indices:
            if 'GOOD' in dataset.img_paths[idx][0]:
                good_indices.append(idx)
            else:
                normal_indices.append(idx)

        # 将GOOD数据平均分配给其他组
        num_good = len(good_indices)
        num_normal_chunks = len(normal_chunks)

        if num_normal_chunks > 0:
            # 计算每个正常组应该获得多少GOOD数据
            indices_per_chunk = num_good // num_normal_chunks
            remainder = num_good % num_normal_chunks

            # 分配GOOD数据
            current_good_idx = 0
            for i, normal_chunk_idx in enumerate(normal_chunks):
                extra = 1 if i < remainder else 0
                num_to_add = indices_per_chunk + extra

                # 添加GOOD数据到正常组
                chunks[normal_chunk_idx] = Subset(dataset,
                    list(chunks[normal_chunk_idx].indices) +
                    good_indices[current_good_idx:current_good_idx + num_to_add]
                )
                current_good_idx += num_to_add

        # 更新原始GOOD组，只保留非GOOD数据
        chunks[good_chunk_idx] = Subset(dataset, normal_indices)

    return chunks
def get_data_loader(split, class_name, img_size, args, defect_name = None):
    if split in ['train']:
        dataset = TrainDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path, downsampling=args.downsampling, angle=args.rotate_angle,small=args.small)
    elif split in ['test']:
        dataset = TestDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path, downsampling=args.downsampling, angle=args.rotate_angle,small=args.small)
    elif split in ['validation']:
        dataset = ValidDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path, downsampling=args.downsampling, angle=args.rotate_angle,small=args.small, defect_name = defect_name)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader

def get_data_set(split, class_name, img_size, args, defect_name = None):
    if split in ['train']:
        dataset = TrainDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path, downsampling=args.downsampling, angle=args.rotate_angle)
    elif split in ['test']:
        print('test')
        dataset = TestDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path, downsampling=args.downsampling, angle=args.rotate_angle)
    elif split in ['validation']:
        print('validation')
        dataset = ValidDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path, downsampling=args.downsampling, angle=args.rotate_angle,small=args.small, defect_name = defect_name)

    return dataset

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Process some integers.')
#
#     parser.add_argument('--method_name', default='DINO+Point_MAE+Fusion', type=str,
#                         choices=['DINO','Point_MAE','Fusion','DINO+Point_MAE','DINO+Point_MAE+Fusion','DINO+Point_MAE+add','DINO+FPFH','DINO+FPFH+Fusion',
#                                  'DINO+FPFH+Fusion+ps','DINO+Point_MAE+Fusion+ps','DINO+Point_MAE+ps','DINO+FPFH+ps','ours','ours2','ours3','ours_final','ours_final1'
#                                  ,'ours_final1_VS'],
#                         help='Anomaly detection modal name.')
#     parser.add_argument('--max_sample', default=400, type=int,
#                         help='Max sample number.')
#     parser.add_argument('--memory_bank', default='multiple', type=str,
#                         choices=["multiple", "single"],
#                         help='memory bank mode: "multiple", "single".')
#     parser.add_argument('--rgb_backbone_name', default='vit_base_patch8_224_dino', type=str,
#                         choices=['vit_base_patch8_224_dino', 'vit_base_patch8_224', 'vit_base_patch8_224_in21k', 'vit_small_patch8_224_dino'],
#                         help='Timm checkpoints name of RGB backbone.')
#     parser.add_argument('--xyz_backbone_name', default='Point_MAE', type=str, choices=['Point_MAE', 'Point_Bert','FPFH'],
#                         help='Checkpoints name of RGB backbone[Point_MAE, Point_Bert, FPFH].')
#     parser.add_argument('--fusion_module_path', default='checkpoints/checkpoint-0.pth', type=str,
#                         help='Checkpoints for fusion module.')
#     parser.add_argument('--save_feature', default=False, action='store_true',
#                         help='Save feature for training fusion block.')
#     parser.add_argument('--use_uff', default=False, action='store_true',
#                         help='Use UFF module.')
#     parser.add_argument('--save_feature_path', default='datasets/patch_lib', type=str,
#                         help='Save feature for training fusion block.')
#     parser.add_argument('--save_preds', default=False, action='store_true',
#                         help='Save predicts results.')
#     parser.add_argument('--group_size', default=128, type=int,