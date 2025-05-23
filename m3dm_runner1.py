import torch
from tqdm import tqdm
import os
import numpy as np

from feature_extractors import multiple_features_clean as multiple_features
        
from dataset import get_data_loader

class M3DM():
    def __init__(self, args):
        self.args = args
        self.image_size = args.img_size
        self.count = args.max_sample
        if args.method_name == 'DINO':
            self.methods = {
                "DINO": multiple_features.RGBFeatures(args),
            }
        elif args.method_name == 'Point_MAE':
            self.methods = {
                "Point_MAE": multiple_features.PointFeatures(args),
            }
        elif args.method_name == 'Fusion':
            self.methods = {
                "Fusion": multiple_features.FusionFeatures(args),
            }
        elif args.method_name == 'DINO+Point_MAE':
            self.methods = {
                "DINO+Point_MAE": multiple_features.DoubleRGBPointFeatures(args),
            }
        elif args.method_name == 'DINO+Point_MAE+add':
            self.methods = {
                "DINO+Point_MAE": multiple_features.DoubleRGBPointFeatures_add(args),
            }
        elif args.method_name == 'DINO+Point_MAE+Fusion':
            self.methods = {
                "DINO+Point_MAE+Fusion": multiple_features.TripleFeatures(args),
            }
        elif args.method_name == 'DINO+FPFH':
            self.methods = {
                "DINO+FPFH": multiple_features.DoubleRGBPointFeatures(args),
            }
        elif args.method_name == 'DINO+FPFH+Fusion':
            self.methods = {
                "DINO+FPFH+Fusion": multiple_features.TripleFeatures(args),
            }
        elif args.method_name == 'DINO+FPFH+Fusion+ps':
            self.methods = {
                "DINO+FPFH+Fusion+ps": multiple_features.TripleFeatures_PS(args),
            }
        elif args.method_name == 'DINO+Point_MAE+Fusion+ps':
            self.methods = {
                "DINO+Point_MAE+Fusion+ps": multiple_features.TripleFeatures_PS(args),
            }
        elif args.method_name == 'DINO+Point_MAE+ps':
            self.methods = {
                "DINO+Point_MAE+ps": multiple_features.DoubleRGB_PS_Features(args),
            }
        elif args.method_name == 'DINO+FPFH+ps':
            self.methods = {
                "DINO+FPFH+ps": multiple_features.DoubleRGB_PS_Features(args),
            }
        elif args.method_name == 'ours':
            self.methods = {
                "ours": multiple_features.PSRGBPointFeatures_add(args),
            }
        elif args.method_name == 'ours2':
            self.methods = {
                "ours2": multiple_features.TripleFeatures_PS2(args),
            }
        elif args.method_name == 'ours3':
            self.methods = {
                "ours3": multiple_features.FourFeatures(args),
            }
        elif args.method_name == 'ours_final':
            self.methods = {
                "ours_final": multiple_features.TripleFeatures_PS_EX(args),
            }
        elif args.method_name == 'ours_final1':
            self.methods = {
                "ours_final1": multiple_features.PSRGBPointFeatures_add_EX(args),
            }
        elif args.method_name == 'm3dm_uninterpolate':
            self.methods = {
                "m3dm_uninterpolate": multiple_features.DoubleRGBPointFeatures_uninter_full(args),
            }

    def fit(self, class_name):
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size, args=self.args)

        flag = 0
        for sample, _ in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            for method in self.methods.values():
                if self.args.save_feature:
                    method.add_sample_to_mem_bank(sample, class_name=class_name)
                else:
                    method.add_sample_to_mem_bank(sample)
                flag += 1
            if flag > self.count:
                flag = 0
                break
                
        for method_name, method in self.methods.items():
            print(f'\n\nRunning coreset for {method_name} on class {class_name}...')
            method.run_coreset()
            

        if self.args.memory_bank == 'multiple':    
            flag = 0
            for sample, _ in tqdm(train_loader, desc=f'Running late fusion for {method_name} on class {class_name}..'):
                for method_name, method in self.methods.items():
                    method.add_sample_to_late_fusion_mem_bank(sample)
                    flag += 1
                if flag > self.count:
                    flag = 0
                    break
        
            for method_name, method in self.methods.items():
                print(f'\n\nTraining Dicision Layer Fusion for {method_name} on class {class_name}...')
                method.run_late_fusion()

    def evaluate(self, class_name):
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        valid_dir = os.path.join(self.args.dataset_path, class_name, 'validation')
        defect_names = os.listdir(valid_dir)
        # print(defect_names)
        path_list = []
        for defect_name in defect_names:
            if defect_name == 'GOOD':
                continue
            test_loader = get_data_loader("validation", class_name=class_name, img_size=self.image_size, args=self.args, defect_name=defect_name)
            with torch.no_grad():
                print(class_name, defect_name)
                print(f'len of loader:{len(test_loader)}')
                for sample, mask, label, rgb_path in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                    for method in self.methods.values():
                        method.predict(sample, mask, label)
                        path_list.append(rgb_path)
                            

            for method_name, method in self.methods.items():
                method.calculate_metrics()
                image_rocauc = method.image_rocauc
                pixel_rocauc = method.pixel_rocauc
                au_pro = method.au_pro
            
                print(f"Debug - Raw values: Image ROCAUC: {image_rocauc}, Pixel ROCAUC: {pixel_rocauc}, AU-PRO: {au_pro}")
            
                if np.isnan(image_rocauc) or np.isnan(pixel_rocauc) or np.isnan(au_pro):
                    print(f"Warning: NaN detected for {method_name}")
                # 可以在这里添加更多的调试信息
                    # 可以在这里添加更多的调试信息
                image_rocaucs[f'{method_name}_{defect_name}'] = round(method.image_rocauc, 3)
                pixel_rocaucs[f'{method_name}_{defect_name}'] = round(method.pixel_rocauc, 3)
                au_pros[f'{method_name}_{defect_name}'] = round(method.au_pro, 3)
                print(
                   f'Debug - Class: {class_name}, {method_name}, defect_name:{defect_name} Image ROCAUC: {method.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {method.pixel_rocauc:.3f}, {method_name} AU-PRO: {method.au_pro:.3f}')
                if self.args.save_preds:
                    method.save_prediction_maps('./pred_maps', path_list)
        return image_rocaucs, pixel_rocaucs, au_pros
