python3 main.py \
--method_name DINO+Point_MAE+Fusion \
--use_uff \
--memory_bank multiple \
--rgb_backbone_name vit_base_patch8_224_dino \
--xyz_backbone_name Point_MAE \
--fusion_module_path checkpoints/checkpoint-0.pth
