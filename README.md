D³M is a comprehensive framework for industrial anomaly detection, localization, and classification, leveraging a variety of deep learning-based feature extractors for 2D (RGB) and 3D (Point Cloud) data, along with flexible fusion strategies. This repository provides an implementation of the D3M framework and tools for evaluating its performance on standard and custom datasets.

The project is designed to explore and benchmark different uni-modal and multi-modal approaches for anomaly detection in industrial settings, inspired by the challenges of real-world defect identification.

## 📄 Citation

If you find this work useful in your research, please consider citing our associated paper:

```bibtex
@article{zhu2025real,
  title={Real-IAD D3: A Real-World 2D/Pseudo-3D/3D Dataset for Industrial Anomaly Detection},
  author={Zhu, Wenbing and Wang, Lidong and Zhou, Ziqing and Wang, Chengjie and Pan, Yurui and Zhang, Ruoyi and Chen, Zhuhao and Cheng, Linjie and Gao, Bin-Bin and Zhang, Jiangning and others},
  journal={arXiv preprint arXiv:2504.14221},
  year={2025}
}
```

---

## ✨ Features

* **Versatile Anomaly Detection Framework:** Supports a wide range of feature extraction and fusion methodologies.
* **Multi-Modal Support:** Handles both 2D RGB images and 3D point cloud data.
    * **RGB Feature Extractors:** Utilizes pre-trained models like DINO (e.g., `vit_base_patch8_224_dino`).
    * **Point Cloud Feature Extractors:** Incorporates models like PointMAE, PointBert, and traditional FPFH.
* **Flexible Fusion Strategies:** Implements various early and late fusion techniques for combining multi-modal features (e.g., simple addition, dedicated fusion modules).
* **Multiple Method Configurations:** Easily experiment with different combinations of backbones and fusion approaches via command-line arguments (e.g., `DINO`, `Point_MAE`, `DINO+Point_MAE`, `DINO+Point_MAE+Fusion`, and custom "ours" variants).
* **Dataset Compatibility:**
    * Works with standard 3D anomaly detection datasets like MVTec 3D-AD and Eyecandies.
    * Easily adaptable to custom datasets (`test_3d` option).
* **Comprehensive Evaluation:**
    * Calculates image-level ROCAUC.
    * Calculates pixel/point-level ROCAUC for localization.
    * Calculates AU-PRO scores.
* **Memory Bank & Coreset Subsampling:** Implements memory bank concepts and coreset subsampling for efficient training and inference, particularly with large feature sets.
* **Extensible:** Designed to be easily extended with new feature extractors, fusion modules, or datasets.
* **Result Visualization:** Option to save prediction maps for qualitative analysis.

---

## 🚀 Getting Started

### Prerequisites

* Python >= 3.9
* PyTorch >= 2.5.1 (preferably with CUDA support for GPU acceleration)
* Other dependencies as listed in `requirements.txt`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/d3m.git](https://github.com/your-username/d3m.git)
    cd d3m
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Or using Conda:
    ```bash
    conda create -n d3m python=3.9
    conda activate d3m
    ```

3.  **Install dependencies:**
    A `requirements.txt` file is recommended. Create one with the following content or adapt it to your precise needs:
    ```text
    # requirements.txt

    # Core Deep Learning & Computer Vision
    torch>=2.5.1 # Install with specific CUDA version if needed, e.g., torch>=2.5.1+cu121
    torchvision>=0.20.1
    torchaudio>=2.5.1
    numpy>=1.26.3
    opencv-python>=4.11.0
    Pillow>=11.0.0
    matplotlib>=3.4.2
    scikit-learn>=1.6.1
    scikit-image>=0.24.0
    einops>=0.8.1
    timm>=1.0.15

    # Utilities & Configuration
    pyyaml>=5.4.1
    tqdm>=4.67.1
    easydict>=1.13 # or addict>=2.4.0
    pandas>=2.2.3
    h5py>=3.13.0

    # Experiment Tracking & Model Hubs (Optional, uncomment if used)
    # wandb>=0.19.10
    # huggingface-hub>=0.29.3

    # Potentially 3D-Specific (Uncomment if your project uses these)
    # open3d>=0.19.0
    # pointnet2-ops>=3.0.0 # May require custom compilation
    # chamferdist>=1.0.3
    ```
    Then install using:
    ```bash
    pip install -r requirements.txt
    ```
    **Note on PyTorch:** Ensure you install the PyTorch version compatible with your CUDA toolkit. Refer to the [official PyTorch website](https://pytorch.org/) for installation instructions.

---

## 💻 Usage

The main script for running experiments is likely named `your_main_script_name.py` (the one containing the `run_3d_ads` function and `argparse` definitions). You can configure experiments using command-line arguments.

### Example Command

```bash
python your_main_script_name.py \
    --dataset_type mvtec3d \
    --dataset_path /path/to/your/mvtec3d_anomaly_detection \
    --method_name DINO+Point_MAE+Fusion \
    --rgb_backbone_name vit_base_patch8_224_dino \
    --xyz_backbone_name Point_MAE \
    --fusion_module_path /path/to/your/fusion_checkpoint.pth \
    --img_size 224 \
    --max_sample 400 \
    --coreset_eps 0.9 \
    --save_preds
    # Add other arguments as needed
```
Refer to the `argparse` section in the main script for a full list of available arguments and their descriptions.

---

## 📊 Output

The script will print tables summarizing the performance metrics for each class and method:

* **Image ROCAUC Results**
* **Pixel ROCAUC Results**
* **AU-PRO Results**

If `--save_preds` is enabled, anomaly maps will be saved to the specified directory (default or `./pred_maps`).

---

## 🙏 Acknowledgements

* This framework builds upon concepts from various SOTA anomaly detection and feature extraction literature.
* Utilizes awesome libraries like PyTorch, TIMM, Open3D, etc.

---

## 📞 Contact

For questions, issues, or suggestions, please open an issue in this repository.

```
