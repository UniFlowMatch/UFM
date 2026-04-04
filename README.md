
<div align="center">
<h1>UFM: A Simple Path towards Unified Dense Correspondence with Flow</h1>
<a href="https://uniflowmatch.github.io/assets/UFM.pdf"><img src="https://img.shields.io/badge/paper-blue" alt="Paper"></a>
<a href="https://arxiv.org/abs/2506.09278"><img src="https://img.shields.io/badge/arXiv-2506.09278-b31b1b" alt="arXiv"></a>
<a href="https://uniflowmatch.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href='https://huggingface.co/spaces/infinity1096/UFM'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>


**Carnegie Mellon University**

[Yuchen Zhang](https://infinity1096.github.io/), [Nikhil Keetha](https://nik-v9.github.io/), [Chenwei Lyu](https://www.linkedin.com/in/chenwei-lyu/), [Bhuvan Jhamb](https://www.linkedin.com/in/bhuvanjhamb/), [Yutian Chen](https://www.yutianchen.blog/about/), [Yuheng Qiu](https://haleqiu.github.io), [Jay Karhade](https://jaykarhade.github.io/), [Shreyas Jha](https://www.linkedin.com/in/shreyasjha/), [Yaoyu Hu](http://www.huyaoyu.com/), [Deva Ramanan](https://www.cs.cmu.edu/~deva/), [Sebastian Scherer](https://theairlab.org/team/sebastian/), [Wenshan Wang](http://www.wangwenshan.com/)
</div>

<p align="center">
    <img src="assets/teaser.jpg" alt="example" width=80%>
    <br>
    <em>UFM unifies the tasks of Optical Flow Estimation and Wide Baseline Matching and provides accurate dense correspondences for in-the-wild images at significantly fast inference speeds.</em>
</p>

## Updates
- [2026/04/03] UFM-G version initialized from DINOv2-G weights. More robust and accurate!
- [2026/02/20] Smaller UFM initialized from DINOv2 weights, faster at similar performance.
- [2025/10/21] Complete training and most data processing scripts. (branch: train)
- [2025/10/20] Benchmark & data script for primary results. (branch: benchmark)
- [2025/10/08] Released 980 resolution models.
- [2025/06/10] Initial release of model checkpoint and inference code.

## Stay Tuned for the Upcoming Updates!
- UFM-Tiny for real-time applications such as robotics.

## Models

| Checkpoint | Typical Runtime (ms, RTX 5090) | Parameters | HuggingFace |
|---|---|---|---|
| UFM-Base | 33  | 0.4B | [infinity1096/UFM-Base](https://huggingface.co/infinity1096/UFM-Base) |
| UFM-Refine | 44 | 0.4B | [infinity1096/UFM-Refine](https://huggingface.co/infinity1096/UFM-Refine) |
| UFM-Base-980 | 77 | 0.4B | [infinity1096/UFM-Base-980](https://huggingface.co/infinity1096/UFM-Base-980) |
| UFM-Refine-980 | 96 | 0.4B | [infinity1096/UFM-Refine-980](https://huggingface.co/infinity1096/UFM-Refine-980) |
| UFM-Base-DINOv2L-init | 28 | 0.3B | [infinity1096/UFM-Base-DINOv2L-init](https://huggingface.co/infinity1096/UFM-Base-DINOv2L-init) |
| UFM-Base-DINOv2G-init | 55 | 1B | [infinity1096/UFM-Base-DINOv2G-init](https://huggingface.co/infinity1096/UFM-Base-DINOv2G-init) |

## Overview

UFM (Unified Flow & Matching, UniFlowMatch) is a simple, end-to-end trained transformer model that directly regresses pixel displacement images (flow) and can be applied concurrently to both optical flow and wide-baseline matching tasks.

## Quick Start

### Installation

We use [UniCeption](https://github.com/castacks/UniCeption), a library which contains modular, config-swappable components for assembling end-to-end networks. To install UFM, recursively clone this repository and install the package with all dependencies:

```bash
git clone --recursive https://github.com/UniFlowMatch/UFM.git
cd UFM

# In case you cloned without --recursive:
# git submodule update --init

# Create and activate conda environment
conda create -n ufm python=3.11 -y
conda activate ufm

# Install UniCeption dependency
cd UniCeption
pip install -e .
cd ..

# Install UFM with all dependencies
pip install -e .

# Optional: Install with specific extras
# pip install -e ".[dev]"     # For development
# pip install -e ".[demo]"    # For demo
# pip install -e ".[all]"     # All optional dependencies

# Optional: For development and linting
pre-commit install  # Install pre-commit hooks
```

### Verify Installation

Verify your installation by running the basic model test:

```bash
# Test installation
ufm test

# Or run the basic model test
python uniflowmatch/models/ufm.py
```

Verify that `ufm_output.png` looks like `examples/example_ufm_output.png`.

### Command Line Interface

UFM provides a convenient CLI for common tasks:

```bash
# Test installation
ufm test

# Launch interactive demo
ufm demo

# Launch demo with specific settings
ufm demo --port 8080 --share --model refine

# Run inference on image pair
ufm infer source.jpg target.jpg --output results/

# Run inference with refinement model
ufm infer img1.png img2.png --model refine --output ./output
```

### Python API

```python
import cv2
import torch

# Load the base model (for general use)
from uniflowmatch.models.ufm import UniFlowMatchConfidence
model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base")

# Or load the refinement model (for higher accuracy)
from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement
model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")

# Choose from
# UFM-Base, UFM-Refine, UFM-Base-980, UFM-Refine-980, UFM-Base-DINOv2L-init, UFM-Base-DINOv2G-init

# Set the model to evaluation mode
model.eval()

# Load images using cv2 or PIL
source_image = cv2.imread("path/to/source.jpg")
target_image = cv2.imread("path/to/target.jpg")
source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Convert to torch tensors (uint8 or float32)
# Forward call takes care of normalizing uint8 images appropriate to the UFM model
source_image = torch.from_numpy(source_rgb)  # Shape: (H, W, 3)
target_image = torch.from_numpy(target_rgb)  # Shape: (H, W, 3)

# Predict correspondences
with torch.no_grad():
    result = model.predict_correspondences_batched(
        source_image=source_image,
        target_image=target_image,
    )

    flow = result.flow.flow_output[0].cpu().numpy()
    covisibility = result.covisibility.mask[0].cpu().numpy()
```

## Interactive Demo

### Online Demo

Try our online demo without installation: [🤗 Hugging Face Demo](https://huggingface.co/spaces/infinity1096/UFM)

### Local Gradio Demo

Run the interactive Gradio demo locally to visualize UFM outputs:

```bash
# Using the CLI (recommended)
ufm demo

# Or run directly
python gradio_demo.py

# Advanced options
ufm demo --port 8080 --share --model refine
```

## License

This code is licensed under a fully open-source [BSD-3-Clause license](LICENSE). The pre-trained UFM model checkpoints inherit the licenses of the underlying training datasets and as result, may not be used for commercial purposes (CC BY-NC-SA 4.0). Please refer to the respective training dataset licenses for more details.

Based on community interest, we can look into releasing an Apache 2.0 licensed version of the model in the future. Please upvote the issue [here](https://github.com/UniFlowMatch/UFM/issues/1#issue-3135416718) if you would like to see this happen.

## Acknowledgements

We thank the folowing projects for their open-source code: [DUSt3R](https://github.com/naver/dust3r), [MASt3R](https://github.com/naver/mast3r), [RoMA](https://github.com/Parskatt/RoMa), and [DINOv2](https://github.com/facebookresearch/dinov2).

## Citation
If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:

```bibtex
@inproceedings{zhang2025ufm,
 title={UFM: A Simple Path towards Unified Dense Correspondence with Flow},
 author={Zhang, Yuchen and Keetha, Nikhil and Lyu, Chenwei and Jhamb, Bhuvan and Chen, Yutian and Qiu, Yuheng and Karhade, Jay and Jha, Shreyas and Hu, Yaoyu and Ramanan, Deva and Scherer, Sebastian and Wang, Wenshan},
 booktitle={arXiV},
 year={2025}
}
```
