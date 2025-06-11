
<div align="center">
<h1>UFM: A Simple Path towards Unified Dense Correspondence with Flow</h1>
<p align="center">
  <a href="https://arxiv.org/abs/0000.00000">
    <img src="https://img.shields.io/badge/Paper-blue" alt="Paper">
  </a>
  <a href="https://arxiv.org/abs/0000.00000">
    <img src="https://img.shields.io/badge/arXiv-0000.00000-b31b1b" alt="arXiv">
  </a>
  <a href="https://uniflowmatch.github.io/">
    <img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page">
  </a>
  <a href="https://huggingface.co/spaces/infinity1096/UFM">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" alt="🤗 Hugging Face Demo">
  </a>
</p>


**Carnegie Mellon University**

[Yuchen Zhang](https://infinity1096.github.io/), [Nikhil Keetha](https://nik-v9.github.io/), [Chenwei Lyu](https://www.linkedin.com/in/chenwei-lyu/), [Bhuvan Jhamb](https://www.linkedin.com/in/bhuvanjhamb/), [Yutian Chen](https://www.yutianchen.blog/about/), [Yuheng Qiu](https://haleqiu.github.io), [Jay Karhade](https://jaykarhade.github.io/), [Shreyas Jha](https://www.linkedin.com/in/shreyasjha/), [Yaoyu Hu](http://www.huyaoyu.com/), [Deva Ramanan](https://www.cs.cmu.edu/~deva/), [Sebastian Scherer](https://theairlab.org/team/sebastian/), [Wenshan Wang](http://www.wangwenshan.com/)
</div>

<p align="center">
    <img src="assets/teaser.jpg" alt="example" width=80%>
    <br>
    <em>UFM unifies the tasks of Optical Flow Estimation and Wide Baseline Matching and provides accurate dense correspondences for in-the-wild images at significantly fast inference speeds.</em>
</p>

## Updates
- [2025/06/10] UFM training and benchmarking code coming soon! Star the repository to get notified when we release it.
- [2025/06/10] Initial release of model checkpoint and inference code.

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
import torch

# Load the base model (for general use)
from uniflowmatch.models.ufm import UniFlowMatchConfidence
model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base")

# Or load the refinement model (for higher accuracy)
from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement
model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")

# Set the model to evaluation mode
model.eval()

# Load your images (as torch tensors)
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
