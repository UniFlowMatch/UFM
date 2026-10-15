
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

## Welcome to the Training Branch
If you reach here, you are interested in building upon UFM, thank you for your effort contributing to generalizable correspondence prediction!

This branch contains the data downloading and processing instructions, example and full training scripts used to create UFM.

## ✅ Data Processing Is Ready

We have completed cleaning the data processing, all data should be ready. Please let us know if something does not work.

## Updates
- [2025/10/26] Complete data source and processing scripts.
- [2025/10/21] Complete training and most data processing scripts.
- [2025/10/20] Benchmark & data script for primary results.
- [2025/10/08] Released 980 resolution models.
- [2025/06/10] Initial release of model checkpoint and inference code.

## What's Included

- **Data downloading and processing scripts** for recreate entire UFM data.
- **Example and complete training code** for recreating UFM training.
- **Logging and visualization support** to monitor training.

## Getting Started
### Setup Environment

You will need to `pip install -e .` again if you switched to here from the other branches.


### Setup Data

Please refer to `preprocess_datasets/README.md` for a complete guide. We provided preprocessed blendedmvs dataset for running the example training scripts.

### Setup Configuration

You need to setup data paths for your machine. please modify `configs/machine/your_machine.yaml` to another name and modify `root_data_dir` to point to the root folder of all `*_processed` folders. 

The `machines` configuration allows the same code to run on multiple places seamlessly. You can see how they work at the top of the example training scripts. 

### Example Training

Modify the logic for setting the `machine` environment variable in the file, and run: 

```bash
bash bash_scripts/examples/blendedmvs.sh
```

## Functional Components

|Component | Code location | Documentation |
|-------------|---------------|---------------|
|Base Model Architecture| `uniflowmatch/models/ufm.py` line 474 | paper Section `3.1`|
|Refinement Model Architecture| `uniflowmatch/models/ufm.py` line 710 | paper Section `3.1`|
|Computing Covisible Masks| `uniflowmatch/datasets/base/flow_postprocessing.py` line `544`| paper Appendix `A`|
|Robust EPE Loss|`uniflowmatch/loss/epe.py` | paper Section `3.4`, Appendix `H`|
|Refinement Loss|`uniflowmatch/loss/refinement_cross_entropy[_efficient].py`| paper Appendix `D`|
|Visualizations|`uniflowmatch/utils/viz.py`| |

## Entry Points
| Script | Function |
|--------|----------|
| `bash_scripts/examples/blendedmvs.sh` | Example training of the UFM base model |
| `bash_scripts/examples/refinement.sh` | Example training of the refinement network |
| `bash_scripts/training/megatraining_560.sh` | Train `UFM 560` |
| `bash_scripts/training/megatraining_980.sh` | Train `UFM 980` from `UFM 560` |
| `bash_scripts/training/megatraining_refinement_560.sh` | Train `UFM 560 Refine` from `UFM 560` |
| `bash_scripts/training/megatraining_refinement_980.sh` | Train `UFM 980 Refine` from `UFM 560` |


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
