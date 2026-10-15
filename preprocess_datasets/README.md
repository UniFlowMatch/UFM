# Data Prepration

This script handles how to download and prepare all data required for UFM training. We created a [UFM Data Mirror](https://huggingface.co/collections/infinity1096/ufm-data-mirror) at HuggingFace that hosts some of the preprocessed version.

## Metadata (from DUSt3R)

Download all the pairing information from [DUSt3R](https://github.com/naver/dust3r/tree/main) and place them under `data_root/dust3r_data_pairs`. 

## BlendedMVS
Download BlendedMVS from [here](https://github.com/YoYo000/BlendedMVS) and follow DUSt3R's preprocessing [here](https://github.com/naver/dust3r/blob/main/datasets_preprocess/preprocess_blendedMVS.py).

Alternatively, download the zipped processed version from [UFM Data Mirror](https://huggingface.co/datasets/infinity1096/blendedmvs_processed).

## MegaDepth
Follow DUSt3R's preprocessing script at [here](https://github.com/naver/dust3r/tree/main/datasets_preprocess).

## TA_WB Training Set
Download TA-WB from [here](https://huggingface.co/datasets/infinity1096/TA-WB) and place it as

```
data_root/TartanAir/assembled/tartanair_640_mega_training_0203_pinhole_good/
    train/
    val/
```

## Scannet++ V2

Download the pairs metadata from [here](https://huggingface.co/datasets/infinity1096/ufm_metadata) to be used with the preprocessing scripts.

Download Scannet++ V2 from [here](https://scannetpp.mlsg.cit.tum.de/scannetpp/) and follow `preprocess_datasets/preprocess_scannetpp.py`.

## Habitat CAD
Follow DUSt3R's preprocessing script at [here](https://github.com/naver/dust3r/tree/main/datasets_preprocess/habitat), but only process the `ReplicaCAD` data. Place the processed data as

```
data_root/
  habitat_processed/
    replica_cad_baked_lighting/
```

## Static Things
Follow DUSt3R's preprocessing script at [here](https://github.com/naver/dust3r/blob/main/datasets_preprocess/preprocess_staticthings3d.py). Place the processed data as `data_root/staticthings3d_processed/TRAIN`

## Kubrics 4D

Download the training set from [here](https://gcd.cs.columbia.edu/gcd_kubric4d_train_dl.sh) [dataset project page](https://gcd.cs.columbia.edu) and the precomputed metadata files from [here](https://huggingface.co/datasets/infinity1096/kubric4d_metadata/tree/main), place the files as:

```
kubric4d/
  kubric4d_raw/
    scn00000/
    scn00001/
    ...
  kubric4d_metadata/
```

## FlyingThings

```bash
bash preprocess_datasets/download_scripts/download_flyingthings.sh [data_root]
```

It will download raw data to `[data_root]/raw_data/FlyingThings`. (Download source: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

Then run the `preprocess_datasets/preprocess_flyingthings.py`

We provide the processed version on [UFM Data Mirror](https://huggingface.co/datasets/infinity1096/flyingthings3d_processed).

## FlyingChairs

```bash
bash preprocess_datasets/download_scripts/download_flyingchairs.sh [data_root]
```

It will directly setup `[data_root]/FlyingChairs`. (Download source: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

Download [the validation split](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs_train_val.txt) and place it under `FlyingChairs/`.

## Spring
Download the training set from https://spring-benchmark.org and use `preprocess_datasets/preprocess_spring.py`.

## Monkaa

```bash
bash preprocess_datasets/download_scripts/download_monkaa.sh [data_root]
```

It will download raw data to [data_root]/raw_data/Monkaa. (Download source: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

Then run `preprocess_datasets/preprocess_monkaa.py`

Alternatively, download the zipped processed version from [UFM Data Mirror](https://huggingface.co/datasets/infinity1096/hd1k_processed).

## HD1K
The original data is hosted at [here](https://hci.iwr.uni-heidelberg.de/benchmarks). However, since the site is unreachable, we host a copy of [the original data](infinity1096/HD1K-Backup). 

process the data with `preprocess_datasets/preprocess_hd1k.py`.

Alternatively, download the zipped processed version from [UFM Data Mirror](https://huggingface.co/datasets/infinity1096/hd1k_processed).

## In case some dataset is hard to work with
You can always try to train with less data, by modifying `configs/dataset/quantity_options`.
