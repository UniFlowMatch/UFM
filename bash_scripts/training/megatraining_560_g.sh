#!/bin/bash

# Change this to fit your PC and cluster setup
if [ -e "/ocean/projects" ]; then
    if [ -e "/local/datasets" ]; then
        machine=psc_local
    else
        machine=psc
    fi
else
    if [ -e "/data2" ]; then
        machine=cluster
    else
        machine=yuchen
    fi
fi

echo "Machine: ${machine}"
# change batch size to 32 for 8 gpus
python scripts/train.py \
    --config-name default \
    model=uniflowmatch_covisibility_giant \
    dataset=ufm_560_all \
    loss=robust_epe_covisibility \
    machine=${machine} \
    training_scheme.effective_batch_size=24 \
    training_scheme.num_epochs=48 \
    training_scheme.warmup_epochs=4.8 \
    training_scheme.learning_rate.encoder=1e-6 \
    training_scheme.learning_rate.info_sharing=2e-5 \
    training_scheme.learning_rate.output_head=1e-4 \
    training_scheme.learning_rate.uncertainty_head=1e-4 \
    dataset.num_workers=12 \
    find_unused_parameters=True \
    visualizer=flow_covisible \
    training_scheme.validate_every_epochs=1 \
    training_scheme.accumulate_grad_batches=1 \
    training_scheme.symmetrize_inputs=True \
    hydra.run.dir=outputs/training/ufm_giant_560_new \
    +wandb_tags=[ufm_giant_24_16_560] \
    +wandb_name=ufm_giant24_16_560 \
    resume_all='"/jet/home/yzhang25/UFM/outputs/training/ufm_giant_560_new/checkpoints_epoch/last.ckpt"'
    # resume_all='"/jet/home/yzhang25/UFM/outputs/training/ufm_giant_560/checkpoints_epoch/last.ckpt"'
    # resume_model=infinity1096/UFM-Base # load from huggingface, you can also load from .ckpt files
    # resume_all='"outputs/training/ufm_giant_560/checkpoints_epoch/last.ckpt"' # resume from partial training