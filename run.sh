#!/bin/bash
#SBATCH -p ma2-gpu
#SBATCH -w compute-4-26
#SBATCH --gres gpu:1
#SBATCH --mem=46MB
#SBATCH --job-name dash
#SBATCH --time=2-0:00:00
#SBATCH -o dash_cohn1.out
#SBATCH -e dash_cohn1.err




DS="human_enhancers_cohn" # SPHERICAL DARCY-FLOW-5 PSICOV COSMIC NINAPRO FSD ECG SATELLITE DEEPSEA MNIST MUSIC
# DEEPSEA_FULL, human_enhancers_cohn, human_enhancers_ensembl, human_ocr_ensembl



for SEED in 1; do
    
    for i in $DS ; do
        # python3 -W ignore main.py --dataset $i --baseline 1 --seed $SEED
        python -W ignore main.py --dataset $i --arch unet --experiment_id unet --seed $SEED --valid_split 0 # valid split 0, 1
    done

done

# speed test
# python3 speed.py --experiment_id 0
# python3 speed.py --experiment_id 0 --test_input_size 1
