#!/bin/bash
#SBATCH -p dgx2
#SBATCH -A dgx2
#SBATCH --job-name=intern-fw
#SBATCH -t 1-00:00:00
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --output=/nfs/guille/eecs_research/soundbendor/beerya/intern-fw/logs/%j.out

module load cuda/12.4
source /nfs/guille/eecs_research/soundbendor/beerya/miniforge3/bin/activate
source activate intern-fw
eval_embed experiment_config.ini
