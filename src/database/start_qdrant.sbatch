#!/bin/bash
#SBATCH -w cn-m-1
#SBATCH -p soundbendor
#SBATCH -A soundbendor
#SBATCH --job-name=qdrant_server
#SBATCH -t 0
#SBATCH -c 2
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

module load gcc
bash deps/qdrant
