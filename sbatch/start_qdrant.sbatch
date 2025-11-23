#!/bin/bash
#SBATCH -p dgx2
#SBATCH -A eecs
#SBATCH --job-name=qdrant_server
#SBATCH -t 1-00:00:00
#SBATCH -c 2
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

module load gcc
./deps/qdrant/qdrant
