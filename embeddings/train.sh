#! /bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=5g
#SBATCH -J "PatientEmbeddings"
#SBATCH -p acdemic
#SBATCH -t 3:00:00
#SBATCH --gres=gpu:2
generate_embeddings.py
