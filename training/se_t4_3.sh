#!/bin/bash

#SBATCH --account nvr_lpr_llm
#SBATCH --partition batch_block1,batch_short
#SBATCH --time 04:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name se_t4_3
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=se_t4_3.out
#SBATCH --error=se_t4_3.err

set -x

hostname -i
source ~/.bashrc
conda activate vllm1
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-2-9b-it --port 1409 &
CUDA_VISIBLE_DEVICES=1 vllm serve codellama/CodeLlama-7b-Instruct-hf --port 1410 &
CUDA_VISIBLE_DEVICES=2,3 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --enable-auto-tool-choice --tool-call-parser hermes --port 1411 --tensor-parallel-size 2 &

sleep 14000