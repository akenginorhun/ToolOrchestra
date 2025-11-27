#!/bin/bash

#SBATCH --account nvr_lpr_llm
#SBATCH --partition batch_block1,batch_short
#SBATCH --time 04:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name se_t4_2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=se_t4_2.out
#SBATCH --error=se_t4_2.err

set -x

hostname -i
source ~/.bashrc
conda activate vllm1
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-3.3-70B-Instruct --enable-auto-tool-choice --tool-call-parser llama3_json --chat-template tool_chat_template_llama3.1_json.jinja --tensor-parallel-size 4 --port 1405 &
CUDA_VISIBLE_DEVICES=4,5 vllm serve Qwen/Qwen3-32B --enable-auto-tool-choice --tool-call-parser hermes --port 1406 --tensor-parallel-size 2 &
CUDA_VISIBLE_DEVICES=6,7 vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 1407 --tensor-parallel-size 2

sleep 14000