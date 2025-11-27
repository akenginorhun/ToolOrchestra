#!/bin/bash

#SBATCH --account nvr_lpr_llm
#SBATCH --partition batch_block1,batch_short
#SBATCH --time 04:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name se_t4_1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=se_t4_1.out
#SBATCH --error=se_t4_1.err

set -x

hostname -i
source ~/.bashrc
conda activate retriever
CUDA_VISIBLE_DEVICES=0 python retrieval_general_thought.py --port 1401 &
sleep 60
conda activate vllm1
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-auto-tool-choice --tool-call-parser llama3_json --chat-template tool_chat_template_llama3.1_json.jinja --port 1402 &
sleep 60
CUDA_VISIBLE_DEVICES=2 vllm serve microsoft/Phi-4-mini-instruct --port 1408 &
sleep 60
CUDA_VISIBLE_DEVICES=3,4,5,6 vllm serve Qwen/Qwen2.5-Math-72B-Instruct --port 1403 --tensor-parallel-size 4 &
sleep 60
CUDA_VISIBLE_DEVICES=7 vllm serve Qwen/Qwen2.5-Math-7B-Instruct --port 1404

sleep 14000