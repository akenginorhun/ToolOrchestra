#!/bin/bash

#SBATCH --account nvr_lpr_llm
#SBATCH --partition batch_block1,interactive
#SBATCH --time 04:00:00
#SBATCH --nodes 2
#SBATCH --gpus-per-node=8
#SBATCH --job-name run
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton

set -x



ROLLOUT=8
ALGO="grpo"
TRAIN_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=16
PPO_EPOCHS=1
max_prompt_length=24000
max_response_length=8768

# FIXED
LR=1e-6
TEMP=1.0
ENTROPY=0

# Cosine length penalty
MARGIN=-1
MIN_VALUE_CORRECT=0.5
MAX_VALUE_WRONG=0.0
REPETITION_PENALTY=-0.001
LENGTH_PENALTY_TYPE="cosine"

export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1

export WANDB_API_KEY="YOUR_WANDB_KEY"
MODEL_PATH="Qwen/Qwen3-8B"
MODEL_TYPE="Qwen/Qwen3-8B"
GPFS="train"

export CLIENT_ID="CLIENT_ID"
export CLIENT_SECRET="CLIENT_SECRET"

TRAIN_DATA="['data.jsonl']"

VAL_DATA="['data.jsonl']"

PROJECT="project"
EXPNAME="orchestra"
CKPT_DIR="outputs/orchestra/ckpt"
RESULTS_DIR="outputs/orchestra"
TRANSFER_DIR="outputs/orchestra/transfer"
mkdir -p $RESULTS_DIR
mkdir -p $CKPT_DIR

MAIN_CONTAINER="verl/image/path"

MOUNTS="--container-mounts=${GPFS}:${GPFS},/lustre:/lustre,${GPFS}:/verl"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

#server_ip
server_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{printf "\"%s\",", $0}')
server_ip=${server_ip%,}  # Remove trailing comma
echo "server_ip: $server_ip"

worker_num=$((SLURM_JOB_NUM_NODES))

port=6379
ip_head=$head_node_ip:$port
export ip_head

headnode_cmd="ray start --head --node-ip-address=$head_node_ip --port=$port "' --num-cpus=$(lscpu -p=CPU | grep -v "#" | wc -l) --block --disable-usage-stats --include-dashboard=true --dashboard-port=8265 --ray-client-server-port 6478 --redis-shard-ports 6580 --dashboard-grpc-port 6681 --node-manager-port 6782 --object-manager-port 6883 --runtime-env-agent-port 6984 --dashboard-agent-grpc-port 7085 --dashboard-agent-listen-port 7186 --metrics-export-port 7297'

worker_cmd="ray start --address $ip_head "' --num-cpus=$(lscpu -p=CPU | grep -v "#" | wc -l)'

srun --overlap --nodes=1 --ntasks=1 -w $head_node -o "$RESULTS_DIR/output-%j-head-node.out" -e "$RESULTS_DIR/output-%j-head-node.err" --no-container-mount-home --container-image="$MAIN_CONTAINER" $MOUNTS $EXPORTS bash -c "$headnode_cmd" &
sleep 30s

worker_num=$((SLURM_JOB_NUM_NODES))
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes_array[$i]}
    base_port=$((30000 + (i-1)*100))
    worker_min_port=$base_port
    worker_max_port=$((base_port + 4999))
    srun --overlap --nodes=1 --ntasks=1 -w $node_i -o "$RESULTS_DIR/output-%j-worker-node-$i.out" -e "$RESULTS_DIR/output-%j-worker-node-$i.err" --no-container-mount-home --container-image="$MAIN_CONTAINER" $MOUNTS $EXPORTS bash -c "$worker_cmd --block --disable-usage-stats --node-manager-port 6782 --object-manager-port 6883 --runtime-env-agent-port 6984 --dashboard-agent-grpc-port 7085 --dashboard-agent-listen-port 7186 --metrics-export-port 7297" &
done

check_cmd="while true; do
    num_nodes=\$(ray list nodes | grep node_id | wc -l)
    echo found \$num_nodes
    if [ \$num_nodes -eq $worker_num ]; then
        break
    fi
    echo sleeping
    sleep 3s
done"
srun --overlap --nodes=1 --ntasks=1 -w $head_node -o "$RESULTS_DIR/checker-%j-head-node.out" -e "$RESULTS_DIR/checker-%j-head-node.err" --no-container-mount-home --container-image="$MAIN_CONTAINER" $MOUNTS $EXPORTS bash -c "$check_cmd"

srun --overlap --nodes=1 --ntasks=1 -w $head_node -o "$RESULTS_DIR/command-%j-head-node.out" -e "$RESULTS_DIR/command-%j-head-node.err" --no-container-mount-home --container-image="$MAIN_CONTAINER" $MOUNTS $EXPORTS bash -c \
"ray status && lscpu && ray job submit --address=http://localhost:8265 \
    --runtime-env-json='{\"working_dir\": \"/verl\"}' \
    -- python3 -u -m recipe.algo.main_grpo_quick3 \
    +data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=$ALGO \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_tool_config_path=tools.json \
    data.test_tool_config_path=tools.json \
    +data.vllm_model_configs=serve_train_tool_orchestra.json \
    +data.my_output_dir=$RESULTS_DIR \
    +data.cur_transfer_dir=$TRANSFER_DIR \
    +data.model_type=$MODEL_TYPE \
    +data.topk_doc=30 \
    +data.exp_tag=orchestra \
    +data.use_llm_reward=true \
    +data.efficiency_reward=true \
    +data.use_qa_reward=true \
    data.prompt_template=qwen-base \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.gen_batch_size=384 \
    data.val_batch_size=64 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='left' \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=seq_reward \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=$LR \
    +actor_rollout_ref.rollout.n_agent=$ROLLOUT \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMP \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.correct_sample_advantage_boost_value=0.1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    stop_properly_penalty.penalty_coef=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT \
    trainer.experiment_name=$EXPNAME \
    trainer.val_before_train=False \
    trainer.log_val_generations=8 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$SLURM_JOB_NUM_NODES \
    trainer.save_freq=1 \
    trainer.test_freq=1000000 \
    +max_turns=30 \
    +retriever.url=http://127.0.0.1:8000/retrieve \
    +retriever.topk=5 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.total_epochs=10 \
    reward_manager.type=match \
    reward_manager.max_concurrency=256 \
    reward_manager.use_remote_reward=True \
    reward_manager.server_ip=[$server_ip]"

