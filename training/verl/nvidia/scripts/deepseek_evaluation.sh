#!/bin/bash

#SBATCH -A nvr_lpr_agentic
#SBATCH -p interactive
#SBATCH -t 04:00:00
#SBATCH -N2
#SBATCH --gpus-per-node=8
#SBATCH -J nvr-system2:r1_infer2-8k-verlv2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton


set -x
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1

# PROJECT="code-reasoning-nemotron"
# EXPNAME="math-code-gym-stem-ifeval-grpo-8b-8k-length"
# CKPT="global_step_500"
# EXPNAME="math-code-gym-stem-ifeval-grpo-8b-8k-dynamic-fix"
# CKPT="global_step_000"
PROJECT="code-reasoning"
EXPNAME="math-code-gym-stem-ifeval-grpo-1.5b-8k-8192-grpo-clean"
CKPT="global_step_85"
# get task type argument, default is math, # option: math, reasoning, code ...
TASK_TYPE=${1:-math}

# Fixed parameters
TEMPERATURE=0.6
N_SAMPLES=16
RESPONSE_LENGTH=32768

HOME="/lustre/fsw/portfolios/nvr/users/jianh"
HOME2="/lustre/fsw/portfolios/nvr/users/mingjiel"
GPFS="$HOME/projects/verl_internal2"

MODEL="$GPFS/target/results/$PROJECT/$EXPNAME/ckpt/$CKPT/actor/huggingface"
# MODEL="/lustre/fsw/portfolios/nvr/users/yidong/data/models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
# MODEL="./results/code-reasoning/$EXPNAME/ckpt/global_step_2000/actor/huggingface"
echo "Checkpoint directory: $MODEL"
BATCH_SIZE=2048

# TODO: add image name
container_name="$HOME/data/images/nvidian+nemo+verl_v2_dev.sqsh"

SANDBOX_IMAGE="$HOME2/containers/nvidian+nemo+reasoninggym.sqsh"
CODE_SANDBOX_IMAGE="$HOME/data/images/code_sandbox+dev.sqsh"

MOUNTS="--container-mounts=${GPFS}:${GPFS},/lustre:/lustre,${GPFS}:/verl"
export HF_HOME="$HOME/.cache/huggingface"

RESULTS_DIR="$GPFS/target/tests/$PROJECT/$EXPNAME"
mkdir -p $RESULTS_DIR
OUTPUT_DIR="$GPFS/target/tests/$PROJECT/$EXPNAME/ckpt/$CKPT/eval"
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

#server_ip
server_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{printf "\"%s\",", $0}')
server_ip=${server_ip%,}  # Remove trailing comma
echo "server_ip: $server_ip"

worker_num=$((SLURM_JOB_NUM_NODES))
# start code sandbox on all nodes
# use 64 cpus 
for ((i = 0; i < worker_num; i++)); do
    node_i=${nodes_array[$i]}
    srun --nodes=1 --cpus-per-task=64 --ntasks=1 -w $node_i -o "$RESULTS_DIR/sandbox-code-%j-worker-node-$i.out" -e "$RESULTS_DIR/sandbox-code-%j-worker-node-$i.err" --no-container-mount-home --overlap --container-image=$CODE_SANDBOX_IMAGE bash -c "cd /workspace && export PYTHONPATH=/workspace/verl_internal && python /workspace/verl_internal/scripts/remote_reward_server/launch_default_reward_server.py" &
done
sleep 10s
 
# start reasoning gym sandbox on all nodes
# use 64 cpus 
for ((i = 0; i < worker_num; i++)); do
    node_i=${nodes_array[$i]}
    srun --nodes=1 --cpus-per-task=2 --ntasks=1 -w $node_i -o "$RESULTS_DIR/sandbox-reasoning-%j-head-node-$i.out" -e "$RESULTS_DIR/sandbox-reasoning-%j-head-node-$i.err" --mem=524288M --overlap --no-container-mount-home --container-image=${SANDBOX_IMAGE} bash -c "python scripts/host_reward_server.py" &
wait 30

done
sleep 10s

port=6379
ip_head=$head_node_ip:$port
export ip_head

headnode_cmd="ray start --head --node-ip-address=$head_node_ip --port=$port "' --num-cpus=$(lscpu -p=CPU | grep -v "#" | wc -l) --block --disable-usage-stats --include-dashboard=true --dashboard-port=8265 --ray-client-server-port 6478 --redis-shard-ports 6580 --dashboard-grpc-port 6681 --node-manager-port 6782 --object-manager-port 6883 --runtime-env-agent-port 6984 --dashboard-agent-grpc-port 7085 --dashboard-agent-listen-port 7186 --metrics-export-port 7297'

worker_cmd="ray start --address $ip_head "' --num-cpus=$(lscpu -p=CPU | grep -v "#" | wc -l)'

srun --overlap --nodes=1 --ntasks=1 -w $head_node -o "$RESULTS_DIR/output-%j-head-node.out" -e "$RESULTS_DIR/output-%j-head-node.err" --no-container-mount-home --container-image="$container_name" $MOUNTS $EXPORTS bash -c "$headnode_cmd" &
sleep 30s

worker_num=$((SLURM_JOB_NUM_NODES))
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes_array[$i]}
    base_port=$((30000 + (i-1)*100)) 
    worker_min_port=$base_port
    worker_max_port=$((base_port + 4999))
    srun --overlap --nodes=1 --ntasks=1 -w $node_i -o "$RESULTS_DIR/output-%j-worker-node-$i.out" -e "$RESULTS_DIR/output-%j-worker-node-$i.err" --no-container-mount-home --container-image="$container_name" $MOUNTS $EXPORTS bash -c "$worker_cmd --block --disable-usage-stats --node-manager-port 6782 --object-manager-port 6883 --runtime-env-agent-port 6984 --dashboard-agent-grpc-port 7085 --dashboard-agent-listen-port 7186 --metrics-export-port 7297" &
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
srun --overlap --nodes=1 --ntasks=1 -w $head_node -o "$RESULTS_DIR/checker-%j-head-node.out" -e "$RESULTS_DIR/checker-%j-head-node.err" --no-container-mount-home --container-image="$container_name" $MOUNTS $EXPORTS bash -c "$check_cmd"

if [ "$TASK_TYPE" == "math" ]; then
    eval_cmd="cd /verl
              export PYTHONPATH=/verl
              bash verl/nvidia/scripts/eval_model.sh --batch-size ${BATCH_SIZE} --server-ip ${server_ip} --num-nodes $worker_num --model $MODEL --data-dir $HOME2/data/deepscaler --datasets aime_2025 aime amc math minerva olympiad_bench --output-dir $OUTPUT_DIR --temperature $TEMPERATURE --n-samples $N_SAMPLES --response-length $RESPONSE_LENGTH"
elif [ "$TASK_TYPE" == "reasoning" ]; then
    eval_cmd="cd /verl
              export PYTHONPATH=/verl
              bash verl/nvidia/scripts/eval_model.sh --batch-size ${BATCH_SIZE} --server-ip ${server_ip} --num-nodes $worker_num --model $MODEL --data-dir $HOME2/data/reasoning_gym_dedup --datasets ab advanced_geometry aiw arc_1d arc_agi base_conversion basic_arithmetic bf binary_alternation binary_matrix bitwise_arithmetic caesar_cipher calendar_arithmetic chain_sum circuit_logic codeio color_cube_rotation complex_arithmetic count_bits countdown count_primes course_schedule cryptarithm decimal_arithmetic decimal_chain_sum dice emoji_mystery family_relationships figlet_font fraction_simplification futoshiki game_of_life gcd graph_color group_anagrams gsm_symbolic intermediate_integration isomorphic_strings jugs knights_knaves knight_swap largest_island lcm leg_counting letter_counting letter_jumble list_functions mahjong_puzzle manipulate_matrix maze mini_sudoku needle_haystack n_queens number_filtering number_format number_sequence number_sorting palindrome palindrome_partitioning polynomial_equations polynomial_multiplication pool_matrix power_function prime_factorization products propositional_logic quantum_lock ransom_note rearc rectangle_count rotate_matrix rotten_oranges rubiks_cube rush_hour self_reference sentence_reordering shortest_path simple_equations simple_geometry simple_integration sokoban spell_backward spiral_matrix string_insertion string_manipulation string_splitting string_synthesis sudoku syllogism time_intervals tower_of_hanoi tsumego word_ladder word_sequence_reversal word_sorting zebra_puzzles --output-dir $OUTPUT_DIR --temperature $TEMPERATURE --n-samples $N_SAMPLES --response-length $RESPONSE_LENGTH"
elif [ "$TASK_TYPE" == "code" ]; then
    eval_cmd="cd /verl
              export PYTHONPATH=/verl
              bash verl/nvidia/scripts/eval_model.sh --batch-size ${BATCH_SIZE} --server-ip ${server_ip} --num-nodes $worker_num --model $MODEL --data-dir $HOME2/data/eurus2-rl-data --datasets codeforces apps codecontests taco --output-dir $OUTPUT_DIR --temperature $TEMPERATURE --n-samples $N_SAMPLES --response-length $RESPONSE_LENGTH"
elif [ "$TASK_TYPE" == "gpqa" ]; then
    eval_cmd="cd /verl
              export PYTHONPATH=/verl
              bash verl/nvidia/scripts/eval_model.sh --batch-size ${BATCH_SIZE} --server-ip ${server_ip} --num-nodes $worker_num --model $MODEL --data-dir $HOME2/data/test --datasets gpqa_diamond --output-dir $OUTPUT_DIR --temperature $TEMPERATURE --n-samples $N_SAMPLES --response-length $RESPONSE_LENGTH"
elif [ "$TASK_TYPE" == "ifeval" ]; then
    eval_cmd="cd /verl
              export PYTHONPATH=/verl
              bash verl/nvidia/scripts/eval_model.sh --batch-size ${BATCH_SIZE} --server-ip ${server_ip} --num-nodes $worker_num --model $MODEL --data-dir $HOME/data/training_data/ifeval/ --datasets if_eval_google --output-dir $OUTPUT_DIR --temperature $TEMPERATURE --n-samples $N_SAMPLES --response-length $RESPONSE_LENGTH"
else
    echo "Invalid task type: $TASK_TYPE"
    exit 1
fi

srun --nodes=1 --ntasks=1 -w $head_node -o "$RESULTS_DIR/command-%j-head-node.out" -e "$RESULTS_DIR/command-%j-head-node.err" --no-container-mount-home  --overlap --container-image="$container_name" $MOUNTS $EXPORTS bash -c "$eval_cmd"

exit 0
