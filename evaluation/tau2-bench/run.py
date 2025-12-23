# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import time
import subprocess, signal
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

SERVE_REPEAT = 1
serve_script = """#!/bin/bash

#SBATCH --account nvr_lpr_llm
#SBATCH --partition interactive
#SBATCH --time 04:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name EXPERIMENT_NAME
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=EXPERIMENT_NAME.out
#SBATCH --error=EXPERIMENT_NAME.err

set -x

hostname -i
source ~/.bashrc
source /lustre/fsw/portfolios/llmservice/users/sdiao/anaconda3/bin/activate vllm1
echo SHIZHE DEBUG HF_HOME: $HF_HOME
echo SHIZHE DEBUG USER_PATH: $USER_PATH
export VLLM_CACHE_ROOT="$USER_PATH/cache/vllm/EXPERIMENT_NAME_20"
CUDA_VISIBLE_DEVICES=0 vllm serve CHECKPOINT_DIR --enable-auto-tool-choice --tool-call-parser hermes --port 1900 &
sleep 60
export VLLM_CACHE_ROOT="$USER_PATH/cache/vllm/EXPERIMENT_NAME_21"
CUDA_VISIBLE_DEVICES=1 vllm serve CHECKPOINT_DIR --enable-auto-tool-choice --tool-call-parser hermes --port 1901 &
sleep 60
export VLLM_CACHE_ROOT="$USER_PATH/cache/vllm/EXPERIMENT_NAME_22"
CUDA_VISIBLE_DEVICES=2 vllm serve CHECKPOINT_DIR --enable-auto-tool-choice --tool-call-parser hermes --port 1902 &
sleep 60
export VLLM_CACHE_ROOT="$USER_PATH/cache/vllm/EXPERIMENT_NAME_23"
CUDA_VISIBLE_DEVICES=3 vllm serve CHECKPOINT_DIR --enable-auto-tool-choice --tool-call-parser hermes --port 1903 &
sleep 60
export VLLM_CACHE_ROOT="$USER_PATH/cache/vllm/EXPERIMENT_NAME_24"
CUDA_VISIBLE_DEVICES=4,5 vllm serve Qwen/Qwen3-32B --enable-auto-tool-choice --tool-call-parser hermes --port 1904 --tensor-parallel-size 2 &
sleep 60
export VLLM_CACHE_ROOT="$USER_PATH/cache/vllm/EXPERIMENT_NAME_25"
CUDA_VISIBLE_DEVICES=6,7 vllm serve Qwen/Qwen3-32B --enable-auto-tool-choice --tool-call-parser hermes --port 1905 --tensor-parallel-size 2  &
sleep 15000"""

def get_jobs():
    exec_result = subprocess.run(['squeue', '-u',os.environ.get('USER',None)], timeout=3600, capture_output=True, text=True)
    lines = exec_result.stdout.strip().split('\n')[1:]
    jobs = []
    for l in lines:
        components = l.split(' ')
        components = [e for e in components if e!='']
        running_time = components[5]
        total_time = 0
        time_components = running_time.split(':')
        if '-' in time_components[0]:
            total_time = 3600
        elif len(time_components)==2:
            total_time = int(time_components[0])*60+int(time_components[1])
        elif len(time_components)==3:
            total_time = int(time_components[0])*3600+int(time_components[1])*60+int(time_components[2])
        jobs.append({
            'name': components[2],
            'id': components[0],
            'status': components[4],
            'total_time': total_time,
            'reason': components[-1]
        })
    return jobs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str)
args = parser.parse_args()

SERVE_IPS = []
run_done = True
log("========== Starting main loop ==========")
loop_count = 0
while True:
    loop_count += 1
    log(f">>> Loop iteration {loop_count} started")
    jobs = get_jobs()
    log(f"Got {len(jobs)} jobs from squeue")
    for j in jobs:
        if j['reason'].strip().lower()=='held)':
            os.system(f"scancel {j['id']}")
            time.sleep(120)
    cur_ckpt_dir = os.getenv("CKPT_DIR")
    log(f"CKPT_DIR = {cur_ckpt_dir}")
    serve_collections = []
    for repeat in range(SERVE_REPEAT):
        exp_name = f"eaa_1{repeat}"
        serve_collections.append(exp_name)
        cur_serve_script = serve_script.replace('CHECKPOINT_DIR',cur_ckpt_dir)
        cur_serve_script = cur_serve_script.replace('EXPERIMENT_NAME',exp_name)
        with open(f'{exp_name}.sh','w') as f:
            f.write(cur_serve_script)
    log(f"Generated {SERVE_REPEAT} serve scripts: {serve_collections}")
    jobs = get_jobs()
    job_names = [j['name'] for j in jobs]
    for j in jobs:
        if j['name'] not in serve_collections and j['name'].startswith('eaa'):
            os.system(f"scancel {j['id']}")
    for repeat in range(SERVE_REPEAT):
        exp_name = f"eaa_1{repeat}"
        if not exp_name in job_names:
            log(f"Submitting new job: {exp_name}")
            if os.path.isfile(f'{exp_name}.out'):
                os.remove(f'{exp_name}.out')
                os.remove(f'{exp_name}.err')
            os.system(f'sbatch {exp_name}.sh')
    job_ids = [j['id'] for j in jobs]
    already_serve = []
    for j in jobs:
        if j['name'] in serve_collections and j['status'].strip().lower()=='r':
            if not os.path.isfile(f'{j["name"]}.out'):
                os.system(f"scancel {j['id']}")
            else:
                if j['total_time']>=600:
                    log(f"Server {j['name']} ready after {j['total_time']}s")
                    already_serve.append({
                        'name': j['name'],
                        'total_time': j['total_time']
                    })
                else:
                    log(f"Server {j['name']} not ready long enough, waiting {600-j['total_time']}s...")
    if len(already_serve)==0:
        log("No ready servers yet, waiting 30s...")
        time.sleep(30)
        continue
    log(f"Found {len(already_serve)} ready servers: {already_serve}")
    all_times = [s['total_time'] for s in already_serve]
    # if max(all_times)<600:
    #     wait_time = 600-max(all_times)
    #     log(f"Servers not ready long enough, waiting {wait_time}s...")
    #     time.sleep(wait_time)
    serve_ips = []
    for s in already_serve:
        with open(f'{s["name"]}.out') as f:
            lines = f.readlines()
        serve_ip = lines[0].strip()
        serve_ips.append(serve_ip)
    log(f"Collected serve IPs: {serve_ips}")
    change_flag = False
    if os.path.isfile('eaa.json'):
        with open('eaa.json') as f:
            old_config = json.load(f)
        if not cur_ckpt_dir in old_config:
            change_flag = True
    if SERVE_IPS!=serve_ips or change_flag:
        log(f"Config changed (IPs changed: {SERVE_IPS!=serve_ips}, ckpt changed: {change_flag}), updating eaa.json...")
        SERVE_IPS = serve_ips
        model_config = {cur_ckpt_dir:[],'Qwen/Qwen3-32B':[]}
        for sip in serve_ips:
            model_config[cur_ckpt_dir].append({"ip_addr": sip,"port": "1900"})
            model_config[cur_ckpt_dir].append({"ip_addr": sip,"port": "1901"})
            model_config[cur_ckpt_dir].append({"ip_addr": sip,"port": "1902"})
            model_config[cur_ckpt_dir].append({"ip_addr": sip,"port": "1903"})
            model_config['Qwen/Qwen3-32B'].append({"ip_addr": sip,"port": "1904"})
            model_config['Qwen/Qwen3-32B'].append({"ip_addr": sip,"port": "1905"})
        model_config['vllm_model_config_path'] = 'eaa.json'
        with open('eaa.json','w') as f:
            json.dump(model_config,f,indent=2)
        log("eaa.json updated successfully")
    REPO_PATH = os.environ.get('REPO_PATH')
    retail_task_path = os.path.join(REPO_PATH, 'data/tau2/domains/retail/tasks.json')
    telecom_task_path = os.path.join(REPO_PATH, 'data/tau2/domains/telecom/tasks.json')
    airline_task_path = os.path.join(REPO_PATH, 'data/tau2/domains/airline/original_tasks.json')
    log("========== Starting evaluation: RETAIL ==========")
    os.system(f"python tau2/cli.py --domain retail --agent-llm {cur_ckpt_dir} "
            f"--user-llm gpt-5 --num-trials 1 --task_path {retail_task_path} "
            f"--max-steps 200 --output_file outputs/retail.json "
            f"--model_config_path eaa.json --use_model_tool")
    log("========== Finished RETAIL, Starting: TELECOM ==========")
    os.system(f"python tau2/cli.py --domain telecom --agent-llm {cur_ckpt_dir} "
            f"--user-llm gpt-5 --num-trials 1 --task_path {telecom_task_path} "
            f"--max-steps 200 --output_file outputs/telecom.json "
            f"--model_config_path eaa.json --use_model_tool")
    log("========== Finished TELECOM, Starting: AIRLINE ==========")
    os.system(f"python tau2/cli.py --domain airline --agent-llm {cur_ckpt_dir} "
            f"--user-llm gpt-5 --num-trials 1 --task_path {airline_task_path} "
            f"--max-steps 200 --output_file outputs/airline.json "
            f"--model_config_path eaa.json --use_model_tool")
    log("========== Finished AIRLINE, loop iteration complete ==========")


        

