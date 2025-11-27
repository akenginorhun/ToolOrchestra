# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import time
import subprocess
import requests
from datetime import datetime
import pytz

def print_time():
    la_timezone = pytz.timezone('America/Los_Angeles')
    current_time_la = datetime.now(la_timezone)
    print(f"Current time: {current_time_la.strftime('%Y-%m-%d %H:%M:%S')}")

def get_jobs():
    exec_result = subprocess.run(['squeue', '-u',os.environ.get('USER',None)], timeout=3600, capture_output=True, text=True)
    lines = exec_result.stdout.strip().split('\n')[1:]
    jobs = []
    for l in lines:
        components = l.split(' ')
        components = [e for e in components if e!='']
        running_time = components[5]
        total_time = None
        time_components = running_time.split(':')
        while total_time is None:
            if '-' in time_components[0]:
                total_time = 3600
            else:
                try:
                    if len(time_components)==2:
                        total_time = int(time_components[0])*60+int(time_components[1])
                    elif len(time_components)==3:
                        total_time = int(time_components[0])*3600+int(time_components[1])*60+int(time_components[2])
                except Exception as error:
                    print(error,time_components)
                    time.sleep(10)
        jobs.append({
            'name': components[2],
            'id': components[0],
            'status': components[4],
            'total_time': total_time,
            'reason': components[-1]
        })
    return jobs

EXPERIMENT_NAME1 = os.environ.get('EXPERIMENT_NAME1', 'se_t4_1')
EXPERIMENT_NAME2 = os.environ.get('EXPERIMENT_NAME2', 'se_t4_2')
EXPERIMENT_NAME3 = os.environ.get('EXPERIMENT_NAME3', 'se_t4_3')
serve_collections = [EXPERIMENT_NAME1, EXPERIMENT_NAME2, EXPERIMENT_NAME3]
while True:
    jobs = get_jobs()
    for j in jobs:
        if j['reason'].strip().lower()=='held)':
            os.system(f"scancel {j['id']}")
            time.sleep(120)
    job_names = [j['name'] for j in jobs]
    for exp_name in serve_collections:
        if not exp_name in job_names:
            from filelock import FileLock
            with FileLock(f'cache/slurm_out/{exp_name}.lock'):
                if os.path.isfile(f'cache/slurm_out/{exp_name}.out'):
                    os.remove(f'cache/slurm_out/{exp_name}.out')
                os.system('sbatch '+f' {exp_name}.sh')
    already_serve = []
    for j in jobs:
        if j['name'] in serve_collections and j['status'].strip().lower()=='r':
            if not os.path.isfile(f'{j["name"]}.out'):
                os.system(f"scancel {j['id']}")
            else:
                if j['total_time']>=600:
                    already_serve.append({
                        'name': j['name'],
                        'total_time': j['total_time']
                    })
    if len(already_serve)!=3:
        time.sleep(30)
        continue
    all_times = [s['total_time'] for s in already_serve]
    if min(all_times)<600:
        time.sleep(600-min(all_times))
    exp_name = EXPERIMENT_NAME1
    with open(f'{exp_name}.out') as f:
        lines = f.readlines()
    serve_ip1 = lines[0].strip()
    exp_name = EXPERIMENT_NAME2
    with open(f'{exp_name}.out') as f:
        lines = f.readlines()
    serve_ip2 = lines[0].strip()
    exp_name = EXPERIMENT_NAME3
    with open(f'{exp_name}.out') as f:
        lines = f.readlines()
    serve_ip3 = lines[0].strip()
    change_flag = False
    if os.path.isfile('serve_train_tool_orchestra.json'):
        with open('serve_train_tool_orchestra.json') as f:
            old_config = json.load(f)
        if old_config['Qwen/Qwen3-32B'][0]['ip_addr']!=serve_ip2:
            change_flag = True
        if old_config['retrieval'][0]['ip_addr']!=serve_ip1:
            change_flag = True
    else:
        change_flag = True
    payload = {
        "queries": ["How to compute f(f(x)) when f is piecewise-defined"],
        "topk": 3,
        "return_scores": True,
        "eid": '84176'
    }
    serve_alive = True
    for testing_port in [1401]:
        try_count = 0
        testing_alive = False
        while not testing_alive and try_count<5:
            try_count += 1
            try:
                testing = requests.post(f'http://{serve_ip1}:{testing_port}/retrieve', json=payload).json()
                testing_alive = True
                break
            except Exception as serve_error:
                print(f'{testing_port} serve failure',serve_ip1,serve_error)
                time.sleep(20)
        if not testing_alive:
            serve_alive = False
            print_time()
            jobs = get_jobs()
            job_names = [j['name'] for j in jobs]
            for j in jobs:
                if j['name'].startswith(EXPERIMENT_NAME1):
                    print(f"scancel {j['id']}")
                    os.system(f"scancel {j['id']}")
            break
    if not serve_alive:
        continue
    print('serve success',serve_ip1)
    model_config = {
        "retrieval": [{"ip_addr": serve_ip1,"port": "1401"}],
        "meta-llama/Llama-3.1-8B-Instruct": [{"ip_addr": serve_ip1,"port": "1402"}],
        "microsoft/Phi-4-mini-instruct": [{"ip_addr": serve_ip1,"port": "1408"}],
        "Qwen/Qwen2.5-Math-72B-Instruct": [{"ip_addr": serve_ip1,"port": "1403"}],
        "Qwen/Qwen2.5-Math-7B-Instruct": [{"ip_addr": serve_ip1,"port": "1404"}],
        "meta-llama/Llama-3.3-70B-Instruct": [{"ip_addr": serve_ip2,"port": "1405"}],
        "Qwen/Qwen3-32B": [{"ip_addr": serve_ip2,"port": "1406"}],
        "Qwen/Qwen2.5-Coder-32B-Instruct": [{"ip_addr": serve_ip2,"port": "1407"}],
        "google/gemma-2-9b-it": [{"ip_addr": serve_ip3,"port": "1409"}],
        "codellama/CodeLlama-7b-Instruct-hf": [{"ip_addr": serve_ip3,"port": "1410"}],
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": [{"ip_addr": serve_ip3,"port": "1411"}],
        "vllm_model_config_path": "serve_train_tool_orchestra.json"
    }
    with open('serve_train_tool_orchestra.json','w') as f:
        json.dump(model_config,f,indent=2)
    if not 'run' in job_names:
        os.system('sbatch '+' resume_run_h100.sh')
    time.sleep(60)

    
    




