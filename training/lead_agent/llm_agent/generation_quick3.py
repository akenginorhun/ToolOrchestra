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
#
# -----------------------------------------------------------------------------

from math import remainder
import random
import torch
import re
import asyncio
from collections import defaultdict
import os
import copy
import json
import time
import subprocess
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from openai import OpenAI
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
import shutil
import requests
import sys
REPO_PATH = os.getenv("REPO_PATH")
sys.path.append(REPO_PATH)
from LLM_CALL import get_llm_response
import multiprocessing as mp
from transformers import AutoTokenizer

from datetime import datetime
import string
def generate_random_string(length):
    """Generates a random string of specified length using alphanumeric characters."""
    characters = string.ascii_letters + string.digits + '~!@#$%^&*()-=_+[]'
    return ''.join(random.choice(characters) for _ in range(length))

oss_client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("OSS_KEY")
)

def merge_documents(main_list,sub_list):
    if len(sub_list)==0:
        return main_list
    if len(main_list)<len(sub_list):
        return main_list+sub_list
    merged_list = []
    multiple = len(main_list)//len(sub_list)
    assert multiple>0
    idx_main = 0
    idx_sub = 0
    while idx_sub<len(sub_list):
        if not sub_list[idx_sub] in merged_list:
            merged_list.append(sub_list[idx_sub])
        assert idx_main+multiple<=len(main_list)
        for iter_idx in range(idx_main,idx_main+multiple):
            if not main_list[iter_idx] in merged_list:
                merged_list.append(main_list[iter_idx])
        idx_main = idx_main+multiple
        idx_sub += 1
    merged_list += main_list[multiple*len(sub_list):]
    return merged_list

ALL_TOOLS = {
    "enhance_reasoning": {
        'model': ["reasoner-1", "reasoner-2", "reasoner-3"]
    },
    "answer": {
        'model': ["answer-math-1", "answer-math-2", "answer-1", "answer-2", "answer-3", "answer-4"]
    },
    "search": {
        "model": ["search-1", "search-2", "search-3"]
    },
}

# tokenizer_qwen25 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
# tokenizer_qwen3 = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

def cut_seq(tokenizer,seq,l):
    if len(seq)==0:
        return {
            'effective_length': 0,
            'string_after_cut': ''
        }
    token_ids = tokenizer(seq)['input_ids']
    rs = tokenizer.batch_decode(token_ids[-l:], skip_special_tokens=True)
    return {
        'effective_length': len(token_ids),
        'string_after_cut': ''.join(rs)
    }

def cut_middle_turns(tokenizer,messages,max_length):
    exec_count = 0
    while exec_count<100:
        try:
            exec_count += 1
            messages_str = ''
            start_identifier = generate_random_string(15)
            end_identifier = generate_random_string(15)
            assert not start_identifier in str(messages) and not end_identifier in str(messages) and start_identifier!=end_identifier
            for mid,m in enumerate(messages):
                messages_str += f"{m}{start_identifier}{mid}{end_identifier}"
            token_ids = tokenizer(str(messages_str))['input_ids']
            if len(token_ids)<=max_length:
                return messages
            p1_tokens = tokenizer.batch_decode(token_ids[:max_length//2])
            p1 = ''.join(p1_tokens)
            p1_idx = int(p1.split(start_identifier)[-1].split(end_identifier)[0])
            p2_tokens = tokenizer.batch_decode(token_ids[-max_length//2:])
            p2 = ''.join(p2_tokens)
            p2_idx = int(p2.split(end_identifier)[0].split(start_identifier)[-1])
            return messages[:p1_idx+1]+messages[p2_idx:]
        except Exception as cut_error:
            pass

def seq_length(tokenizer,seq):
    input_data = tokenizer(seq, return_tensors='pt', add_special_tokens=True)
    return len(tokenizer.convert_ids_to_tokens(input_data['input_ids'][0]))

def call_tool(arguments):
    start = time.time()
    start_time = time.time()
    if arguments['category']=='qa':
        if arguments['tool']=='enhance_reasoning':
            cost = 0
            generated_code = ''
            model_name = arguments['cur_model_mapping'][arguments['model']]
            cur_tool_pricing = arguments['cur_tool_pricing']
            if model_name in ['o3','o3-mini','gpt-5','gpt-5-mini']:
                prompt = arguments['context_str'].strip()+'\n\n'
                prompt += f"Question: {arguments['problem']}\nInstead of directly answering the question, please write additional python code that will give intermidiate results after execution. Wrap the code within ```python and ```. The code should be self-contained with all the import and initialization."
                latency_testing_start_time = time.time()
                response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,temperature=1,max_length=28000)
                latency_testing_end_time = time.time()
                if isinstance(response,str) or not response.choices[0].message.content:
                    latency = time.time()-start
                    arguments['generated_code'] = ''
                    arguments['exec_result'] = ''
                    arguments['prompt_tokens'] = 0
                    arguments['completion_tokens'] = 0
                    arguments['latency'] = time.time() - start_time
                    arguments['cost'] = cost
                    return arguments
                if not 'tokens_pic' in arguments:
                    arguments['tokens_pic'] = []
                arguments['tokens_pic'].append({
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'model': model_name,
                    'latency': latency_testing_end_time-latency_testing_start_time
                })
                cost = cost + (cur_tool_pricing[model_name]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[model_name]['output_tokens_per_million'])
                try:
                    generated_code = response.choices[0].message.content.split('```python')[-1].split('```')[0]
                except:
                    latency = time.time()-start
                    arguments['generated_code'] = ''
                    arguments['exec_result'] = ''
                    arguments['prompt_tokens'] = 0
                    arguments['completion_tokens'] = 0
                    arguments['latency'] = time.time() - start_time
                    arguments['cost'] = cost
                    return arguments
            elif 'qwen2.5-coder' in model_name.lower() or 'llama' in model_name.lower():
                prompt = arguments['context_str'].strip()+'\n\n'
                prompt += f"Question: {arguments['problem']}\nInstead of directly answering the question, please write additional python code that will give intermidiate results after execution. Wrap the code within ```python and ```. The code should be self-contained with all the import and initialization."
                latency_testing_start_time = time.time()
                response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['id'])
                latency_testing_end_time = time.time()
                if isinstance(response,str):
                    response = ''
                    while not response:
                        try:
                            response = oss_client.chat.completions.create(
                                model="nvdev/qwen/qwen2.5-coder-32b-instruct", 
                                messages=[{"role":"user","content":prompt}],temperature=0.2,
                                top_p=0.7,
                                max_tokens=30000,
                            )
                        except Exception as qwen_error:
                            time.sleep(60)
                if isinstance(response,str) or not response.choices[0].message.content:
                    latency = time.time()-start
                    arguments['generated_code'] = ''
                    arguments['exec_result'] = ''
                    arguments['prompt_tokens'] = 0
                    arguments['completion_tokens'] = 0
                    arguments['latency'] = time.time() - start_time
                    arguments['cost'] = cost
                    return arguments
                if not 'tokens_pic' in arguments:
                    arguments['tokens_pic'] = []
                arguments['tokens_pic'].append({
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'model': model_name,
                    'latency': latency_testing_end_time-latency_testing_start_time
                })
                cost = cost + (cur_tool_pricing[model_name]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[model_name]['output_tokens_per_million'])
                try:
                    generated_code = response.choices[0].message.content.split('```python')[-1].split('```')[0]
                except:
                    latency = time.time()-start
                    arguments['generated_code'] = ''
                    arguments['exec_result'] = ''
                    arguments['prompt_tokens'] = 0
                    arguments['completion_tokens'] = 0
                    arguments['latency'] = time.time() - start_time
                    arguments['cost'] = cost
                    return arguments
            code_path = str(os.path.join(arguments['cur_output_dir'],f'exec_code_{arguments["id"]}.py'))
            with open(code_path,'w') as f:
                f.write(generated_code)
            exec_result = ''
            try:
                sandbox_start_time = time.time()
                exec_result = subprocess.run(['python', code_path], timeout=60, capture_output=True, text=True)
                exec_result = exec_result.stdout
                sandbox_latency = time.time() - sandbox_start_time
                cost = cost + cur_tool_pricing['code_interpreter_per_second'] * sandbox_latency
                with open(os.path.join(arguments['cur_output_dir'],f'exec_out_{arguments["id"]}.txt'),'w') as f:
                    f.write(exec_result)
            except Exception as e:
                pass
            latency = time.time()-start
            arguments['generated_code'] = generated_code
            arguments['exec_result'] = exec_result
            arguments['prompt_tokens'] = response.usage.prompt_tokens
            arguments['completion_tokens'] = response.usage.completion_tokens
            arguments['latency'] = time.time() - start_time
            arguments['cost'] = cost
            arguments['used_llm'] = model_name
            if 'tokenizer' in arguments:
                arguments.pop('tokenizer')
            return arguments
        
        elif arguments['tool']=='answer':
            prompt = arguments['context_str'].strip()+'\n\n'+arguments['problem']
            response_str = ''
            pred = ''
            cost = 0
            cur_answer_model = arguments['cur_model_mapping'][arguments['model']]
            cur_tool_pricing = arguments['cur_tool_pricing']
            
            start_time = time.time()
            if 'math' in cur_answer_model.lower():
                model_name = cur_answer_model
                messages = [{"content": prompt+"\nLet's think step by step and output the final answer within \\boxed{}.", "role": "user"}]
                latency_testing_start_time = time.time()
                response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=2000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['id'])
                latency_testing_end_time = time.time()
                if isinstance(response,str):
                    latency = time.time()-start
                    arguments['response'] = ''
                    arguments['pred'] = ''
                    arguments['correctness'] = False
                    arguments['prompt_tokens'] = 0
                    arguments['completion_tokens'] = 0
                    arguments['latency'] = time.time() - start_time
                    arguments['cost'] = cost
                    return arguments
                if not 'tokens_pic' in arguments:
                    arguments['tokens_pic'] = []
                arguments['tokens_pic'].append({
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'model': model_name,
                    'latency': latency_testing_end_time-latency_testing_start_time
                })
                response_str = response.choices[0].message.content
                cost = cost + (cur_tool_pricing[cur_answer_model]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[cur_answer_model]['output_tokens_per_million'])
                if not '\\boxed{' in response_str:
                    pred = ''
                else:
                    pred_components = response.choices[0].message.content.split('\\boxed{')[-1].split('}')[:-1]
                    pred = '}'.join(pred_components).strip()
            elif 'qwen' in cur_answer_model.lower() or 'phi' in cur_answer_model.lower():
                model_name = cur_answer_model
                messages = [{"content": prompt+"\nLet's think step by step and output the final answer within \\boxed{}.", "role": "user"}]
                latency_testing_start_time = time.time()
                response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['id'])
                latency_testing_end_time = time.time()
                if isinstance(response,str) or not response.choices[0].message.content:
                    latency = time.time()-start
                    arguments['response'] = ''
                    arguments['pred'] = ''
                    arguments['correctness'] = False
                    arguments['prompt_tokens'] = 0
                    arguments['completion_tokens'] = 0
                    arguments['latency'] = time.time() - start_time
                    arguments['cost'] = cost
                    return arguments
                if not 'tokens_pic' in arguments:
                    arguments['tokens_pic'] = []
                arguments['tokens_pic'].append({
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'model': model_name,
                    'latency': latency_testing_end_time-latency_testing_start_time
                })
                response_str = response.choices[0].message.content
                cost = cost + (cur_tool_pricing[cur_answer_model]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[cur_answer_model]['output_tokens_per_million'])
                if not '\\boxed{' in response_str:
                    pred = ''
                else:
                    pred_components = response.choices[0].message.content.split('\\boxed{')[-1].split('}')[:-1]
                    pred = '}'.join(pred_components).strip()
            elif 'meta-llama' in cur_answer_model.lower():
                model_name = cur_answer_model
                messages = [{"content": prompt+"\nWrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>.", "role": "user"}]
                latency_testing_start_time = time.time()
                response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=40000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['id'])
                latency_testing_end_time = time.time()
                if isinstance(response,str) or not response.choices[0].message.content:
                    response = ''
                    while not response:
                        try:
                            response = oss_client.chat.completions.create(
                                model="nvdev/meta/llama-3.3-70b-instruct", 
                                messages=messages,temperature=0.2,
                                top_p=0.7,
                                max_tokens=40000,
                            )
                        except Exception as llama_error:
                            time.sleep(60)
                    if isinstance(response,str) or not response.choices[0].message.content:
                        latency = time.time()-start_time
                        arguments['response'] = ''
                        arguments['pred'] = ''
                        arguments['correctness'] = False
                        arguments['prompt_tokens'] = 0
                        arguments['completion_tokens'] = 0
                        arguments['latency'] = latency
                        arguments['cost'] = cost
                        return arguments
                if not 'tokens_pic' in arguments:
                    arguments['tokens_pic'] = []
                arguments['tokens_pic'].append({
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'model': model_name,
                    'latency': latency_testing_end_time-latency_testing_start_time
                })
                response_str = response.choices[0].message.content
                cost = cost + (cur_tool_pricing[cur_answer_model]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[cur_answer_model]['output_tokens_per_million'])
                pred = response.choices[0].message.content.split('<answer>')[-1].split('</answer>')[0].strip()
            elif cur_answer_model in ['o3','o3-mini','gpt-5','gpt-5-mini']:
                model_name = cur_answer_model
                prompt += "\nWrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>."
                latency_testing_start_time = time.time()
                response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,max_length=28000)
                latency_testing_end_time = time.time()
                if isinstance(response,str) or not response.choices[0].message.content:
                    latency = time.time()-start
                    arguments['response'] = ''
                    arguments['pred'] = ''
                    arguments['correctness'] = False
                    arguments['prompt_tokens'] = 0
                    arguments['completion_tokens'] = 0
                    arguments['latency'] = time.time() - start_time
                    arguments['cost'] = cost
                    return arguments
                if not 'tokens_pic' in arguments:
                    arguments['tokens_pic'] = []
                arguments['tokens_pic'].append({
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'model': model_name,
                    'latency': latency_testing_end_time-latency_testing_start_time
                })
                response_str = response.choices[0].message.content
                cost = cost + (cur_tool_pricing[cur_answer_model]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[cur_answer_model]['output_tokens_per_million'])
                pred = response.choices[0].message.content.split('<answer>')[-1].split('</answer>')[0].strip()
            elif 'nvidia' in cur_answer_model.lower():
                model_name = cur_answer_model
                prompt += "\nWrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>."
                latency_testing_start_time = time.time()
                response = ''
                while not response:
                    try:
                        response = oss_client.chat.completions.create(
                                        model=model_name,
                                        messages=[{'role': 'user','content': prompt}],
                                        temperature=0.2,
                                        top_p=0.7,
                                        max_tokens=30000,
                                    )
                    except Exception as error:
                        time.sleep(60)
                latency_testing_end_time = time.time()
                if isinstance(response,str) or not response.choices[0].message.content:
                    latency = time.time()-start
                    arguments['response'] = ''
                    arguments['pred'] = ''
                    arguments['correctness'] = False
                    arguments['prompt_tokens'] = 0
                    arguments['completion_tokens'] = 0
                    arguments['latency'] = time.time() - start_time
                    arguments['cost'] = cost
                    return arguments
                if not 'tokens_pic' in arguments:
                    arguments['tokens_pic'] = []
                arguments['tokens_pic'].append({
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'model': model_name,
                    'latency': latency_testing_end_time-latency_testing_start_time
                })
                response_str = response.choices[0].message.content
                cost = cost + (cur_tool_pricing[cur_answer_model]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[cur_answer_model]['output_tokens_per_million'])
                pred = response.choices[0].message.content.split('<answer>')[-1].split('</answer>')[0].strip()
            latency = time.time()-start_time
            
            if pred.strip()=='' or len(pred.split(' '))>500:
                correctness = False
            elif pred.strip().lower()==arguments['answer'].strip().lower():
                correctness = True
            else:
                eval_prompt = (f"Question: {arguments['problem']}\n\n"
                        f"Student answer: {pred}\n\n"
                        f"Reference answer: {arguments['answer']}\n\n"
                        "Assume that the reference answer is correct. Output <correct>True</correct> if the student answer matches the reference answer. Output <correct>False</correct> if the student answer does not match the reference answer.")
                eval_result = get_llm_response(model='gpt-5',messages=eval_prompt,temperature=1)
                eval_result = eval_result.split('<correct>')[-1].split('</correct>')[0]
                if eval_result.lower()=='true':
                    correctness = True
                else:
                    correctness = False
            arguments['response'] = response_str
            arguments['pred'] = pred
            arguments['correctness'] = correctness
            arguments['prompt_tokens'] = response.usage.prompt_tokens
            arguments['completion_tokens'] = response.usage.completion_tokens
            arguments['latency'] = latency
            arguments['cost'] = cost
            arguments['used_llm'] = cur_answer_model
            if 'tokenizer' in arguments:
                arguments.pop('tokenizer')
            return arguments

        elif arguments['tool'] in ['search']:

            contents = []
            cost = 0
            latency = 0
            cur_tool_pricing = arguments['cur_tool_pricing']
            if not arguments['model'] in arguments['cur_model_mapping']:
                pass
            else:
                prompt = arguments['context_str'].strip()+'\n\n'
                prompt += f"Question: {arguments['problem']}\nInstead of directly answering the question, please write a query to search for a piece of relevant and missing information. The query should be a few key words about the information to search or a short sentence. Wrap the query within <query> and </query>."
                cur_query_writer = arguments['cur_model_mapping'][arguments['model']]
                arguments['used_llm'] = cur_query_writer
                query_to_call = None
                start_time = time.time()
                if cur_query_writer in ['o3','o3-mini','gpt-5','gpt-5-mini']:
                    latency_testing_start_time = time.time()
                    response = get_llm_response(model=cur_query_writer,messages=prompt,return_raw_response=True,temperature=1,max_length=28000)
                    latency_testing_end_time = time.time()
                    if isinstance(response,str) or not response.choices[0].message.content:
                        query_to_call = arguments['problem']
                    else:
                        if not 'tokens_pic' in arguments:
                            arguments['tokens_pic'] = []
                        arguments['tokens_pic'].append({
                            'input_tokens': response.usage.prompt_tokens,
                            'output_tokens': response.usage.completion_tokens,
                            'model': cur_query_writer,
                            'latency': latency_testing_end_time-latency_testing_start_time
                        })
                        query_to_call = response.choices[0].message.content.split('<query>')[-1].split('</query>')[0]
                        if len(query_to_call)<5:
                            query_to_call = arguments['problem']
                        cost = cost + (cur_tool_pricing[cur_query_writer]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[cur_query_writer]['output_tokens_per_million'])
                elif 'nvidia' in cur_query_writer.lower():
                    latency_testing_start_time = time.time()
                    response = ''
                    while not response:
                        try:
                            response = oss_client.chat.completions.create(
                                            model=cur_query_writer,
                                            messages=[{'role': 'user','content': prompt}],
                                            temperature=0.2,
                                            top_p=0.7,
                                            max_tokens=30000,
                                        )
                        except Exception as error:
                            time.sleep(60)
                    latency_testing_end_time = time.time()
                    if isinstance(response,str) or not response.choices[0].message.content:
                        query_to_call = arguments['problem']
                    else:
                        if not 'tokens_pic' in arguments:
                            arguments['tokens_pic'] = []
                        arguments['tokens_pic'].append({
                            'input_tokens': response.usage.prompt_tokens,
                            'output_tokens': response.usage.completion_tokens,
                            'model': cur_query_writer,
                            'latency': latency_testing_end_time-latency_testing_start_time
                        })
                        query_to_call = response.choices[0].message.content.split('<query>')[-1].split('</query>')[0]
                        if len(query_to_call)<5:
                            query_to_call = arguments['problem']
                        cost = cost + (cur_tool_pricing[cur_query_writer]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[cur_query_writer]['output_tokens_per_million'])
                elif 'qwen' in cur_query_writer.lower() or 'llama' in cur_query_writer.lower() or 'phi' in cur_query_writer.lower():
                    latency_testing_start_time = time.time()
                    response = get_llm_response(model=cur_query_writer,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][cur_query_writer],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=0)
                    latency_testing_end_time = time.time()
                    if isinstance(response,str) or not response.choices[0].message.content:
                        query_to_call = arguments['problem']
                    else:
                        if not 'tokens_pic' in arguments:
                            arguments['tokens_pic'] = []
                        arguments['tokens_pic'].append({
                            'input_tokens': response.usage.prompt_tokens,
                            'output_tokens': response.usage.completion_tokens,
                            'model': cur_query_writer,
                            'latency': latency_testing_end_time-latency_testing_start_time
                        })
                        query_to_call = response.choices[0].message.content.split('<query>')[-1].split('</query>')[0]
                        if len(query_to_call)<5:
                            query_to_call = arguments['problem']
                        cost = cost + (cur_tool_pricing[cur_query_writer]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[cur_query_writer]['output_tokens_per_million'])
                elif 'google' in cur_query_writer.lower():
                    latency_testing_start_time = time.time()
                    response = get_llm_response(model=cur_query_writer,messages=prompt,return_raw_response=True,model_type='vllm',max_length=2000,temperature=0.2,model_config=arguments['vllm_model_configs'][cur_query_writer],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=0)
                    latency_testing_end_time = time.time()
                    if isinstance(response,str) or not response.choices[0].message.content:
                        query_to_call = arguments['problem']
                    else:
                        if not 'tokens_pic' in arguments:
                            arguments['tokens_pic'] = []
                        arguments['tokens_pic'].append({
                            'input_tokens': response.usage.prompt_tokens,
                            'output_tokens': response.usage.completion_tokens,
                            'model': cur_query_writer,
                            'latency': latency_testing_end_time-latency_testing_start_time
                        })
                        query_to_call = response.choices[0].message.content.split('<query>')[-1].split('</query>')[0]
                        if len(query_to_call)<5:
                            query_to_call = arguments['problem']
                        cost = cost + (cur_tool_pricing[cur_query_writer]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[cur_query_writer]['output_tokens_per_million'])
                if query_to_call is None:
                    pass
                else:
                    if arguments['cur_index'].startswith('wiki'):
                        query_length = len(arguments['tokenizer'](query_to_call)['input_ids'])
                        cost = cost + query_length * cur_tool_pricing['Qwen/Qwen3-8B']['input_tokens_per_million']
                    else:
                        cost = cost + cur_tool_pricing['tavily']['search'] + cur_tool_pricing['tavily']['extract']*arguments['topk_doc']
                    assert len(query_to_call)>3,f"{query_to_call}"
                    payload = {
                        "queries": [query_to_call[:390]],
                        "topk": arguments['topk_doc'],
                        "return_scores": True,
                        "eid": arguments['cur_index'].split('____')[-1]
                    }
                    results = None
                    with open(arguments['vllm_model_configs']['vllm_model_config_path']) as f:
                        all_vllm_model_configs = json.load(f)
                    cur_model_config = None
                    search_try_count = 0
                    while not results:
                        search_try_count += 1
                        try:
                            if arguments['cur_index'].startswith('wiki'):
                                cur_model_config = random.choice(all_vllm_model_configs['wiki_retrieval'])
                            else:
                                cur_model_config = random.choice(all_vllm_model_configs['retrieval'])
                            results = requests.post(f'http://{cur_model_config["ip_addr"]}:{cur_model_config["port"]}/retrieve', json=payload).json()
                        except Exception as search_error:
                            time.sleep(60)
                    if results:
                        for r in results[0]:
                            if 'content' in r['document']:
                                contents.append(r['document']['content'])
                            elif 'contents' in r['document']:
                                contents.append(r['document']['contents'])
                latency = latency + (time.time() - start_time)
            arguments['search_results_data'] = contents
            arguments['cost'] = cost
            arguments['latency'] = time.time() - start_time
            if 'tokenizer' in arguments:
                arguments.pop('tokenizer')
            return arguments
    elif arguments['category']=='func_call':
        assert isinstance(arguments['tool'],list)
        mode_to_call = None
        message_to_user = None
        cost = 0
        cur_tool_pricing = arguments['cur_tool_pricing']
        valid_tools = []
        for iter_one_tool_call in arguments['tool']:
            if isinstance(iter_one_tool_call,dict) and 'name' in iter_one_tool_call and 'arguments' in iter_one_tool_call and iter_one_tool_call['name']=='call_expert' and 'expert' in iter_one_tool_call['arguments'] and iter_one_tool_call['arguments']['expert'] in ['expert-1', 'expert-2', 'expert-3']:
                mode_to_call = arguments['cur_model_mapping'][iter_one_tool_call['arguments']['expert']]
                arguments['model'] = iter_one_tool_call['arguments']['expert']
            elif isinstance(iter_one_tool_call,dict):
                valid_tools.append(iter_one_tool_call)
            elif isinstance(iter_one_tool_call,str):
                message_to_user = iter_one_tool_call
        if mode_to_call:
            arguments['used_llm'] = mode_to_call
        else:
            arguments['used_llm'] = 'func_call_no_llm'
        if mode_to_call:
            updated_messages = arguments['input_messages']
            tool_calls = []
            response_content = "Wait a minute, I will take it very soon"
            if mode_to_call in ['o3','o3-mini','gpt-5','gpt-5-mini']:
                latency_testing_start_time = time.time()
                response = get_llm_response(model=mode_to_call,messages=updated_messages,tools=arguments['input_tools'], return_raw_response=True,max_length=40000)
                latency_testing_end_time = time.time()
                if not isinstance(response,str):
                    if not 'tokens_pic' in arguments:
                        arguments['tokens_pic'] = []
                    arguments['tokens_pic'].append({
                        'input_tokens': response.usage.prompt_tokens,
                        'output_tokens': response.usage.completion_tokens,
                        'model': mode_to_call,
                        'latency': latency_testing_end_time-latency_testing_start_time
                    })
                    cost = cost + (cur_tool_pricing[mode_to_call]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[mode_to_call]['output_tokens_per_million'])
                    response_content = response.choices[0].message.content
                    if response.choices[0].message.tool_calls:
                        for one_tool_call in response.choices[0].message.tool_calls:
                            tool_calls.append({
                                'name': one_tool_call.function.name,
                                'arguments': json.loads(one_tool_call.function.arguments)
                            })
                else:
                    response_content = "Wait a minute, I will take it very soon"
            elif 'nemotron-ultra' in mode_to_call.lower() or 'nemotron-super' in mode_to_call.lower():
                latency_testing_start_time = time.time()
                response = response = oss_client.chat.completions.create(
                    model=mode_to_call,
                    messages=updated_messages,
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=30000,
                    tools=arguments['input_tools']
                )
                latency_testing_end_time = time.time()
                if not isinstance(response,str):
                    if not 'tokens_pic' in arguments:
                        arguments['tokens_pic'] = []
                    arguments['tokens_pic'].append({
                        'input_tokens': response.usage.prompt_tokens,
                        'output_tokens': response.usage.completion_tokens,
                        'model': mode_to_call,
                        'latency': latency_testing_end_time-latency_testing_start_time
                    })
                    cost = cost + (cur_tool_pricing[mode_to_call]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[mode_to_call]['output_tokens_per_million'])
                    response_content = response.choices[0].message.content
                    if response.choices[0].message.tool_calls:
                        for one_tool_call in response.choices[0].message.tool_calls:
                            tool_calls.append({
                                'name': one_tool_call.function.name,
                                'arguments': json.loads(one_tool_call.function.arguments)
                            })
                else:
                    response_content = "Wait a minute, I will take it very soon"
            elif 'qwen3' in mode_to_call.lower() or 'llama' in mode_to_call.lower():
                latency_testing_start_time = time.time()
                response = get_llm_response(model=mode_to_call,messages=updated_messages,tools=arguments['input_tools'],return_raw_response=True,model_type='vllm',max_length=2000,temperature=0.2,model_config=arguments['vllm_model_configs'][mode_to_call],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['id'])
                latency_testing_end_time = time.time()
                if not isinstance(response,str):
                    if not 'tokens_pic' in arguments:
                        arguments['tokens_pic'] = []
                    arguments['tokens_pic'].append({
                        'input_tokens': response.usage.prompt_tokens,
                        'output_tokens': response.usage.completion_tokens,
                        'model': mode_to_call,
                        'latency': latency_testing_end_time-latency_testing_start_time
                    })
                    cost = cost + (cur_tool_pricing[mode_to_call]['input_tokens_per_million'] * response.usage.prompt_tokens + response.usage.completion_tokens * cur_tool_pricing[mode_to_call]['output_tokens_per_million'])
                    response_content = response.choices[0].message.content
                    if response.choices[0].message.tool_calls:
                        for one_tool_call in response.choices[0].message.tool_calls:
                            tool_calls.append({
                                'name': one_tool_call.function.name,
                                'arguments': json.loads(one_tool_call.function.arguments)
                            })
                else:
                    response_content = "Wait a minute, I will take it very soon"
            response = {
                'content': response_content,
                'tool_calls': tool_calls,
                'updated_messages': updated_messages,
                'mode': mode_to_call,
            }
        elif len(valid_tools)>0:
            response = {
                'content': None,
                'tool_calls': valid_tools,
                'mode': 'original_tool_call'
            }
        else:
            if not message_to_user:
                message_to_user = "Wait a minute, I will take it very soon"
            response = {
                'content': message_to_user,
                'tool_calls': [],
                'mode': 'message_to_user'
            }
        with open(arguments['transfer_path'],'w') as f:
            json.dump(response,f,indent=2)
        arguments['response'] = response
        if 'tokenizer' in arguments:
            arguments.pop('tokenizer')
        arguments['cost'] = cost
        arguments['latency'] = time.time() - start_time
        return arguments


def call_tool_all(all_arguments):
    all_return_arguments = []
    for one_arguments in all_arguments['all_call_arguments']:
        try:
            all_return_arguments.append(call_tool(one_arguments))
        except Exception as tool_call_error:
            pass
    return {
        'id': all_arguments['id'],
        'all_tool_call_results': all_return_arguments
    }

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Tuple, Any, Callable

# task_list is an iterable of (func, arg) pairs
async def run_all(
    task_list: Iterable[Tuple[Callable[[Any], Any], Any]],
    progress: bool = False,
    return_exceptions: bool = False,
):
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        # wrap each task so it obeys the semaphore
        async def run_one(idx: int, func: Callable, arg: Any):
            async with sem:
                try:
                    if asyncio.iscoroutinefunction(func):
                        res = await func(arg)
                    else:
                        res = await loop.run_in_executor(executor, func, arg)
                    return idx, res, None
                except Exception as e:
                    return idx, None, e

        task_list = list(task_list)
        tasks = [asyncio.create_task(run_one(i, f, a))
                 for i, (f, a) in enumerate(task_list)]

        results = [None] * len(tasks)

        if progress:
            from tqdm import tqdm
            pbar = tqdm(total=len(tasks))
        else:
            pbar = None

        try:
            # update progress as tasks complete
            for fut in asyncio.as_completed(tasks):
                idx, res, err = await fut
                if err is None:
                    results[idx] = res
                else:
                    if return_exceptions:
                        results[idx] = err
                    else:
                        for t in tasks:
                            t.cancel()
                        with contextlib.suppress(Exception):
                            await asyncio.gather(*tasks, return_exceptions=True)
                        raise err
                if pbar:
                    pbar.update(1)
        finally:
            if pbar:
                pbar.close()

        return results

@dataclass
class GenerationConfig:
    max_turns: int
    max_prompt_length: int 
    max_response_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        train_tool_config_path=None,
        test_tool_config_path=None,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        # [t.shape for t in tensors_with_mask] [torch.Size([2560, 0]), torch.Size([2560, 657]), torch.Size([2560, 500])]
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        # concatenated_with_info.shape torch.Size([2560, 1157])
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        # concatenated.shape torch.Size([2560, 1004]) sorted_indices.shape torch.Size([2560, 1004])
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch,tokenizer_config,global_steps,topk_doc,use_llm_reward,efficiency_reward,exp_tag,use_qa_reward):

        loop_batch_size = len(gen_batch.non_tensor_batch['problem'])
        active_mask = torch.ones(loop_batch_size, dtype=torch.bool)
        tokenizer = tokenizer_config['tokenizer']
        retrieved_documents = [[] for _ in range(loop_batch_size)]
        code_snippets = [[] for _ in range(loop_batch_size)]
        attempts = [[] for _ in range(loop_batch_size)]
        total_costs = [0 for _ in range(loop_batch_size)]
        rollings = gen_batch
        all_turn_input_ids = []
        all_turn_attention_mask = []
        all_turn_position_ids = []
        all_turn_responses = []
        all_turn_index = []
        all_turn_ids = []
        all_turn_turn_ids = []
        all_turn_repeat_ids = []
        all_turn_answers = []
        all_turn_answer_preds = []
        all_turn_problems = []
        all_turn_tools = []
        all_turn_success = []
        all_turn_costs = []
        all_turn_latency = []
        all_turn_formats = []
        all_turn_valid_answers_generated = []
        all_turn_used_llms = []
        all_turn_steps = []
        all_turn_categories = []
        all_pref_vecs = {}
        meta_info = {}
        vllm_model_configs = {}
        example_correct_by_rollout_id = {}
        
        tool_total = 0
        TRAINED_MODEL_TYPE = None
        cur_batch_indices = copy.deepcopy(gen_batch.non_tensor_batch['index'])
        cur_batch_repeat_ids = copy.deepcopy(gen_batch.non_tensor_batch['repeat_id'])
        cur_batch_output_files = []
        tmp_my_output_dir = gen_batch.non_tensor_batch['my_output_dir'][0]
        my_output_dir = gen_batch.non_tensor_batch['my_output_dir'][0]
        if os.path.isdir(os.path.join(tmp_my_output_dir,f"global_step_{global_steps}")):
            shutil.rmtree(os.path.join(tmp_my_output_dir,f"global_step_{global_steps}"))
        tmp_transfer_dir = os.path.join(gen_batch.non_tensor_batch['cur_transfer_dir'][0],str(global_steps))
        if os.path.isdir(tmp_transfer_dir):
            shutil.rmtree(tmp_transfer_dir)
        for step in range(self.config.max_turns):
            if not os.path.isdir(os.path.join(my_output_dir,f"global_step_{global_steps}",f"rollout_step_{step}")):
                os.makedirs(os.path.join(my_output_dir,f"global_step_{global_steps}",f"rollout_step_{step}"))
            if not active_mask.sum():
                break
            if step==0:
                for item_idx in range(loop_batch_size):
                    category = gen_batch.non_tensor_batch['category'][item_idx]
                    item_index = gen_batch.non_tensor_batch['index'][item_idx]
                    item_repeat_id = gen_batch.non_tensor_batch['repeat_id'][item_idx]
                    iter_rollout_id = f"{item_index}_____{item_repeat_id}"
                    all_pref_vecs[iter_rollout_id] = gen_batch.non_tensor_batch['pref_vec'][item_idx]
                    if category!='func_call':
                        continue
                    cur_transfer_dir = os.path.join(gen_batch.non_tensor_batch['cur_transfer_dir'][item_idx],str(global_steps),iter_rollout_id)
                    task_path = os.path.join(cur_transfer_dir,'task.json')
                    cur_func_call_output_path = os.path.join(cur_transfer_dir,'output.json')
                    if not os.path.isdir(cur_transfer_dir):
                        os.makedirs(cur_transfer_dir,exist_ok=True)
                    with open(task_path,'w') as f:
                        json.dump([gen_batch.non_tensor_batch['example'][item_idx]],f,indent=2)
                    cur_tool = gen_batch.non_tensor_batch['tools'][item_idx]
                    cur_domain = item_index.split('____')[0]
                    func_call_cmd = ['python','rollout/tau2/cli.py','--domain',cur_domain,'--agent-llm','train',
                        '--user-llm','gpt-5','--num-trials','1','--task_path',str(task_path),'--max-steps','40','--cur_transfer_dir',
                        str(cur_transfer_dir),'--output_file',str(cur_func_call_output_path),'--use_model_tool']
                    subprocess.Popen(func_call_cmd)
            all_input_ids = []
            all_attention_mask = []
            all_categories = []
            all_indices = []
            all_transfer_dirs = []
            all_input_messages = []
            all_input_tools = []
            all_transfer_paths = []
            all_output_file_path = []
            all_model_mappings = []
            all_tool_pricing = []
            assert len(retrieved_documents)==len(code_snippets)
            assert len(retrieved_documents)==len(attempts)
            item_idx = -1
            cached_inputs = []
            for doc_list,code_list,attempt_list in zip(retrieved_documents,code_snippets,attempts):
                item_idx += 1
                category = gen_batch.non_tensor_batch['category'][item_idx]
                all_categories.append(category)
                item_index = gen_batch.non_tensor_batch['index'][item_idx]
                item_repeat_id = gen_batch.non_tensor_batch['repeat_id'][item_idx]
                cur_model_mapping = gen_batch.non_tensor_batch['model_mapping'][item_idx]
                cur_tool_pricing = gen_batch.non_tensor_batch['tool_pricing'][item_idx]
                TRAINED_MODEL_TYPE = gen_batch.non_tensor_batch['model_type'][item_idx]
                all_indices.append(item_index)
                all_model_mappings.append(cur_model_mapping)
                all_tool_pricing.append(cur_tool_pricing)
                vllm_model_configs = gen_batch.non_tensor_batch['vllm_model_configs'][item_idx]
                if category=='qa':
                    problem = gen_batch.non_tensor_batch['problem'][item_idx]
                    tools = []
                    raw_tools = gen_batch.non_tensor_batch['tools'][item_idx]
                    my_output_dir = gen_batch.non_tensor_batch['my_output_dir'][item_idx]
                    for t in raw_tools:
                        tools.append(t)
                    doc_str = ''
                    for doc_idx, doc in enumerate(doc_list):
                        doc_str += f"Doc {doc_idx+1}: {doc[:4000]}\n\n"
                    code_str = ''
                    for code_idx, code_piece in enumerate(code_list):
                        code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                    attempt_str = ''
                    for attempt_idx, attempt in enumerate(attempt_list):
                        attempt_str += f"Attempt{attempt_idx+1} answer by {attempt['model']}: {attempt['answer']}\n"
                    str_cut = cut_seq(tokenizer=tokenizer,seq=attempt_str,l=8000)
                    attempt_str = str_cut['string_after_cut']
                    if not attempt_str.startswith('Attempt') and len(attempt_str)>0:
                        attempt_str = 'Attempt answer: '+attempt_str
                    str_cut = cut_seq(tokenizer=tokenizer,seq=code_str+attempt_str,l=16000)
                    code_attempt_str = str_cut['string_after_cut']
                    code_attempt_str_len = str_cut['effective_length']
                    if not code_attempt_str.startswith('```') and len(code_attempt_str)>0:
                        code_attempt_str = '```\n'+code_attempt_str
                    doc_flag = False
                    if code_attempt_str_len<24000:
                        context_str = cut_seq(tokenizer=tokenizer,seq=doc_str+code_attempt_str,l=24000)
                        context_str = context_str['string_after_cut']
                        if len(doc_str)>0:
                            doc_flag = True
                            context_str = 'Documents:\n'+context_str
                    else:
                        context_str = code_attempt_str
                    chat = [
                                {"role": "system", "content": "You are good at using tools. "},
                                {"role": "user", "content": f"Problem: {problem}\n\n{context_str}\n\nChoose an approriate tool."}
                            ]
                    all_transfer_dirs.append('')
                    all_input_messages.append(chat)
                    all_input_tools.append(tools)
                    all_transfer_paths.append(None)
                    all_output_file_path.append('')
                    if step==0:
                        cur_batch_output_files.append('')
                elif category=='func_call':
                    iter_rollout_id = f"{item_index}_____{item_repeat_id}"
                    cur_transfer_dir = os.path.join(gen_batch.non_tensor_batch['cur_transfer_dir'][item_idx],str(global_steps),iter_rollout_id)
                    all_transfer_dirs.append(cur_transfer_dir)
                    task_path = os.path.join(cur_transfer_dir,'task.json')
                    cur_func_call_output_path = os.path.join(cur_transfer_dir,'output.json')
                    transfer_idx = 0
                    while os.path.isfile(os.path.join(cur_transfer_dir,f"output_{transfer_idx}.json")) and os.path.isfile(os.path.join(cur_transfer_dir,f"input_{transfer_idx}.json")):
                        transfer_idx += 1
                    receive_end_signal = False
                    debug_file_path = os.path.join(cur_transfer_dir,f"wait_input_{transfer_idx}")
                    while not os.path.isfile(os.path.join(cur_transfer_dir,f"input_{transfer_idx}.json")):
                        if os.path.isfile(os.path.join(cur_transfer_dir,'done')):
                            try:
                                with open(os.path.join(cur_transfer_dir,'done')) as f:
                                    tmp_result = f.read()
                                if tmp_result=="Done!":
                                    correct = 0
                                    for subfile in os.listdir(os.path.join(cur_transfer_dir,'output')):
                                        if subfile.endswith('.json'):
                                            with open(os.path.join(cur_transfer_dir,'output',subfile)) as f:
                                                r = json.load(f)
                                            correct += r["reward_info"]["reward"]
                                    if correct>0:
                                        simulation_reward = 1
                                    else:
                                        simulation_reward = 0
                                    example_correct_by_rollout_id[iter_rollout_id] = simulation_reward
                            except:
                                pass
                            active_mask[item_idx] = 0
                            receive_end_signal = True
                            break
                        time.sleep(5)
                    try:
                        with open(os.path.join(cur_transfer_dir,f"input_{transfer_idx}.json")) as f:
                            input_dict = json.load(f)
                    except:
                        receive_end_signal = True
                    if receive_end_signal:
                        with open('tools_debug.json') as f:
                            debug_tool = json.load(f)
                        original_tools = debug_tool
                        input_dict = {
                            'tools': debug_tool,
                            'original_tools': debug_tool,
                            'messages': [{'role': 'user','content': 'fake input'}],
                            'original_messages': [{'role': 'user','content': 'fake input'}]
                        }
                    else:
                        with open(os.path.join(cur_transfer_dir,f"input_{transfer_idx}.json")) as f:
                            input_dict = json.load(f)
                        original_tools = input_dict['original_tools']
                    tools = input_dict['tools']
                    tools_length = len(tokenizer(str(tools))['input_ids'])
                    chat = cut_middle_turns(tokenizer=tokenizer,messages=input_dict['messages'],max_length=23000-tools_length)
                    all_input_messages.append(input_dict['original_messages'])
                    all_input_tools.append(original_tools)
                    all_transfer_paths.append(os.path.join(cur_transfer_dir,f"output_{transfer_idx}.json"))
                    all_output_file_path.append(cur_func_call_output_path)
                    if step==0:
                        cur_batch_output_files.append(cur_transfer_dir)
                    
                prompt_with_chat_template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tools=tools,tokenize=False)
                cached_inputs.append(prompt_with_chat_template)
                input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                                tokenizer=tokenizer,
                                                                                max_length=tokenizer_config['max_prompt_length'],
                                                                                pad_token_id=tokenizer.pad_token_id,
                                                                                left_pad=True,
                                                                                truncation='middle')
                # assert attention_mask[0][0]==0,f"input_ids: {input_ids}\nattention_mask: {attention_mask}"
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)

            if not active_mask.sum():
                break
            # exit(0)
            if not os.path.isdir(os.path.join(my_output_dir,f"global_step_{global_steps}",f"rollout_step_{step}",'code_execution')):
                os.makedirs(os.path.join(my_output_dir,f"global_step_{global_steps}",f"rollout_step_{step}",'code_execution'))
            assert my_output_dir is not None
            input_ids = torch.cat(all_input_ids, dim=0)
            attention_mask = torch.cat(all_attention_mask, dim=0)
            input_tokens_count = torch.sum(attention_mask,dim=1)
            position_ids = compute_position_id_with_mask(attention_mask)
            rollings.batch['input_ids'] = input_ids
            rollings.batch['attention_mask'] = attention_mask
            rollings.batch['position_ids'] = position_ids
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            input_padding = torch.ones((len(rollings_active.batch['input_ids']), tokenizer_config['max_response_length']+tokenizer_config['max_prompt_length']-len(rollings_active.batch['input_ids'][0])-len(responses_ids[0])), dtype=torch.long)*tokenizer.pad_token_id
            mask_padding = torch.zeros((len(rollings_active.batch['input_ids']), tokenizer_config['max_response_length']+tokenizer_config['max_prompt_length']-len(rollings_active.batch['input_ids'][0])-len(responses_ids[0])), dtype=torch.long)
            round_input_ids = torch.cat([input_padding,rollings_active.batch['input_ids'],responses_ids], dim=1)
            round_attention_mask = torch.cat([mask_padding,self.tensor_fn.create_attention_mask(rollings_active.batch['input_ids']),self.tensor_fn.create_attention_mask(responses_ids)], dim=1)
            round_position_ids = self.tensor_fn.create_position_ids(round_attention_mask)
            all_turn_input_ids.append(round_input_ids)
            all_turn_attention_mask.append(round_attention_mask)
            all_turn_position_ids.append(round_position_ids)
            all_turn_responses.append(responses_ids)
            cur_turn_index = [gen_batch.non_tensor_batch['index'][local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['index'])) if active_mask[local_idx]]
            assert len(cur_turn_index)==round_input_ids.shape[0],f"len(cur_turn_index): {len(cur_turn_index)}, round_input_ids.shape[0]: {round_input_ids.shape[0]}"
            all_turn_index += cur_turn_index
            cur_turn_turn_ids = [gen_batch.non_tensor_batch['turn_id'][local_idx]+step for local_idx in range(len(gen_batch.non_tensor_batch['turn_id'])) if active_mask[local_idx]]
            assert len(cur_turn_turn_ids)==round_input_ids.shape[0]
            all_turn_turn_ids += cur_turn_turn_ids
            cur_turn_repeat_ids = [gen_batch.non_tensor_batch['repeat_id'][local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['repeat_id'])) if active_mask[local_idx]]
            assert len(cur_turn_repeat_ids) == round_input_ids.shape[0]
            all_turn_repeat_ids += cur_turn_repeat_ids
            cur_turn_categories = [gen_batch.non_tensor_batch['category'][local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['category'])) if active_mask[local_idx]]
            assert len(cur_turn_categories) == round_input_ids.shape[0]
            all_turn_categories += cur_turn_categories
            cur_turn_ids = [gen_batch.non_tensor_batch['id'][local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['id'])) if active_mask[local_idx]]
            assert len(cur_turn_ids) == round_input_ids.shape[0]
            all_turn_ids += cur_turn_ids
            cur_turn_problems = [gen_batch.non_tensor_batch['problem'][local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['problem'])) if active_mask[local_idx]]
            assert len(cur_turn_problems) == round_input_ids.shape[0]
            all_turn_problems += cur_turn_problems
            cur_turn_steps = [step for local_idx in range(len(gen_batch.non_tensor_batch['problem'])) if active_mask[local_idx]]
            assert len(cur_turn_steps) == round_input_ids.shape[0]
            all_turn_steps += cur_turn_steps
            cur_answers = [gen_batch.non_tensor_batch['answer'][local_idx] for local_idx in
                            range(len(gen_batch.non_tensor_batch['answer'])) if active_mask[local_idx]]
            assert len(cur_answers) == round_input_ids.shape[0]
            all_turn_answers += cur_answers
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            response_id_mask = (responses_ids!=tokenizer.pad_token_id)
            response_tokens_count = torch.sum(response_id_mask,dim=1)
            main_agent_cost = []
            assert len(all_tool_pricing)==len(input_tokens_count)
            assert len(all_tool_pricing)==len(response_tokens_count)
            for one_tool_pricing,ic,rc in zip(all_tool_pricing,input_tokens_count.tolist(),response_tokens_count.tolist()):
                main_agent_cost.append(one_tool_pricing[TRAINED_MODEL_TYPE]['input_tokens_per_million']*ic+one_tool_pricing[TRAINED_MODEL_TYPE]['output_tokens_per_million']*rc)
            assert len(total_costs)==len(active_mask)
            assert len(total_costs)==len(main_agent_cost)
            for iter_idx in range(len(total_costs)):
                if active_mask[iter_idx]:
                    total_costs[item_idx] += main_agent_cost[iter_idx]
            assert len(all_categories)==len(responses_str)
            assert len(all_categories)==len(responses_str)
            assert len(all_indices)==len(all_categories)
            new_documents, new_code, new_attempts, dones, new_attempt_correct, new_costs, tool_summary,format_correctness, latencies, all_attempt_results,valid_answers_generated,used_llms = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, retrieved_documents=retrieved_documents, user_problems=gen_batch.non_tensor_batch['problem'],
                code_snippets=code_snippets,attempts=attempts,tokenizer=tokenizer,vllm_model_configs=vllm_model_configs,cur_output_dir=os.path.join(my_output_dir,f"global_step_{global_steps}",f"rollout_step_{step}",'code_execution'),
                answers=gen_batch.non_tensor_batch['answer'],total_costs=total_costs,all_categories=all_categories,all_transfer_dirs=all_transfer_dirs,
                all_input_messages=all_input_messages,all_input_tools=all_input_tools,all_transfer_paths=all_transfer_paths,all_model_mappings=all_model_mappings,
                all_tool_pricing=all_tool_pricing,all_indices=all_indices,topk_doc=topk_doc
            )
            for tmp_tool_l1 in tool_summary:
                assert isinstance(tmp_tool_l1,list)
                for tmp_tool in tmp_tool_l1:
                    if tmp_tool[0]!='':
                        tool_total += 1
            for i,iter_docs in enumerate(new_documents):
                assert isinstance(iter_docs,list)
                retrieved_documents[i] = merge_documents(main_list=retrieved_documents[i],sub_list=iter_docs)
            for i,c in enumerate(new_code):
                assert isinstance(c,list)
                if len(c)>0:
                    code_snippets[i] += c
            for i,a in enumerate(new_attempts):
                assert isinstance(a,list)
                if len(a)>0:
                    attempts[i] += a
            assert len(tool_summary)==len(new_attempt_correct)
            assert len(new_costs)==len(total_costs)
            assert len(active_mask)==len(total_costs)
            for iter_idx in range(len(total_costs)):
                if active_mask[iter_idx]:
                    total_costs[iter_idx] += new_costs[iter_idx]
            cur_turn_tools = [tool_summary[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['problem'])) if active_mask[local_idx]]
            assert len(cur_turn_tools) == round_input_ids.shape[0]
            all_turn_tools += cur_turn_tools
            cur_turn_success = [new_attempt_correct[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['problem'])) if active_mask[local_idx]]
            assert len(cur_turn_success) == round_input_ids.shape[0]
            all_turn_success += cur_turn_success
            cur_turn_costs = [new_costs[local_idx]+main_agent_cost[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['problem'])) if active_mask[local_idx]]
            assert len(cur_turn_costs) == round_input_ids.shape[0]
            all_turn_costs += cur_turn_costs
            cur_turn_used_llms = [used_llms[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['problem'])) if active_mask[local_idx]]
            assert len(cur_turn_used_llms) == round_input_ids.shape[0]
            all_turn_used_llms += cur_turn_used_llms
            cur_turn_valid_answers_generated = [valid_answers_generated[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['problem'])) if active_mask[local_idx]]
            assert len(cur_turn_valid_answers_generated) == round_input_ids.shape[0]
            all_turn_valid_answers_generated += cur_turn_valid_answers_generated
            cur_turn_latency = [latencies[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['problem'])) if active_mask[local_idx]]
            assert len(cur_turn_latency) == round_input_ids.shape[0]
            all_turn_latency += cur_turn_latency
            cur_turn_formats = [format_correctness[local_idx] for local_idx in range(len(gen_batch.non_tensor_batch['problem'])) if active_mask[local_idx]]
            assert len(cur_turn_formats) == round_input_ids.shape[0]
            all_turn_formats += cur_turn_formats

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask

        valid_answers = {}
        assert len(attempts)==len(cur_batch_indices)
        assert len(attempts)==len(cur_batch_repeat_ids)
        assert len(attempts)==len(cur_batch_output_files)
        for a,tmp_idx,tmp_repeat_id,tmp_output_file in zip(attempts,cur_batch_indices,cur_batch_repeat_ids,cur_batch_output_files):
            rollout_id = f"{tmp_idx}_____{tmp_repeat_id}"
            if not tmp_output_file:
                continue
            for tmp_a in a:
                if len(tmp_a['answer'].strip())>0:
                    valid_answers[rollout_id] = True
            try:
                correct = 0
                for subfile in os.listdir(os.path.join(tmp_output_file,'output')):
                    if subfile.endswith('.json'):
                        with open(os.path.join(tmp_output_file,'output',subfile)) as f:
                            r = json.load(f)
                        correct += r["reward_info"]["reward"]
                if correct>-1:
                    valid_answers[rollout_id] = True
                if correct>0:
                    example_correct_by_rollout_id[rollout_id] = 1
                else:
                    example_correct_by_rollout_id[rollout_id] = 0
            except Exception as error_1013:
                pass
        assert len(all_turn_index)==len(all_turn_valid_answers_generated)
        for example_idx,example_repeat_id,turn_valid_answer in zip(all_turn_index,all_turn_repeat_ids,all_turn_valid_answers_generated):
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            if not rollout_id in valid_answers:
                valid_answers[rollout_id] = turn_valid_answer
            elif not valid_answers[rollout_id]:
                valid_answers[rollout_id] = turn_valid_answer

        all_turn_input_ids = torch.cat(all_turn_input_ids, dim=0)
        all_turn_attention_mask = torch.cat(all_turn_attention_mask, dim=0)
        all_turn_position_ids = torch.cat(all_turn_position_ids, dim=0)
        all_turn_responses = torch.cat(all_turn_responses, dim=0)

        assert len(all_turn_index)==len(all_turn_success)
        assert len(all_turn_index)==len(all_turn_repeat_ids)
        assert len(all_turn_index)==len(all_turn_used_llms)
        assert len(all_turn_index)==len(all_turn_categories)
        assert len(all_turn_index)==len(all_turn_tools)
        assert len(all_turn_index)==len(all_turn_steps)
        example_correctness = {}
        for example_idx,example_repeat_id,turn_success in zip(all_turn_index,all_turn_repeat_ids,all_turn_success):
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            if turn_success or (rollout_id in example_correct_by_rollout_id and example_correct_by_rollout_id[rollout_id]):
                example_correctness[rollout_id] = True
            elif not rollout_id in example_correctness:
                example_correctness[rollout_id] = False
        example_costs = defaultdict(int)
        example_latency = defaultdict(int)
        for example_idx,example_repeat_id,turn_cost,turn_latency in zip(all_turn_index,all_turn_repeat_ids,all_turn_costs,all_turn_latency):
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            example_costs[rollout_id] += turn_cost
            example_latency[rollout_id] += turn_latency
        example_rewards = defaultdict(list)
        rewards_by_rollout_id = {}
        for turn_format,example_idx,example_repeat_id in zip(all_turn_formats,all_turn_index,all_turn_repeat_ids):
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            if str(efficiency_reward).lower()=='true' and example_correctness[rollout_id]:
                cost_reward = example_costs[rollout_id]*5
                latency_reward = example_latency[rollout_id]/500
                if cost_reward+latency_reward>0.8:
                    rewards_by_rollout_id[rollout_id] = int(example_correctness[rollout_id])-0.8
                    example_rewards[example_idx].append(int(example_correctness[rollout_id])-0.8)
                else:
                    rewards_by_rollout_id[rollout_id] = int(example_correctness[rollout_id])-latency_reward-cost_reward
                    example_rewards[example_idx].append(int(example_correctness[rollout_id])-latency_reward-cost_reward)
            else:
                rewards_by_rollout_id[rollout_id] = int(example_correctness[rollout_id])
                example_rewards[example_idx].append(int(example_correctness[rollout_id]))

        tool_counts = {}
        max_repeat_id = 0
        tool_counts_min = {}
        tool_counts_max = {}
        for turn_format,example_idx,example_repeat_id,turn_used_llms,turn_category,turn_step,turn_tools in zip(all_turn_formats,all_turn_index,all_turn_repeat_ids,all_turn_used_llms,all_turn_categories,all_turn_steps,all_turn_tools):
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            tool_counts_min[example_idx] = {}
            tool_counts_max[example_idx] = {}
            max_repeat_id = max(max_repeat_id,example_repeat_id)
            if not rollout_id in tool_counts:
                tool_counts[rollout_id] = defaultdict(int)
            if turn_tools and turn_tools[0][1]:
                tool_counts[rollout_id][turn_tools[0][1]] += 1
        for rollout_id,tool_nums in tool_counts.items():
            example_idx,example_repeat_id = rollout_id.split('_____')
            for tn,tc in tool_nums.items():
                if not tn in tool_counts_min[example_idx]:
                    tool_counts_min[example_idx][tn] = tc
                if not tn in tool_counts_max[example_idx]:
                    tool_counts_max[example_idx][tn] = tc
                tool_counts_min[example_idx][tn] = min(tool_counts_min[example_idx][tn],tc)
                tool_counts_max[example_idx][tn] = max(tool_counts_max[example_idx][tn],tc)
            if not 'accuracy' in tool_counts_min[example_idx]:
                tool_counts_min[example_idx]['accuracy'] = int(example_correctness[rollout_id])
            tool_counts_min[example_idx]['accuracy'] = min(tool_counts_min[example_idx]['accuracy'],int(example_correctness[rollout_id]))
            if not 'accuracy' in tool_counts_max[example_idx]:
                tool_counts_max[example_idx]['accuracy'] = int(example_correctness[rollout_id])
            tool_counts_max[example_idx]['accuracy'] = max(tool_counts_max[example_idx]['accuracy'],int(example_correctness[rollout_id]))
            if not 'cost' in tool_counts_min[example_idx]:
                tool_counts_min[example_idx]['cost'] = -example_costs[rollout_id]
            tool_counts_min[example_idx]['cost'] = min(tool_counts_min[example_idx]['cost'],-example_costs[rollout_id])
            if not 'cost' in tool_counts_max[example_idx]:
                tool_counts_max[example_idx]['cost'] = -example_costs[rollout_id]
            tool_counts_max[example_idx]['cost'] = max(tool_counts_max[example_idx]['cost'],-example_costs[rollout_id])
            if not 'latency' in tool_counts_min[example_idx]:
                tool_counts_min[example_idx]['latency'] = -example_latency[rollout_id]
            tool_counts_min[example_idx]['latency'] = min(tool_counts_min[example_idx]['latency'],-example_latency[rollout_id])
            if not 'latency' in tool_counts_max[example_idx]:
                tool_counts_max[example_idx]['latency'] = -example_latency[rollout_id]
            tool_counts_max[example_idx]['latency'] = max(tool_counts_max[example_idx]['latency'],-example_latency[rollout_id])
            
        rewards_by_rollout_id = {}
        example_rewards = defaultdict(list)
        for turn_format,example_idx,example_repeat_id in zip(all_turn_formats,all_turn_index,all_turn_repeat_ids):
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            if rollout_id in rewards_by_rollout_id:
                continue
            rewards_by_rollout_id[rollout_id] = 0
            if not example_correctness[rollout_id]:
                example_rewards[example_idx].append(rewards_by_rollout_id[rollout_id])
                continue
            cur_acc = int(example_correctness[rollout_id])
            cur_cost = example_costs[rollout_id]
            cur_latency = example_latency[rollout_id]
            cur_pref_vec = all_pref_vecs[rollout_id]
            features = list(tool_counts_min[example_idx].keys())#+['accuracy','latency','cost']
            for one_feature in features:
                if tool_counts_max[example_idx][one_feature]>tool_counts_min[example_idx][one_feature]:
                    rewards_by_rollout_id[rollout_id] += cur_pref_vec[one_feature]*(tool_counts[rollout_id][one_feature]-tool_counts_min[example_idx][one_feature])/(tool_counts_max[example_idx][one_feature]-tool_counts_min[example_idx][one_feature])
            one_feature = 'accuracy'
            if tool_counts_max[example_idx][one_feature]>tool_counts_min[example_idx][one_feature]:
                rewards_by_rollout_id[rollout_id] += cur_pref_vec[one_feature]*(cur_acc-tool_counts_min[example_idx][one_feature])/(tool_counts_max[example_idx][one_feature]-tool_counts_min[example_idx][one_feature])
            one_feature = 'latency'
            if tool_counts_max[example_idx][one_feature]>tool_counts_min[example_idx][one_feature]:
                rewards_by_rollout_id[rollout_id] += cur_pref_vec[one_feature]*(cur_latency-tool_counts_min[example_idx][one_feature])/(tool_counts_max[example_idx][one_feature]-tool_counts_min[example_idx][one_feature])
            one_feature = 'cost'
            if tool_counts_max[example_idx][one_feature]>tool_counts_min[example_idx][one_feature]:
                rewards_by_rollout_id[rollout_id] += cur_pref_vec[one_feature]*(cur_cost-tool_counts_min[example_idx][one_feature])/(tool_counts_max[example_idx][one_feature]-tool_counts_min[example_idx][one_feature])
            example_rewards[example_idx].append(rewards_by_rollout_id[rollout_id])

        example_indices = list(example_rewards.keys())
        example_reward_average = {}
        example_reward_std = {}
        from statistics import stdev
        for example_idx in example_indices:
            assert len(example_rewards[example_idx])>2
            example_reward_average[example_idx] = sum(example_rewards[example_idx])/len(example_rewards[example_idx])
            example_reward_std[example_idx] = stdev(example_rewards[example_idx])


        rewards = []
        selected_indices = []
        iter_index = -1
        for turn_format,example_idx,example_repeat_id,turn_used_llms,turn_category,turn_step,turn_tools in zip(all_turn_formats,all_turn_index,all_turn_repeat_ids,all_turn_used_llms,all_turn_categories,all_turn_steps,all_turn_tools):
            iter_index += 1
            rollout_id = f"{example_idx}_____{example_repeat_id}"
            cur_reward = (rewards_by_rollout_id[rollout_id]-example_reward_average[example_idx])/(example_reward_std[example_idx]+1e-6)
            selected_for_train = False
            if example_reward_std[example_idx]>0.1 and rollout_id in valid_answers and valid_answers[rollout_id] and turn_format:
                selected_indices.append(iter_index)
                selected_for_train = True
            if cur_reward>3:
                cur_reward = 3
            if cur_reward<-3:
                cur_reward = -3
            rewards.append(cur_reward)

        def selection_list(original_list,selection_indices):
            if isinstance(original_list,list):
                new_list = []
                for sidx in selection_indices:
                    new_list.append(original_list[sidx])
                return new_list
            else:
                return torch.index_select(original_list, dim=0, index=selection_indices)

        total_nodes = self.config.num_gpus 
        if len(selected_indices)<total_nodes:
            return None,None
        
        indices = torch.tensor(selected_indices)
        all_turn_input_ids = torch.index_select(all_turn_input_ids, dim=0, index=indices)
        all_turn_attention_mask = torch.index_select(all_turn_attention_mask, dim=0, index=indices)
        all_turn_position_ids = torch.index_select(all_turn_position_ids, dim=0, index=indices)
        all_turn_responses = torch.index_select(all_turn_responses, dim=0, index=indices)
        all_turn_ids = selection_list(all_turn_ids,selected_indices)
        all_turn_index = selection_list(all_turn_index,selected_indices)
        all_turn_turn_ids = selection_list(all_turn_turn_ids,selected_indices)
        all_turn_repeat_ids = selection_list(all_turn_repeat_ids,selected_indices)
        all_turn_answers = selection_list(all_turn_answers,selected_indices)
        all_turn_problems = selection_list(all_turn_problems,selected_indices)
        all_turn_tools = selection_list(all_turn_tools,selected_indices)
        all_turn_success = selection_list(all_turn_success,selected_indices)
        all_turn_costs = selection_list(all_turn_costs,selected_indices)
        all_turn_formats = selection_list(all_turn_formats,selected_indices)
        rewards = selection_list(rewards,selected_indices)

        cur_remainder = len(all_turn_ids)%total_nodes
        if cur_remainder!=0:
            all_turn_input_ids = all_turn_input_ids[:-cur_remainder]
            all_turn_attention_mask = all_turn_attention_mask[:-cur_remainder]
            all_turn_position_ids = all_turn_position_ids[:-cur_remainder]
            all_turn_responses = all_turn_responses[:-cur_remainder]
            all_turn_ids = all_turn_ids[:-cur_remainder]
            all_turn_index = all_turn_index[:-cur_remainder]
            all_turn_turn_ids = all_turn_turn_ids[:-cur_remainder]
            all_turn_repeat_ids = all_turn_repeat_ids[:-cur_remainder]
            all_turn_answers = all_turn_answers[:-cur_remainder]
            all_turn_problems = all_turn_problems[:-cur_remainder]
            all_turn_tools = all_turn_tools[:-cur_remainder]
            all_turn_success = all_turn_success[:-cur_remainder]
            all_turn_costs = all_turn_costs[:-cur_remainder]
            all_turn_formats = all_turn_formats[:-cur_remainder]
            rewards = rewards[:-cur_remainder]
        if len(all_turn_input_ids)<total_nodes:
            return None

        assert len(all_turn_input_ids)==len(rewards)
        final_dict = {
            'input_ids': all_turn_input_ids,
            'attention_mask': all_turn_attention_mask,
            'position_ids': all_turn_position_ids,
            'responses': all_turn_responses,
            'id': np.array(all_turn_ids),
            'index': np.array(all_turn_index),
            'turn_id': np.array(all_turn_turn_ids),
            'repeat_id': np.array(all_turn_repeat_ids),
            'answer': np.array(all_turn_answers),
            'reward': np.array(rewards),
        }
        return_dict = DataProto.from_single_dict(final_dict,meta_info=meta_info)
        return return_dict,tool_total/loop_batch_size

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True, retrieved_documents=None, user_problems=None, 
                                code_snippets=None, attempts=None, tokenizer=None,vllm_model_configs=None,cur_output_dir=None,answers=None,
                                total_costs=None,all_categories=None,all_transfer_dirs=None,all_input_messages=None,all_input_tools=None,all_transfer_paths=None,
                                all_model_mappings=None,all_tool_pricing=None,all_indices=None,topk_doc=30):
        cur_tool_calls, contents, format_correctness = self.postprocess_predictions(predictions,all_categories)

        assert len(cur_tool_calls) == len(active_mask)
        assert len(cur_tool_calls) == len(retrieved_documents)
        assert len(cur_tool_calls) == len(user_problems)
        assert len(cur_tool_calls) == len(code_snippets)
        assert len(cur_tool_calls) == len(attempts)
        assert len(cur_tool_calls) == len(answers)
        assert len(cur_tool_calls) == len(all_categories)
        assert len(cur_tool_calls) == len(all_transfer_dirs)
        assert len(cur_tool_calls) == len(all_input_messages)
        assert len(cur_tool_calls) == len(all_input_tools)
        assert len(cur_tool_calls) == len(all_transfer_paths)
        assert len(cur_tool_calls) == len(all_model_mappings)

        from tqdm import tqdm
        all_call_tool_arguments = []
        # semaphore = asyncio.Semaphore(512)
        for i, (iter_tool_calls, active, doc_list, user_problem, code_list, attempt_list, answer, cur_category, cur_transfer_dir,cur_input_messages,cur_input_tools,cur_transfer_path,cur_model_mapping,cur_tool_pricing,cur_index) in enumerate(zip(cur_tool_calls, active_mask, retrieved_documents, user_problems, code_snippets, attempts, answers, all_categories, all_transfer_dirs, all_input_messages, all_input_tools,all_transfer_paths,all_model_mappings,all_tool_pricing,all_indices)):
            call_tool_arguments = []
            if cur_category=='qa':
                for tid,tool_call in enumerate(iter_tool_calls):
                    valid_tool_call = True
                    if (not active) or (not isinstance(tool_call,dict)) or (set(list(tool_call.keys()))!={'name',"arguments"}) or (not tool_call['name'] in ALL_TOOLS):
                        valid_tool_call = False
                        continue
                    func_signature = ALL_TOOLS[tool_call['name']]
                    for parameter_name,parameter_values in func_signature.items():
                        if (not parameter_name in tool_call["arguments"]):
                            valid_tool_call = False
                        if (not tool_call["arguments"][parameter_name] in parameter_values) and parameter_values!='any':
                            valid_tool_call = False
                    if not valid_tool_call:
                        continue
                    assert isinstance(doc_list,list)
                    assert isinstance(code_list,list)
                    assert isinstance(attempt_list,list)
                    if tool_call['name']=='enhance_reasoning':
                        if not tool_call["arguments"]['model'] in cur_model_mapping:
                            continue
                        cur_model_to_call = cur_model_mapping[tool_call["arguments"]['model']]
                        if cur_model_to_call in ['o3','o3-mini','gpt-5-mini','gpt-5']:
                            doc_str = ''
                            for doc_idx, doc in enumerate(doc_list):
                                doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                            code_str = ''
                            for code_idx, code_piece in enumerate(code_list):
                                code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                            str_cut = cut_seq(tokenizer=tokenizer,seq=code_str,l=40000)
                            code_str = str_cut['string_after_cut']
                            code_str_len = str_cut['effective_length']
                            if not code_str.startswith('```') and len(code_str)>0:
                                code_str = '```\n'+code_str
                            context_str = cut_seq(tokenizer=tokenizer,seq=doc_str+code_str,l=100000)
                            context_str = context_str['string_after_cut']
                            if len(doc_str)>0:
                                context_str = 'Documents:\n'+context_str
                            call_tool_arguments.append({
                                'tool': tool_call['name'],
                                'model': tool_call["arguments"]['model'],
                                'context_str': context_str,
                                'vllm_model_configs': vllm_model_configs,
                                'cur_output_dir': cur_output_dir,
                                'id': i,
                                'problem': user_problem,
                                'category': cur_category,
                                'tid': tid,
                                'cur_model_mapping': cur_model_mapping,
                                'cur_tool_pricing': cur_tool_pricing,
                                'tokenizer': self.tokenizer,
                                'cur_index': cur_index,
                                'topk_doc': topk_doc,
                            })
                        elif 'qwen2.5' in cur_model_to_call.lower():
                            doc_str = ''
                            for doc_idx, doc in enumerate(doc_list):
                                doc_str += f"Doc {doc_idx+1}: {doc[:200]}\n\n"
                            code_str = ''
                            for code_idx, code_piece in enumerate(code_list):
                                code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                            str_cut = cut_seq(tokenizer=tokenizer,seq=code_str,l=16000)
                            code_str = str_cut['string_after_cut']
                            code_str_len = str_cut['effective_length']
                            if not code_str.startswith('```') and len(code_str)>0:
                                code_str = '```\n'+code_str
                            context_str = cut_seq(tokenizer=tokenizer,seq=doc_str+code_str,l=24000)
                            context_str = context_str['string_after_cut']
                            if len(doc_str)>0:
                                context_str = 'Documents:\n'+context_str
                            call_tool_arguments.append({
                                'tool': tool_call['name'],
                                'model': tool_call["arguments"]['model'],
                                'context_str': context_str,
                                'vllm_model_configs': vllm_model_configs,
                                'cur_output_dir': cur_output_dir,
                                'id': i,
                                'problem': user_problem,
                                'category': cur_category,
                                'tid': tid,
                                'cur_model_mapping': cur_model_mapping,
                                'cur_tool_pricing': cur_tool_pricing,
                                'tokenizer': self.tokenizer,
                                'cur_index': cur_index,
                                'topk_doc': topk_doc
                            })
                        elif 'llama' in cur_model_to_call.lower():
                            doc_str = ''
                            for doc_idx, doc in enumerate(doc_list):
                                doc_str += f"Doc {doc_idx+1}: {doc[:200]}\n\n"
                            code_str = ''
                            for code_idx, code_piece in enumerate(code_list):
                                code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                            str_cut = cut_seq(tokenizer=tokenizer,seq=code_str,l=8000)
                            code_str = str_cut['string_after_cut']
                            code_str_len = str_cut['effective_length']
                            if not code_str.startswith('```') and len(code_str)>0:
                                code_str = '```\n'+code_str
                            context_str = cut_seq(tokenizer=tokenizer,seq=doc_str+code_str,l=12000)
                            context_str = context_str['string_after_cut']
                            if len(doc_str)>0:
                                context_str = 'Documents:\n'+context_str
                            call_tool_arguments.append({
                                'tool': tool_call['name'],
                                'model': tool_call["arguments"]['model'],
                                'context_str': context_str,
                                'vllm_model_configs': vllm_model_configs,
                                'cur_output_dir': cur_output_dir,
                                'id': i,
                                'problem': user_problem,
                                'category': cur_category,
                                'tid': tid,
                                'cur_model_mapping': cur_model_mapping,
                                'cur_tool_pricing': cur_tool_pricing,
                                'tokenizer': self.tokenizer,
                                'cur_index': cur_index,
                                'topk_doc': topk_doc
                            })
                    elif tool_call['name']=='answer':
                        if not tool_call["arguments"]['model'] in cur_model_mapping:
                            continue
                        cur_model_to_call = cur_model_mapping[tool_call["arguments"]['model']]
                        if 'math' in cur_model_to_call.lower():
                            doc_str = ''
                            for doc_idx, doc in enumerate(doc_list):
                                doc_str += f"Doc {doc_idx+1}: {doc[:500]}\n\n"
                            code_str = ''
                            for code_idx, code_piece in enumerate(code_list):
                                code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                            str_cut = cut_seq(tokenizer=tokenizer,seq=code_str,l=1000)
                            code_str = str_cut['string_after_cut']
                            code_str_len = str_cut['effective_length']
                            if not code_str.startswith('```') and len(code_str)>0:
                                code_str = '```\n'+code_str
                            context_str = cut_seq(tokenizer=tokenizer,seq=doc_str+code_str,l=2000)
                            context_str = context_str['string_after_cut']
                            if len(doc_str)>0:
                                context_str = 'Documents:\n'+context_str
                        elif 'qwen' in cur_model_to_call.lower():
                            doc_str = ''
                            for doc_idx, doc in enumerate(doc_list):
                                doc_str += f"Doc {doc_idx+1}: {doc[:4000]}\n\n"
                            code_str = ''
                            for code_idx, code_piece in enumerate(code_list):
                                code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                            str_cut = cut_seq(tokenizer=tokenizer,seq=code_str,l=16000)
                            code_str = str_cut['string_after_cut']
                            code_str_len = str_cut['effective_length']
                            if not code_str.startswith('```') and len(code_str)>0:
                                code_str = '```\n'+code_str
                            context_str = cut_seq(tokenizer=tokenizer,seq=doc_str+code_str,l=24000)
                            context_str = context_str['string_after_cut']
                            if len(doc_str)>0:
                                context_str = 'Documents:\n'+context_str
                        elif cur_model_to_call in ['o3','o3-mini','gpt-5-mini','gpt-5']:
                            doc_str = ''
                            for doc_idx, doc in enumerate(doc_list):
                                doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                            code_str = ''
                            for code_idx, code_piece in enumerate(code_list):
                                code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                            str_cut = cut_seq(tokenizer=tokenizer,seq=code_str,l=40000)
                            code_str = str_cut['string_after_cut']
                            code_str_len = str_cut['effective_length']
                            if not code_str.startswith('```') and len(code_str)>0:
                                code_str = '```\n'+code_str
                            context_str = cut_seq(tokenizer=tokenizer,seq=doc_str+code_str,l=100000)
                            context_str = context_str['string_after_cut']
                            if len(doc_str)>0:
                                context_str = 'Documents:\n'+context_str
                        elif 'llama' in cur_model_to_call.lower() or 'nemotron-ultra' in cur_model_to_call.lower() or 'nemotron-super' in cur_model_to_call.lower() or 'phi' in cur_model_to_call.lower():
                            doc_str = ''
                            for doc_idx, doc in enumerate(doc_list):
                                doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                            code_str = ''
                            for code_idx, code_piece in enumerate(code_list):
                                code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                            str_cut = cut_seq(tokenizer=tokenizer,seq=code_str,l=40000)
                            code_str = str_cut['string_after_cut']
                            code_str_len = str_cut['effective_length']
                            if not code_str.startswith('```') and len(code_str)>0:
                                code_str = '```\n'+code_str
                            context_str = cut_seq(tokenizer=tokenizer,seq=doc_str+code_str,l=80000)
                            context_str = context_str['string_after_cut']
                            if len(doc_str)>0:
                                context_str = 'Documents:\n'+context_str
                        call_tool_arguments.append({
                            'tool': tool_call['name'],
                            'model': tool_call["arguments"]['model'],
                            'context_str': context_str,
                            'vllm_model_configs': vllm_model_configs,
                            'cur_output_dir': cur_output_dir,
                            'id': i,
                            'problem': user_problem,
                            'answer': answer,
                            'category': cur_category,
                            'tid': tid,
                            'cur_model_mapping': cur_model_mapping,
                            'cur_tool_pricing': cur_tool_pricing,
                            'tokenizer': self.tokenizer,
                            'cur_index': cur_index,
                            'topk_doc': topk_doc
                        })
                    elif tool_call['name'] in ['search']:
                        if not tool_call["arguments"]['model'] in cur_model_mapping:
                            continue
                        cur_model_to_call = cur_model_mapping[tool_call["arguments"]['model']]
                        if 'qwen' in cur_model_to_call.lower():
                            doc_str = ''
                            for doc_idx, doc in enumerate(doc_list):
                                doc_str += f"Doc {doc_idx+1}: {doc[:4000]}\n\n"
                            code_str = ''
                            for code_idx, code_piece in enumerate(code_list):
                                code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                            str_cut = cut_seq(tokenizer=tokenizer,seq=code_str,l=16000)
                            code_str = str_cut['string_after_cut']
                            code_str_len = str_cut['effective_length']
                            if not code_str.startswith('```') and len(code_str)>0:
                                code_str = '```\n'+code_str
                            context_str = cut_seq(tokenizer=tokenizer,seq=doc_str+code_str,l=24000)
                            context_str = context_str['string_after_cut']
                            if len(doc_str)>0:
                                context_str = 'Documents:\n'+context_str
                        elif cur_model_to_call in ['o3','o3-mini','gpt-5-mini','gpt-5']:
                            doc_str = ''
                            for doc_idx, doc in enumerate(doc_list):
                                doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                            code_str = ''
                            for code_idx, code_piece in enumerate(code_list):
                                code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                            str_cut = cut_seq(tokenizer=tokenizer,seq=code_str,l=40000)
                            code_str = str_cut['string_after_cut']
                            code_str_len = str_cut['effective_length']
                            if not code_str.startswith('```') and len(code_str)>0:
                                code_str = '```\n'+code_str
                            context_str = cut_seq(tokenizer=tokenizer,seq=doc_str+code_str,l=100000)
                            context_str = context_str['string_after_cut']
                            if len(doc_str)>0:
                                context_str = 'Documents:\n'+context_str
                        elif 'llama' in cur_model_to_call.lower() or 'nemotron-ultra' in cur_model_to_call.lower() or 'nemotron-super' in cur_model_to_call.lower() or 'phi' in cur_model_to_call.lower():
                            doc_str = ''
                            for doc_idx, doc in enumerate(doc_list):
                                doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                            code_str = ''
                            for code_idx, code_piece in enumerate(code_list):
                                code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                            str_cut = cut_seq(tokenizer=tokenizer,seq=code_str,l=40000)
                            code_str = str_cut['string_after_cut']
                            code_str_len = str_cut['effective_length']
                            if not code_str.startswith('```') and len(code_str)>0:
                                code_str = '```\n'+code_str
                            context_str = cut_seq(tokenizer=tokenizer,seq=doc_str+code_str,l=80000)
                            context_str = context_str['string_after_cut']
                            if len(doc_str)>0:
                                context_str = 'Documents:\n'+context_str
                        elif 'google' in cur_model_to_call.lower():
                            doc_str = ''
                            for doc_idx, doc in enumerate(doc_list):
                                doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                            code_str = ''
                            for code_idx, code_piece in enumerate(code_list):
                                code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                            str_cut = cut_seq(tokenizer=tokenizer,seq=code_str,l=2000)
                            code_str = str_cut['string_after_cut']
                            code_str_len = str_cut['effective_length']
                            if not code_str.startswith('```') and len(code_str)>0:
                                code_str = '```\n'+code_str
                            context_str = cut_seq(tokenizer=tokenizer,seq=doc_str+code_str,l=4000)
                            context_str = context_str['string_after_cut']
                            if len(doc_str)>0:
                                context_str = 'Documents:\n'+context_str
                        call_tool_arguments.append({
                            'tool': tool_call['name'],
                            'model': tool_call["arguments"]['model'],
                            'vllm_model_configs': vllm_model_configs,
                            'cur_output_dir': cur_output_dir,
                            'id': i,
                            'problem': user_problem,
                            'answer': answer,
                            'context_str': context_str,
                            'category': cur_category,
                            'tid': tid,
                            'cur_model_mapping': cur_model_mapping,
                            'cur_tool_pricing': cur_tool_pricing,
                            'tokenizer': self.tokenizer,
                            'cur_index': cur_index,
                            'topk_doc': topk_doc
                        })
            elif cur_category=='func_call':
                assert cur_transfer_dir
                call_tool_arguments.append({
                    'tool': iter_tool_calls,
                    'cur_output_dir': cur_output_dir,
                    'id': i,
                    'category': cur_category,
                    'input_messages': cur_input_messages,
                    'input_tools': cur_input_tools,
                    'tid': 0,
                    'transfer_path': cur_transfer_path,
                    'cur_model_mapping': cur_model_mapping,
                    'cur_tool_pricing': cur_tool_pricing,
                    'tokenizer': self.tokenizer,
                    'cur_index': cur_index,
                    'vllm_model_configs': vllm_model_configs,
                    'topk_doc': topk_doc
                })
            all_call_tool_arguments.append(call_tool_arguments)

        tool_responses = {}

        all_cache_arguments = []
        for call_tool_arguments in all_call_tool_arguments:
            cache_arguments = []
            for iter_a in call_tool_arguments:
                tmp_arg = {}
                for k,v in iter_a.items():
                    if not k in ['semaphore','tokenizer']:
                        tmp_arg[k] = v
                cache_arguments.append(tmp_arg)
            all_cache_arguments.append(cache_arguments)
        if not os.path.isdir(os.path.join(cur_output_dir,'tool_return')):
            os.makedirs(os.path.join(cur_output_dir,'tool_return'),exist_ok=True)
        tool_call_list = []
        for iter_call_argument in all_call_tool_arguments:
            if len(iter_call_argument)>0:
                tool_call_list.append([call_tool_all,{
                    'id': iter_call_argument[0]['id'],
                    'all_call_arguments': iter_call_argument
                }])
        tool_call_results = asyncio.run(run_all(tool_call_list))
        for return_contents in tool_call_results:
            tool_responses[return_contents['id']] = return_contents['all_tool_call_results']

        cur_documents, cur_code, cur_attempts, dones, attempt_correct,costs, tool_summary, latencies,all_attempt_results,valid_answers_generated,used_llms = [], [], [], [], [], [], [], [], [], [], []
        # for i,(active,previous_cost) in enumerate(zip(active_mask,total_costs)):
        for i,active in enumerate(active_mask):
            if not active:
                cur_documents.append([])
                cur_code.append([])
                cur_attempts.append([])
                dones.append(1)
                attempt_correct.append(False)
                costs.append(0)
                tool_summary.append([])
                latencies.append(0)
                all_attempt_results.append([])
                valid_answers_generated.append(False)
                used_llms.append([])
            elif not i in tool_responses:
                cur_documents.append([])
                cur_code.append([])
                cur_attempts.append([])
                dones.append(0)
                attempt_correct.append(False)
                costs.append(0)
                latencies.append(0)
                tool_summary.append([])
                all_attempt_results.append([])
                valid_answers_generated.append(False)
                used_llms.append([])
            else:
                assert isinstance(tool_responses[i],list)
                iter_tool_summary = []
                iter_documents = []
                iter_code = []
                iter_attempt = []
                iter_attempt_correct = []
                iter_dones = []
                iter_costs = []
                iter_latency = []
                iter_valid_answer = False
                iter_used_llms = []
                for cur_response in tool_responses[i]:
                    if not 'used_llm' in cur_response:
                        pass
                    else:
                        iter_used_llms.append(cur_response['used_llm'])
                    if 'model' in cur_response:
                        iter_tool_summary.append([cur_response['tool'],cur_response['model']])
                    elif 'query' in cur_response:
                        iter_tool_summary.append([cur_response['tool'],cur_response['query']])
                    else:
                        # tool_summary.append([cur_response['tool'],''])
                        iter_tool_summary.append([cur_response['tool'],''])
                    if cur_response['tool']=='enhance_reasoning':
                        # cur_documents.append([])
                        if len(cur_response['exec_result'].strip())>0:
                            iter_code.append({'code': cur_response['generated_code'], 'output': cur_response['exec_result']})
                    elif cur_response['tool']=='answer':
                        # cur_documents.append([])
                        # cur_code.append({})
                        iter_attempt.append({
                            'model': cur_response['model'],
                            'answer': cur_response['pred'],
                            'response': cur_response['response'],
                            'problem': cur_response['problem'],
                            'ground_truth': cur_response['answer'],
                            'correct': cur_response['correctness']
                        })
                        iter_dones.append(1)
                        if cur_response['correctness']:
                            iter_attempt_correct.append(True)
                        iter_valid_answer = True
                        break
                    elif cur_response['tool']=='search':
                        new_retriever_docs = []
                        for one_doc in cur_response['search_results_data'][::-1]:
                            assert isinstance(one_doc,str)
                            cur_doc_str = one_doc # ['title']+'\n'+one_doc['body']
                            new_retriever_docs.append(cur_doc_str)
                        # if len(new_retriever_docs)>len(iter_documents):
                        #     iter_documents += new_retriever_docs
                        # else:
                        iter_documents = merge_documents(main_list=iter_documents,sub_list=new_retriever_docs)
                            # updated_doc_list = []
                            # multiple = len(iter_documents)//len(new_retriever_docs)
                            # assert multiple>0
                            # idx1 = 0
                            # idx2 = 0
                            # while idx1<len(new_retriever_docs):
                            #     updated_doc_list.append(new_retriever_docs[idx1])
                            #     assert idx2<len(iter_documents)
                            #     for iter_idx in range(idx2,idx2+multiple):
                            #         updated_doc_list.append(iter_documents[iter_idx])
                            #     idx2 = idx2+multiple
                            # updated_doc_list += iter_documents[multiple*len(new_retriever_docs):]
                            # iter_documents = updated_doc_list
                        # cur_code.append({})
                        # cur_attempts.append({})
                        # dones.append(0)
                        # attempt_correct.append(False)
                        # iter_costs.append(0.005)
                    # elif cur_response['tool']=='finish':
                    #     iter_dones.append(1)
                    #     break
                    # else:
                    iter_latency.append(cur_response['latency'])
                    iter_costs.append(cur_response['cost'])
                cur_documents.append(iter_documents)
                cur_code.append(iter_code)
                cur_attempts.append(iter_attempt)
                if len(iter_dones)>0:
                    dones.append(iter_dones[-1])
                else:
                    dones.append(0)
                if len(iter_attempt_correct)>0:
                    attempt_correct.append(iter_attempt_correct[-1])
                else:
                    attempt_correct.append(False)
                all_attempt_results.append(iter_attempt_correct)
                valid_answers_generated.append(iter_valid_answer)
                costs.append(sum(iter_costs))
                latencies.append(sum(iter_latency))
                tool_summary.append(iter_tool_summary)
                used_llms.append(iter_used_llms)

        return cur_documents, cur_code, cur_attempts, dones, attempt_correct,costs,tool_summary,format_correctness,latencies,all_attempt_results,valid_answers_generated,used_llms

    def postprocess_predictions(self, predictions,all_categories):
        thoughts = []
        tool_calls = []
        formats = []
        assert len(all_categories)==len(predictions)
                
        for prediction,category in zip(predictions,all_categories):
            cur_thought = ''
            cur_all_tool_calls = []
            format_correct = False
            if isinstance(prediction, str): # for llm output
                if category=='qa':
                    cur_thought = prediction.split('<think>')[-1].split('</think>')[0]
                    components = prediction.split('<tool_call>')
                    added_tools = set()
                    for c in components:
                        components1 = c.split('</tool_call>')
                        for c1 in components1:
                            try:
                                tmp_tool_call = json.loads(c1)
                                assert set(list(tmp_tool_call.keys()))=={"name","arguments"}
                                assert tmp_tool_call['name'] in ALL_TOOLS
                                if tmp_tool_call['name'] in added_tools:
                                    continue
                                added_tools.add(tmp_tool_call['name'])
                                func_signature = ALL_TOOLS[tmp_tool_call['name']]
                                for parameter_name,parameter_values in func_signature.items():
                                    assert tmp_tool_call["arguments"][parameter_name] in parameter_values or parameter_values=='any'
                                cur_all_tool_calls.append(tmp_tool_call)
                            except:
                                pass
                    if len(cur_all_tool_calls)>0:
                        format_correct = True
                elif category=='func_call':
                    cur_thought = prediction.split('<think>')[-1].split('</think>')[0]
                    if '<tool_call>' in prediction:
                        components = prediction.split('<tool_call>')
                        for c in components:
                            components1 = c.split('</tool_call>')
                            for c1 in components1:
                                try:
                                    tmp_tool_call = json.loads(c1)
                                    cur_all_tool_calls.append(tmp_tool_call)
                                except:
                                    pass
                        if len(cur_all_tool_calls)>0:
                            format_correct = True
                    elif '<message>' in prediction:
                        cur_all_tool_calls.append(prediction.split('<message>')[-1].split('</message>')[0])
            
            thoughts.append(cur_thought)
            tool_calls.append(cur_all_tool_calls)
            formats.append(format_correct)
            
        return tool_calls, thoughts, formats

    def batch_search(self, queries: List[str] = None):
        if len(queries)>0:
            results = self._batch_search(queries)['result']
            return results
        else:
            return []

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            format_reference += f"Doc {idx+1}: {content}\n"

        return format_reference

    def serialize_documents(self, documents):
        documents_str = ''
        for idx, doc in enumerate(documents):
            documents_str += f"Doc {idx+1}: {doc}\n\n"
        return documents_str
