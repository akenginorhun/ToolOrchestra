# ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2511.21689)
[![Code](https://img.shields.io/badge/GitHub-Link-orange)](https://github.com/NVlabs/ToolOrchestra/)
[![Model](https://img.shields.io/badge/HuggingFace-Model-green)](https://huggingface.co/nvidia/Orchestrator-8B)
[![Data](https://img.shields.io/badge/HuggingFace-Data-blue)](https://huggingface.co/datasets/nvidia/ToolScale)
[![Website](https://img.shields.io/badge/Web-Page-purple)](https://research.nvidia.com/labs/lpr/ToolOrchestra/)


[Hongjin Su*](https://hongjin-su.github.io/), [Shizhe Diao*](https://shizhediao.github.io/), [Ximing Lu](https://gloriaximinglu.github.io/), [Mingjie Liu](https://research.nvidia.com/person/mingjie-liu), [Jiacheng Xu](https://jiacheng-xu.github.io/), [Xin Dong](https://simonxin.com/), [Yonggan Fu](https://www.yongganfu.com/), [Peter Belcak](https://pbelcak.com/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Hongxu Yin](https://hongxu-yin.github.io/), [Yi Dong](https://www.linkedin.com/in/yi-dong-04057b18/), [Evelina Bakhturina](https://developer.nvidia.com/blog/author/ebakhturina/), [Tao Yu](https://taoyds.github.io/), [Yejin Choi](https://yejinc.github.io/), [Jan Kautz](https://jankautz.com/), [Pavlo Molchanov](https://www.pmolchanov.com/)  

<span style="color: rgb(133, 184, 55);">**NVIDIA**</span>,  <span style="color: rgb(184, 154, 55);">**The University of Hong Kong**</span>  
*Equal Contribution

<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/NVlabs/ToolOrchestra/main/assets/results_figure.png" alt="ToolOrchestra Performance" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

We introduce ToolOrchestra, a method for training small orchestrators that coordinate the use of intelligent tools. By using both tools and specialized models, ToolOrchestrator surpasses GPT-5 while is much more efficient. Given a task, the Orchestrator alternates between reasoning and tool calling in multiple turns to solve it. Orchestrator interacts with a diverse tool set, including basic tools (e.g., web search, code interpreter), specialized LLMs (e.g., coding models, math models) and generalist LLMs (e.g., GPT-5, Llama-Nemotron-Ultra-253B, Claude Opus 4.1). In training, Orchestrator is jointly optimized by outcome, efficiency and preference rewards via end-to-end reinforcement learning. To aid RL training, we develop an automatic pipeline to synthesize both environment and tool-call tasks at scale.


<p align="center">
    <img src="https://raw.githubusercontent.com/NVlabs/ToolOrchestra/main/assets/method.png" width="100%"/>
<p>

With ToolOrchestra, we produce Orchestrator-8B, a state-of-the-art 8B parameter orchestration model designed to solve complex, multi-turn agentic tasks by coordinating a diverse set of expert models and tools. 
On HLE, Orchestrator-8B achieves a score of 37.1%, outperforming GPT-5 (35.1%) while being 2.5x more efficient. On τ2-Bench and FRAMES, Orchestrator-8B surpasses GPT-5 by a wide margin while using only about 30% of the cost.


## Setup Environment
```
# git clone this repository
git clone https://gitlab-master.nvidia.com/dler/toolorchestra
cd toolorchestra

# change the directory to `toolorchestra`
cd toolorchestra

# download index files and checkpoints
git clone https://huggingface.co/datasets/multi-train/index
export INDEX_DIR='/path/to/index'
git clone https://huggingface.co/multi-train/ToolOrchestrator
export CHECKPOINT_PATH='/path/to/checkpoint'

# environment setup for training
conda create -n toolorchestra python=3.12 -y
conda activate toolorchestra
pip install -r requirements.txt
pip install -e training/rollout

# environment setup for retrieval
conda create -n retriever python=3.12 -y
conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu
pip install uvicorn fastapi

# environment setup for vllm models
conda create -n vllm1 python=3.12 -y
pip install torch transformers vllm
cd evaluation/tau2-bench
pip install -e .

# set environment variables
export HF_HOME="/path/to/huggingface"
export REPO_PATH="/path/to/this_repo"
export CKPT_DIR="/path/to/checkpoint"
```

## Training

```bash
cd training
python resume_h100.py
```

## Evaluation

```bash
cd evaluation
# eval HLE (require env vllm1 and retriever)
python run_hle.py

# eval Frames (require env vllm1 and retriever)
python run_frames.py

# eval tau2-bench (require env vllm1, double check tau2-bench is installed in eval directory)
cd tau2-bench/
python run.py
```

## Search API
Please go to https://app.tavily.com/home and apply for a Tavily API key.
```
export TAVILY_KEY="your key"
```


## How to Make Changes
- Modify function `get_llm_response` in `LLM_CALL.py` to change LLM call to service beyond vllm and openai
- Modify line `455-458` in `eval_hle.py` and `506-509` in `eval_frames.py` to modify prompts
- Substitute tool_config in line 27 in `eval_frames.py` and line 27 in `eval_hle.py` if using different tool set.
- Modify `tools.json` and `call_tool` function in `eval_hle.py` to modify the tools and models.
- If you want to run multiple experiments in parallel, modify variables `{EXPERIMENT_NAME1}`, `{EXPERIMENT_NAME2}`, `{EXPERIMENT_NAME3}` in `training/resume_h100.py`, which should correspond to the file name and job name in `{EXPERIMENT_NAME1}.sh`, `{EXPERIMENT_NAME2}.sh`, `{EXPERIMENT_NAME3}.sh` under the directory.

## License

The code included in this project is licensed under the [Apache 2.0 license](https://github.com/NVlabs/ToolOrchestra/blob/main/LICENSE).

## Citation

If you find this repository useful, please consider giving ⭐ and citing our [paper](https://arxiv.org/abs/2511.21689):

```citation
@misc{toolorchestra,
      title={ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration}, 
      author={Hongjin Su and Shizhe Diao and Ximing Lu and Mingjie Liu and Jiacheng Xu and Xin Dong and Yonggan Fu and Peter Belcak and Hanrong Ye and Hongxu Yin and Yi Dong and Evelina Bakhturina and Tao Yu and Yejin Choi and Jan Kautz and Pavlo Molchanov},
      year={2025},
      eprint={2511.21689},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.21689}, 
}
```
