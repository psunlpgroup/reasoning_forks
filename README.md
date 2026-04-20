# Why Do Reasoning Models Lose Coverage? The Role of Data and Forks in the Road

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the data generation scripts, training pipelines, and evaluation code for the paper [**Why Do Reasoning Models Lose Coverage? The Role of Data and Forks in the Road**]. 

We investigate the phenomenon of coverage shrinkage in reasoning-focused Large Language Models (LLMs). We demonstrate that this collapse is driven by indecipherable "decision points" (or forks in the road) in fine-tuning data, and we propose practical mitigations through data diversity design and first-token manipulation.

## 📌 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference & Evaluation](#inference--evaluation)
- [Citation](#citation)

## 🔍 Overview

This study is mainly aimed to deepen our understanding of coverage shrinkage in reasoning models and provides a data-centric perspective on this phenomenon.

1. We present a systematic, data-centric study of coverage shrinkage in reasoning post-trained models, aiming to understand its underlying factors.
2. We identify **forks-in-the-road patterns** in fine-tuning data as a key driver of coverage shrinkage, and analyze this effect through targeted case studies such as graph branching and alternative mathematical reasoning strategies.
3. Through controlled experiments and training-dynamics analysis, we find a strong correlation between the structure of decision points in data and the severity of coverage shrinkage, providing empirical evidence for the role of data on such behavior.
4. Motivated by our findings, we introduce two simple diversity-aware data synthesis and decoding strategies, and present proof-of-concept results demonstrating their effectiveness in mitigating shrinkage. These results suggest that the lost coverage is not permanently forgotten, but instead suppressed, and can be recovered through inference-time intervention.

## ⚙️ Installation

We recommend using [`uv`](https://github.com/astral-sh/uv) for lightning-fast Python environment management and dependency installation.

```bash
# 1. Install uv (if you haven't already)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# 2. Create a virtual environment using Python 3.12
uv venv --python 3.12

# 3. Activate the environment
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# 4. Install requirements
uv pip install -r requirements.txt
```


## 📊 Data Preparation
We provide scripts to recreate both the synthetic and mathematical reasoning datasets.

### Synthetic Graph Navigation
Generate the 6,400 SFT training samples, 1600 RLVR training samples, and 1,000 test samples in the Alpaca instruction format.

```Bash
python src/data_generation/gen_math_graph.py
```

### Mathematical Reasoning Modes
Process the OpenMathInstruct-1 and OpenMathInstruct-2 datasets to evaluate Data-level vs. Problem-level diversity.

```Bash
python src/data_generation/prepare_math.py --dataset gsm8k --strategy problem_level
```

### CounterFactual Arithmetic
Generate the base-7, 9, 11, and 12 counterfactual arithmetic datasets to test under-thinking phenomena.

[] TODO

### Simple knowledge questions - CapitalQA
Prepare the CapitalQA dataset to evaluate over-thinking behaviors on simple factual queries.

[] TODO

## 🚀 Training

### Supervised Fine-Tuning (SFT)
Run the 16-epoch SFT pipeline used for Qwen-2.5-0.5B and EvoLM-1B.

```Bash
bash run_sft_mathgraph.sh pathstar_2_10_forward qwen2.5_0.5b
bash run_sft_mathgraph.sh pathstar_2_10_reverse qwen2.5_0.5b
```

### Reinforcement Learning (RLVR)
Apply Group Relative Policy Optimization (GRPO) on the SFT checkpoints.

```Bash
bash run_grpo.sh qwen2.5_0.5b-pathstar_2_10-forward-sftep1 4
```


## 🧠 Inference & Evaluation

We use a scalable, multi-GPU batch sampling pipeline orchestrated by a custom bash scheduler.

1. Run High-Throughput Batch Sampling Configure your AVAILABLE_GPUS inside the script, then spawn the batch inference jobs. This process is driven by individual `sampler_config.yaml` files located in your target job directories.
```Bash
bash spawn_sampling.sh
```

2. Evaluate `pass@k` Metrics Once the generations are complete, compute the detailed `pass@k` coverage metrics using the optimized evaluation script.

```Bash
python src/math_eval/evaluate_pass_k.py \
  --dirs runs/ds-distill/ds-qwen-1.5b/default runs/ds-distill/ds-qwen-1.5b/prefix_Alright \
  --dataset math \
  --split math500_test \
  --workers 16
```

## 📝 Citation
If you find this code or our findings useful in your research, please cite our paper: