# 🚀 High-Throughput Batch Sampling

To evaluate how different prefixes and reasoning modes affect model outputs across various benchmarks, we use a scalable, multi-GPU batch sampling pipeline. The process is driven by individual YAML configuration files and orchestrated by a custom bash scheduler.

### 1. Job Configuration (sampler_config.yaml)
Each experimental run (e.g., testing a specific prefix on a specific model) is defined by a sampler_config.yaml file located in the job's directory. This file dictates the vLLM engine parameters and experimental settings.

Example Configuration:

```YAML
sampler:
  class: VLLMSampler
  model_name: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  temperature: 0.6
  top_p: 0.95
  top_k: -1
  max_tokens: 32768
  max_model_len: 32768
  trust_remote_code: true
  thinking_prefix: Alright  # The experimental prefix applied to the prompt
```

### 2. Multi-GPU Scheduler (spawn_sampling.sh)
Instead of running generation tasks sequentially, we use a bash-based scheduler to queue and dispatch jobs across multiple GPUs in parallel.

The script operates in three main steps:

- Job Definition & Queue Building: The script defines an array of `JOB_SPLIT_PAIRS` using the format `"JOB_DIR|RUN_NAME|SPLIT|NUM_SAMPLES"`. It iterates through this list, locates the corresponding sampler_config.yaml directories, and builds a flat queue of all pending tasks. Examples are given in the `runs/ds-distill/ds-qwen-1.5b` directory.

- Dynamic Dispatch: The script monitors the AVAILABLE_GPUS array (e.g., (6 7)). It assigns the next job in the queue to the first idle GPU using CUDA_VISIBLE_DEVICES. It tracks Process IDs (PIDs) and automatically pushes new jobs to GPUs as soon as they finish their current workload, ensuring maximum hardware utilization.

- Execution: Under the hood, the scheduler triggers src/inference/run_sampling.py, passing the specific configuration directory, target dataset split, and assigned GPU ID.

### 3. Running the Pipeline
To execute the batch sampling process:

1. Ensure your sampler_config.yaml files are properly placed in their respective run directories (e.g., runs/ds-distill/ds-qwen-1.5b/prefix_Alright/).

2. Update the AVAILABLE_GPUS array in the bash script to match your hardware availability.

3. Execute the script:

```Bash
bash scripts/run_batch.sh
```
The script will handle graceful shutdowns; if you cancel the process (Ctrl+C), it will trap the exit signal and clean up all running background GPU processes to prevent memory leaks.