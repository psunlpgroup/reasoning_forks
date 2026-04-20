import os
import pandas as pd
import yaml
import time
from pathlib import Path
import random
import argparse

from vllm_sampler import VLLMSampler, DiverPathVLLMSampler

parser = argparse.ArgumentParser(description="Run the sampling script.")
parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Name of the dataset")
parser.add_argument("--split", type=str, default="test", help="Data split to use")
parser.add_argument("--subset_num", type=int, default=None, help="Number of subset to use (optional)")
parser.add_argument("--n", type=int, default=1, help="Number of generations per prompt")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--gpu_memory_utilization", type=float, default=None)
parser.add_argument("--enforce_eager", action="store_true")

parser.add_argument("--job_dir", type=str, required=True)
parser.add_argument("--sampler_config_dir", type=str, required=True)

args = parser.parse_args()

job_dir = Path(args.job_dir)    
prompt_csv_path = job_dir / f'{args.split}.prompts.csv'
sampler_config_dir = job_dir / args.sampler_config_dir

with open(sampler_config_dir / "sampler_config.yaml", "r") as f:
    sampler_config = yaml.safe_load(f)
    
prompt_df = pd.read_csv(prompt_csv_path)

# Extract tokenizer config and sampler config
tokenizer_config = sampler_config.get("tokenizer", {"path": None})
sampler_config_section = sampler_config.get("sampler", {})

random.seed(int(time.time()))
for i in range(os.getpid()):
    sampler_config_section['seed'] = random.randint(1, 2**32)

# Dynamically load the sampler class
sampler_class_name = sampler_config_section.get("class", "ChatCompletionSampler")
sampler_classes = {
    "VLLMSampler": VLLMSampler,
    "DiverPathVLLMSampler": DiverPathVLLMSampler,
}
SamplerClass = sampler_classes[sampler_class_name]

# Remove keys that are not arguments to SamplerClass.__init__
init_args = {}
thinking_prefix = None
stop_str = None
for k, v in sampler_config_section.items():
    if k == "class":
        continue
    if k == "model_name":
        init_args["model_name_or_path"] = v
    elif k == "stop":
        stop_str = v
    elif k == "thinking_prefix":
        thinking_prefix = v
    else:
        init_args[k] = v

if args.gpu_memory_utilization is not None:
    init_args['gpu_memory_utilization'] = args.gpu_memory_utilization

if args.enforce_eager:
    init_args['enforce_eager'] = args.enforce_eager

init_args['seed'] = sampler_config_section['seed']
init_args['tokenizer'] = tokenizer_config['path']
    
print(init_args)
print("Thinking Prefix:", thinking_prefix)
print("Stop String:", stop_str) 

# Create the sampler
sampler = SamplerClass(**init_args)

if thinking_prefix is not None:
    prompts = prompt_df['prompt'].apply(lambda x: x + thinking_prefix).values
else:
    prompts = prompt_df['prompt'].values

print("First Prompt Preview:\n", prompts[0])

response = sampler.complete(prompts, args.n, stop_str)

generations = []
for res, (_, row) in zip(response, prompt_df.iterrows()):
    # Extract metadata with fallbacks in case of sampler errors
    prompt_tokens = res.response_metadata.get("prompt_tokens", 0)
    completion_tokens_list = res.response_metadata.get("completion_tokens", [0] * len(res.choices))
    
    for idx, out in enumerate(res.choices):
        # Match the choice to its specific completion token count
        comp_tokens = completion_tokens_list[idx] if idx < len(completion_tokens_list) else 0
        
        generations.append((
            row['question_id'], 
            row['prompt_id'], 
            out.response_text, 
            None, 
            row['answer'], 
            sampler_config,
            prompt_tokens,
            comp_tokens
        ))

# Add the new token columns to the DataFrame
gen_df = pd.DataFrame(
    data=generations, 
    columns=[
        'question_id', 
        'prompt_id', 
        'response', 
        'pred_answer', 
        'gt_answer', 
        'sampler_config',
        'prompt_tokens',
        'completion_tokens'
    ]
)

# Standardized timestamp formatting
timestamp = time.strftime("%Y%m%d_%H%M%S")
seed = sampler_config_section['seed']
generation_csv_name = f"{args.split}.generations_seed{seed}.{timestamp}_{args.gpu_id}.csv"

gen_df.to_csv(sampler_config_dir / generation_csv_name, index=False)
print(f"Saved generations to {sampler_config_dir / generation_csv_name}")