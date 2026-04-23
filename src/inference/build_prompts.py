import json
import argparse
from multiprocessing import Value
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

# =======================
# Dataset paths
# =======================
DATA_ROOT = Path("./datasets")
DATASET_PATHS = {
    "math500": DATA_ROOT / "math_benchmarks/MATH-500/test.json",
    "aime24": DATA_ROOT / "math_benchmarks/AIME24/test.json",
    "aime25": DATA_ROOT / "math_benchmarks/AIME25/test.json",
    "gsm8k": DATA_ROOT / "math_benchmarks/gsm8k/test.json",
    "arithchain_2_10": DATA_ROOT / "arithchain_2_10/test.json",
    "counterfact_addition": DATA_ROOT / "arithmetic_counterfact.json",
    "capitalQA": DATA_ROOT / "capitalQA.json",
}

# =======================
# Args
# =======================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--tokenizer_path", type=str)
parser.add_argument(
    "--chat_template_path",
    type=str,
    default=None,
    help="Path to chat template file to apply. E.g., 'templates/sharegpt_chat_template.json', etc."
)


args = parser.parse_args()

DATASET_NAME = args.dataset_name
SAVE_DIR = Path(args.save_dir)
TOKENIZER_PATH = args.tokenizer_path
OUTPUT_PATH = SAVE_DIR / f"{DATASET_NAME}.prompts.csv"

# =======================
# Prompt builder
# =======================
def build_prompt(
    question: str,
    prompt_template: str | None = None,
    tokenizer=None,
):
    prompt = question
    if prompt_template:
        prompt = prompt_template.replace("<question>", prompt) if "<question>" in prompt_template else prompt_template.format(question, "")

    if tokenizer:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    
    return prompt
    
# =======================
# Load dataset
# =======================
if DATASET_NAME not in DATASET_PATHS:
    raise ValueError(f"Unknown dataset: {DATASET_NAME}")

data_path = DATASET_PATHS[DATASET_NAME]

with open(data_path, "r", encoding="utf-8") as f:
    examples = json.load(f)

print(f"Num eval samples: {len(examples)}")

# =======================
# Tokenizer
# =======================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
# Determine chat_template: Use file if path provided, else use tokenizer default
if getattr(args, "chat_template_path", None) is not None:
    with open(args.chat_template_path, "r", encoding="utf-8") as f:
        chat_template = f.read()
    tokenizer.chat_template = chat_template

# =======================
# Process prompts
# =======================
rows = []
for e in examples:
    prompt = build_prompt(
        question=e["Question"],
        tokenizer=tokenizer,
    )

    rows.append(
        {
            "question_id": e["id"],
            "prompt_id": e["id"],
            "question": e["Question"],
            "answer": e["answer"],
            "prompt": prompt,
        }
    )

df = pd.DataFrame(rows)

# =======================
# Save
# =======================
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved to {OUTPUT_PATH}")