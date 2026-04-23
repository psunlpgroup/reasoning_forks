import json
import argparse
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
parser.add_argument("--apply_chat_template", action="store_true")
parser.add_argument("--tokenizer_path", type=str, default=None)

args = parser.parse_args()

DATASET_NAME = args.dataset_name
SAVE_DIR = Path(args.save_dir)
APPLY_CHAT_TEMPLATE = args.apply_chat_template
TOKENIZER_PATH = args.tokenizer_path

# =======================
# Load template
# =======================
QUESTION_PROMPT_TEMPLATE = None

OUTPUT_PATH = SAVE_DIR / f"{DATASET_NAME}.prompts.csv"

# =======================
# Prompt builder
# =======================
def build_prompt(
    question: str,
    template: str | None,
    step_by_step: bool = True,
    tokenizer=None,
    apply_chat_template: bool = False,
):
    if tokenizer and apply_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    elif template:
        prompt = template.replace("<question>", question) if "<question>" in template else template.format(question, "")
        return prompt
    # else:
    #     base = "Please answer the following math question."
    #     if step_by_step:
    #         base += " You should think step by step to solve it."
    #     prompt = (
    #         f"{base}\n\n"
    #         "Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"
    #         f"Question:\n{question}\n\n"
    #     )
    

    


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
# Tokenizer (optional)
# =======================
tokenizer = None
if APPLY_CHAT_TEMPLATE:
    if not TOKENIZER_PATH:
        raise ValueError("TOKENIZER_PATH must be set when --apply_chat_template is used")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# =======================
# Process prompts
# =======================
rows = []
for e in examples:
    prompt = build_prompt(
        question=e["Question"],
        template=QUESTION_PROMPT_TEMPLATE,
        step_by_step=True,
        tokenizer=tokenizer,
        apply_chat_template=APPLY_CHAT_TEMPLATE,
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