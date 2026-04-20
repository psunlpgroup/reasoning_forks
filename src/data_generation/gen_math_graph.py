import argparse
import copy
import json
import os
import random
import re
import string

import datasets
import sympy as sp

# ==========================================
# Core Problem Generation Functions
# ==========================================

def generate_base_problem(math_world, num_constants, premise_nodes, target_node, max_const_value=20):
    """Generates the underlying math rules, premises, and letter mappings without solving it."""
    nodes = list(set(re.findall(r"(<\|[a-zA-Z0-9_]+\|>)", math_world)))

    # Assign single letters to nodes
    letters = list(string.ascii_lowercase)
    if len(nodes) > len(letters):
        raise ValueError("Too many nodes for single-letter mapping. Need a broader naming scheme.")
    
    random.shuffle(letters)
    node_to_letter = {node: letters[i] for i, node in enumerate(sorted(nodes))}
    
    # Initialize constants if needed
    if num_constants > 0:
        inited_math_world = math_world.format(*[random.randint(1, max_const_value) for _ in range(num_constants)])
    else:
        inited_math_world = math_world
        
    # Replace node tokens with assigned letters
    for node in nodes:
        inited_math_world = inited_math_world.replace(node, node_to_letter[node])
    
    rules = {}
    value_dict = {}
    
    # Parse rules and known values
    for r in inited_math_world.split(","):
        lhs, rhs = r.split("=")
        if rhs.strip().isnumeric():
            value_dict[lhs] = int(rhs)
            rules[lhs] = rhs
        rules[lhs] = str(sp.sympify(rhs))

    for n in premise_nodes:
        node_letter = node_to_letter[n]
        value_dict[node_letter] = random.randint(1, max_const_value)
        rules[node_letter] = str(value_dict[node_letter])

    premises = {node_to_letter[n]: value_dict[node_to_letter[n]] for n in premise_nodes}
    target = node_to_letter[target_node]

    return {
        "rules": rules,
        "premises": premises,
        "target": target,
        "node_to_letter": node_to_letter,
        "value_dict_init": copy.deepcopy(value_dict)
    }

def generate_solution_trace(base_problem, solution_nodes, reverse=False):
    """Generates a specific reasoning trace (forward or reverse) for a given base problem."""
    rules = base_problem["rules"]
    value_dict = copy.deepcopy(base_problem["value_dict_init"])
    target = base_problem["target"]
    node_to_letter = base_problem["node_to_letter"]
    
    steps = []
    if not reverse:
        for step_node in solution_nodes:
            var_name = node_to_letter[step_node]
            step_exp = rules[var_name]
            var_value = int(sp.sympify(step_exp).subs(value_dict))
            value_dict[var_name] = var_value
            steps.append((var_name, step_exp, var_value))
        target_value = value_dict[target]
    else:
        expr_dict = {}
        target_expression = sp.sympify(rules[target])
        for step_node in solution_nodes:
            var_name = node_to_letter[step_node]
            step_exp = rules[var_name]
            expr_dict[var_name] = sp.sympify(step_exp)
            target_expression = target_expression.subs(expr_dict)
            steps.append((var_name, step_exp, str(target_expression)))
        target_value = int(target_expression)

    return {
        "problem": {
            "rules": rules,
            "premises": base_problem["premises"],
            "target": target
        }, 
        "answer": {
            "steps": steps,
            "target_var_name": target, 
            "target_value": target_value,
            "value_dict": value_dict,
            "is_reverse": reverse,
        }
    }

# ==========================================
# Text Formatting Functions
# ==========================================

RELATION_TEMPLATES = ["{} = {}"]
GIVEN_TEMPLATES = ["{} = {}"]
PROBLEM_TEMPLATES = [
    lambda rels, givens, query: (
        f"Let each letter represent a numerical variable. These variables are defined as follows: {rels}. "
        f"{f'If {givens}, what is the resulting value of {query}?' if givens else f'What is the resulting value of {query}?'}"
    ),
    lambda rels, givens, query: (
        f"Consider a system of variables where each variable is defined as follows: {rels}. "
        f"{f'If {givens}, determine the value of {query}.' if givens else f'Determine the value of {query}.'}"
    ),
]

def generate_problem_description(problem: dict, shuffle=True):
    relation_descriptions = [RELATION_TEMPLATES[0].format(k, expr) for k, expr in problem['rules'].items()]
    relation_descriptions = '; '.join(relation_descriptions)
    
    given_str = None
    if problem['premises']:
        given_strs = [GIVEN_TEMPLATES[0].format(k, v) for k, v in problem['premises'].items()]
        if shuffle: 
            random.shuffle(given_strs)
        given_str = ", ".join(given_strs)

    template = PROBLEM_TEMPLATES[1]
    return template(relation_descriptions, given_str, problem['target'])

def generate_answer_description(problem, answer):
    is_reverse = answer['is_reverse']
    steps = answer['steps']
    target = answer['target_var_name']
    
    lines = ["To find the target value, we compute the following variables step by step:"]
    if not is_reverse:
        for i, (var, expr, val) in enumerate(steps, 1):
            lines.append(f"{i}. {var} = {expr} = {val}.")
        lines.append(f"\nThus, {steps[-1][0]} = \\boxed{{{steps[-1][2]}}}.")
    else:
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. Substitute {step[0]} = {step[1]} into the target expression, yielding {target} = {step[2]}.")
        lines.append(f"\nThus, {target} = \\boxed{{{steps[-1][2]}}}.")
        
    return "\n".join(lines)

def convert_sample_to_prompt(sample, world_id, direction, shuffle=True):
    return {
        "question": generate_problem_description(sample['problem'], shuffle=shuffle),
        "answer": generate_answer_description(sample['problem'], sample['answer']),
        "meta": json.dumps(sample),
        "world_id": world_id,
        "direction": direction
    }

# ==========================================
# Dataset Pipeline
# ==========================================

def generate_unique_problems(size, world):
    """Generates a pool of unique base problems (without attaching solutions yet)."""
    problems = []
    seen_rules = set() 
    
    while len(problems) < size:
        base_problem = generate_base_problem(
            world['math_world'], 
            world['num_constants'], 
            world['premise_nodes'], 
            world['target_node']
        )
        
        # O(1) uniqueness check using frozenset
        rules_frozenset = frozenset(base_problem['rules'].values())
        if rules_frozenset in seen_rules:
            continue
            
        seen_rules.add(rules_frozenset)
        problems.append(base_problem)
        
    return problems

def make_map_fn(split):
    def process_fn(example, idx):
        meta = json.loads(example['meta'])
        question = example["question"]
        return {
            "data_source": example['world_id'],
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "number", "ground_truth": meta["answer"]["target_value"]},
            "answer": example["answer"],
            "question": question,
            "direction": example["direction"],
            "extra_info": {
                "split": split,
                "index": idx,
            },
        }
    return process_fn

# ==========================================
# Main Execution
# ==========================================

WORLDS = [
    {
        "id": "pathstar_2_10",
        "math_world": "<|a1|>=<|p0|>+{},<|a2|>=<|a1|>+{},<|a3|>=<|a2|>+{},<|a4|>=<|a3|>+{},<|a5|>=<|a4|>+{},<|a6|>=<|a5|>+{},<|a7|>=<|a6|>+{},<|a8|>=<|a7|>+{},<|a9|>=<|a8|>+{},<|a10|>=<|a9|>+{},<|b1|>=<|p0|>+{},<|b2|>=<|b1|>+{},<|b3|>=<|b2|>+{},<|b4|>=<|b3|>+{},<|b5|>=<|b4|>+{},<|b6|>=<|b5|>+{},<|b7|>=<|b6|>+{},<|b8|>=<|b7|>+{},<|b9|>=<|b8|>+{},<|b10|>=<|b9|>+{}",
        "num_constants": 20,
        "premise_nodes": ("<|p0|>", ),
        "target_node": "<|a10|>",
        "forward_solution": ("<|a1|>", "<|a2|>", "<|a3|>", "<|a4|>", "<|a5|>", "<|a6|>", "<|a7|>", "<|a8|>", "<|a9|>", "<|a10|>"),
        "reverse_solution": ('<|a9|>', '<|a8|>', '<|a7|>', '<|a6|>', '<|a5|>', '<|a4|>', '<|a3|>', '<|a2|>', '<|a1|>', '<|p0|>')
    }
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic math world datasets.")
    parser.add_argument("--sft_train_size", type=int, default=6400, help="Number of unique base training samples")
    parser.add_argument("--rlvr_train_size", type=int, default=1600, help="Number of unique base training samples")
    parser.add_argument("--test_size", type=int, default=1000, help="Number of unique base test samples")
    parser.add_argument("--data_dir", type=str, default="./datasets/graph", help="Output directory for parquets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    random.seed(args.seed)

    for world_config in WORLDS:
        print(f"Processing world: {world_config['id']}...")
        
        # 1. Generate ALL unique base problems first
        total_size = args.sft_train_size + args.rlvr_train_size + args.test_size
        base_problems = generate_unique_problems(total_size, world_config)

        # 2. Divide into Train and Test subsets
        sft_train_base_problems = base_problems[:args.args.sft_train_size]
        rlvr_train_base_problems = base_problems[args.args.sft_train_size:args.sft_train_size + args.rlvr_train_size]
        test_base_problems = base_problems[args.sft_train_size + args.rlvr_train_size:]

        # 3. Apply traces: Build Forward and Reverse sets independently
        sft_train_forward_samples = []
        sft_train_reverse_samples = []
        rlvr_train_samples = []
        
        for p in sft_train_base_problems:
            # Forward Trace
            fwd_trace = generate_solution_trace(p, world_config['forward_solution'], reverse=False)
            sft_train_forward_samples.append(convert_sample_to_prompt(fwd_trace, world_config['id'], 'forward'))
            
            # Reverse Trace
            rev_trace = generate_solution_trace(p, world_config['reverse_solution'], reverse=True)
            sft_train_reverse_samples.append(convert_sample_to_prompt(rev_trace, world_config['id'], 'reverse'))

            # For rlvr trainset, we only use the traces for verification
            rlvr_trace = generate_solution_trace(p, world_config['forward_solution'], reverse=False) 
            rlvr_train_samples.append(convert_sample_to_prompt(fwd_trace, world_config['id'], 'forward'))
            
        random.shuffle(sft_train_forward_samples)
        random.shuffle(sft_train_reverse_samples)
        random.shuffle(rlvr_train_samples)

        # 4. Apply traces: Test set gets ONLY the forward solution (standard eval behavior)
        test_samples = []
        for p in test_base_problems:
            fwd_trace = generate_solution_trace(p, world_config['forward_solution'], reverse=False)
            test_samples.append(convert_sample_to_prompt(fwd_trace, world_config['id'], 'forward'))

        # 5. Map Formats to HF Dataset
        raw_train_forward = datasets.Dataset.from_list(sft_train_forward_samples)
        raw_train_reverse = datasets.Dataset.from_list(sft_train_reverse_samples)
        raw_train_rlvr = datasets.Dataset.from_list(rlvr_train_samples)
        raw_test = datasets.Dataset.from_list(test_samples)

        processed_train_forward = raw_train_forward.map(function=make_map_fn("train_forward"), with_indices=True)
        processed_train_reverse = raw_train_reverse.map(function=make_map_fn("train_reverse"), with_indices=True)
        processed_train_rlvr = raw_train_rlvr.map(function=make_map_fn("rlvr"), with_indices=True)
        processed_test = raw_test.map(function=make_map_fn("test"), with_indices=True)

        # Save Locally
        output_dir = os.path.join(args.data_dir, world_config['id'])
        os.makedirs(output_dir, exist_ok=True)
        
        processed_train_forward.to_parquet(os.path.join(output_dir, "train_forward.parquet"))
        processed_train_reverse.to_parquet(os.path.join(output_dir, "train_reverse.parquet"))
        processed_train_rlvr.to_parquet(os.path.join(output_dir, "train_rlvr.parquet"))
        processed_test.to_parquet(os.path.join(output_dir, "test.parquet"))

        # Save a quick JSON example for debugging
        with open(os.path.join(output_dir, "train_example.json"), "w") as f:
            json.dump({
                "forward_example": processed_train_forward[0],
                "reverse_example": processed_train_reverse[0]
            }, f, indent=4)
        
        processed_prompt_data = []
        for idx, e in enumerate(processed_test):
            processed_prompt_data.append({
                "id": idx,
                "Question": e["question"],
                "answer": e['reward_model']['ground_truth'],
                "subset": e['data_source'],
            })
            
        with open(os.path.join(output_dir, "test.json"), "w") as f:
            json.dump(processed_prompt_data, f, indent=4)
        
    print(f"Pipeline Complete. Generated {args.train_size} Forward train samples, {args.train_size} Reverse train samples, and {args.test_size} test samples.")