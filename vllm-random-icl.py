import os
from pathlib import Path
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

# === CONFIGURATION ===
input_root = Path(".../gloss_multitask/LLM_grammar/Sigmorphon/")
output_root = Path("Qwen2.5-7B-Sigmorphon_result_incontext_random_4/")
gloss_dict = {}
dict_gloss_lines = open('.../gloss_multitask/LLM_grammar/Sigmorphon_dict_gloss/lezgi_gloss_abbreviations.txt').readlines()
for line in dict_gloss_lines:
    parts = line.strip().split('\t')
    if len(parts) == 2:
        gloss_dict[parts[0]] = parts[1]

languages = ["Lezgi"]
settings = [
    "baseline",
    # "grammatical_lcs",
    # "grammatical_replace_question",
    # "random_drop_keep1",
    # "random_lcs",
    # "random_question"
]

import re
import pandas as pd
from typing import List, Dict, Set, Tuple
def parse_igt_file(path: str) -> List[Dict[str, str]]:
    """
    Parse an IGT file into a list of examples (dicts with 'surface', 'morphemes', 'gloss', 'translation').
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    examples = []
    current = {}
    for line in lines:
        if not line:
            if current:
                examples.append(current)
                current = {}
            continue
        if line.startswith("\\t"):
            current['surface'] = line[2:].strip()
        elif line.startswith("\\m"):
            current['morphemes'] = line[2:].strip()
        elif line.startswith("\\g"):
            current['gloss'] = line[2:].strip()
        elif line.startswith("\\l"):
            current['translation'] = line[2:].strip()
    if current:
        examples.append(current)
    return examples

def tokenize_gloss(gloss_line: str) -> List[str]:
    # Handle multiple separators and clean up tokens
    tokens = re.split(r'[-\s]+', gloss_line)
    # Remove punctuation and empty tokens
    cleaned_tokens = []
    for token in tokens:
        cleaned = re.sub(r'[^\w]', '', token).strip()
        if cleaned.isupper() and not cleaned.isdigit():  # Skip empty tokens and pure numbers
            cleaned_tokens.append(cleaned)
    return cleaned_tokens

def extract_gloss_and_trans(file_path):
    glosses = []
    translations = []

    with open(file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("\\t") and i + 3 < len(lines):
                gloss = lines[i + 2][2:].strip()
                trans = lines[i + 3][2:].strip()
                glosses.append(gloss)
                translations.append(trans)
                i += 4
            else:
                i += 1

    return glosses, translations

def tokenize_uppercase_morphemes(gloss_line):
    """
    Extract only grammatical (uppercase) morphemes from a gloss line.
    """
    tokens = re.split(r'[-\s]+', gloss_line)
    return {
        re.sub(r'[^\w]', '', token)
        for token in tokens
        if token.isupper() and not token.isdigit()
    }

def find_top2_overlapping_glosses(eval_glosses, train_glosses, train_translations):
    top_matches = []
    for eval_gloss in eval_glosses:
        eval_grams = tokenize_uppercase_morphemes(eval_gloss)
        overlaps = []

        for gloss, trans in zip(train_glosses, train_translations):
            train_grams = tokenize_uppercase_morphemes(gloss)
            overlap = len(eval_grams & train_grams)
            overlaps.append((overlap, gloss, trans))

        # Sort by overlap (descending), take top 2
        top5 = sorted(overlaps, key=lambda x: -x[0])[:5]
        top_matches.append(top5)
    return top_matches


from pathlib import Path
import os

import random
from pathlib import Path
import os

def translate_gloss_file_vllm_random(llm, sampling_params, input_path: Path, output_path: Path, lang: str, k: int = 5):
    prompts = []
    metadata = []

    # Load test and inferred train data
    test_data = parse_igt_file(input_path)
    train_path = Path(str(input_path).replace("test", "train"))
    train_data = parse_igt_file(train_path)

    print(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples from:")
    print(f"  Train: {train_path}")
    print(f"  Test : {input_path}")

    for idx, test_example in enumerate(test_data):
        gloss_line = test_example["gloss"]
        translation_line = test_example["translation"]

        # Randomly select k in-context examples
        icl_examples = random.sample(train_data, min(k, len(train_data)))

        # Build in-context block
        in_context_examples = ""
        for j, ex in enumerate(icl_examples):
            in_context_examples += (
                f"Example {j+1}:\n"
                f"Glossed sentence: {ex['gloss']}\n"
                f"Translation: {ex.get('translation', 'N/A')}\n"
            )

        # Final prompt
        prompt = (
            f"As a linguistic expert in {lang}, your task is to translate the following interlinear gloss into natural English.\n"
            f"{in_context_examples}"
            f"Glossed sentence: {gloss_line}\n"
            f"Please return only the English translation of the sentence. Do not say anything else."
        )

        prompts.append(prompt)
        metadata.append((idx, prompt, translation_line))

    if not prompts:
        print(f"⚠️ No usable gloss lines found in {input_path.name}")
        return

    # Generate model outputs
    outputs = llm.generate(prompts, sampling_params)

    # Save output
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for output, (idx, prompt, gold) in zip(outputs, metadata):
            generated_text = output.outputs[0].text.strip()
            f_out.write(f"Prompt #{idx}:\n{prompt}\n")
            f_out.write(f"Gold Translation #{idx}: {gold}\n")
            f_out.write(f"Model Output #{idx}: {generated_text}\n\n")
            print(f"✓ Finished Block {idx} in {input_path.name}")

# === MAIN PROGRAM ===
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    # Load model
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=1024,
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
        stop=None
    )

    for lang in languages:
        for setting in settings:
            input_dir = input_root / lang / setting
            output_dir = output_root / lang / setting

            files = list(input_dir.glob("*-test-track2-uncovered"))
            print(f"🔍 Processing {len(files)} files in {lang}/{setting}...")
            for file in tqdm(files, desc=f"{lang}/{setting}", unit="file"):
                output_file = output_dir / file.name.replace(".txt", "_LLM_output.txt")
                translate_gloss_file_vllm_random(llm, sampling_params, file, output_file, lang,k=4)
