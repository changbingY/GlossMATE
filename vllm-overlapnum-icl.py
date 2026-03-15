import os
from pathlib import Path
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

# === CONFIGURATION ===
input_root = Path(".../gloss_multitask/LLM_grammar/Sigmorphon/")
output_root = Path("Qwen2.5-7B-Sigmorphon_result_incontext_overlapnum/")
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

def translate_gloss_file_vllm(llm, sampling_params, input_path: Path, output_path: Path, lang: str):
    prompts = []
    metadata = []

    eval_glosses, eval_translations = extract_gloss_and_trans(input_path)
    train_glosses, train_translations = extract_gloss_and_trans(
        Path(str(input_path).replace("test", "train"))
    )

    results = find_top2_overlapping_glosses(eval_glosses, train_glosses, train_translations)

    for idx, (gloss_line, translation_line) in enumerate(zip(eval_glosses, eval_translations)):
        top_matches = results[idx]
        
        # Step 1: Explanation block for test gloss
        gloss_tokens = gloss_line.replace('-', ' ').split()
        explanation_lines = [
            f'- "{token}" indicates {gloss_dict[token]}.'
            for token in gloss_tokens if token.isupper() and token in gloss_dict
        ]
        explanation_block = ""
        if explanation_lines:
            explanation_block = (
                "To assist you, here are explanations for some of the grammatical glosses:\n"
                + "\n".join(explanation_lines) + "\n"
            )

        # Step 2: In-context examples
        in_context_examples = ""
        for j, (overlap, ex_gloss, ex_trans) in enumerate(top_matches):
            in_context_examples += (
                f"Example {j+1}:\n"
                f"Glossed sentence: {ex_gloss}\n"
                f"Translation: {ex_trans}\n"
            )

        # Step 3: Final prompt
        prompt = (
            f"As a linguistic expert in {lang}, your task is to translate the following interlinear gloss into natural English.\n"
            f"{in_context_examples}"
            f"Glossed sentence: {gloss_line}\n"
            f"{explanation_block}"
            f"Please return only the English translation of the sentence. Do not say anything else."
        )

        prompts.append(prompt)
        metadata.append((idx, prompt, translation_line))

    if not prompts:
        print(f"⚠️ No gloss lines found in {input_path.name}")
        return

    outputs = llm.generate(prompts, sampling_params)

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
                translate_gloss_file_vllm(llm, sampling_params, file, output_file, lang)
