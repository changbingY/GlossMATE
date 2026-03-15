import os
from pathlib import Path
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

# === CONFIGURATION ===
input_root = Path(".../gloss_multitask/LLM_grammar/Sigmorphon/")
output_root = Path("Qwen2.5-7B-Sigmorphon_result_incontext_distinctive_4/")
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

import re
from typing import List, Dict, Set, Tuple
import random
from collections import Counter, defaultdict

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

def get_top_distinctive_morphemes(train_data: List[Dict[str, str]], 
                                 top_k: int = 20,
                                 min_freq: int = 5,
                                 max_freq_ratio: float = 0.8) -> Set[str]:
    """
    Automatically discover the most distinctive morphemes from training data.
    Uses TF-IDF inspired scoring to find morphemes that are informative but not too common.
    """
    all_morphemes = []
    doc_morphemes = []  # List of sets, one per document
    
    for item in train_data:
        morphemes = tokenize_gloss(item["gloss"])
        all_morphemes.extend(morphemes)
        doc_morphemes.append(set(morphemes))  # Unique morphemes per document
    
    morpheme_counts = Counter(all_morphemes)
    total_examples = len(train_data)
    total_morphemes = len(all_morphemes)
    
    # Calculate document frequency for each morpheme
    doc_frequencies = Counter()
    for doc_morphs in doc_morphemes:
        for morph in doc_morphs:
            doc_frequencies[morph] += 1
    
    print(f"Total morphemes: {total_morphemes}, Total examples: {total_examples}")
    print(f"Unique morphemes: {len(morpheme_counts)}")
    
    # Filter and score morphemes
    candidates = []
    for morpheme, total_count in morpheme_counts.items():
        doc_freq = doc_frequencies[morpheme]  # How many documents contain this morpheme
        
        # Filter conditions
        if (total_count >= min_freq and 
            doc_freq < total_examples * max_freq_ratio and
            doc_freq >= 2):  # Must appear in at least 2 documents
            
            # TF-IDF inspired scoring
            # Term Frequency: how often it appears when it does appear
            tf = total_count / total_morphemes
            
            # Inverse Document Frequency: rarity across documents
            idf = total_examples / doc_freq
            
            # Combined distinctiveness score
            distinctiveness = tf * idf
            
            candidates.append((morpheme, distinctiveness, total_count, doc_freq))
    
    # Sort by distinctiveness score (highest first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {min(top_k, len(candidates))} distinctive morphemes found:")
    print(f"{'Rank':<4} {'Morpheme':<12} {'Score':<8} {'TotalFreq':<9} {'DocFreq':<7} {'DocFreq%':<8}")
    print("-" * 60)
    
    selected_morphemes = set()
    for i, (morpheme, score, total_freq, doc_freq) in enumerate(candidates[:top_k]):
        doc_freq_pct = (doc_freq / total_examples) * 100
        print(f"{i+1:2d}.  {morpheme:<12} {score:8.3f} {total_freq:9d} {doc_freq:7d} {doc_freq_pct:7.1f}%")
        selected_morphemes.add(morpheme)
    
    return selected_morphemes
def compute_morpheme_distinctiveness(train_data: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Compute distinctiveness scores for morphemes based on inverse frequency.
    Less frequent morphemes are more distinctive.
    """
    all_morphemes = []
    for item in train_data:
        all_morphemes.extend(tokenize_gloss(item["gloss"]))
    
    morpheme_counts = Counter(all_morphemes)
    total_count = sum(morpheme_counts.values())
    
    # Inverse frequency as distinctiveness score
    distinctiveness = {}
    for morpheme, count in morpheme_counts.items():
        distinctiveness[morpheme] = 1.0 / (count / total_count)
    
    return distinctiveness

def get_morpheme_diversity_score(examples: List[Dict[str, str]]) -> float:
    """
    Calculate diversity score based on unique morphemes across examples.
    """
    all_morphemes = set()
    for ex in examples:
        all_morphemes.update(tokenize_gloss(ex["gloss"]))
    return len(all_morphemes)

def select_distinctive_examples(test_item: Dict[str, str], 
                                train_data: List[Dict[str, str]],
                                distinctive_morphemes: Set[str],
                                morpheme_distinctiveness: Dict[str, float],
                                k: int = 3) -> List[Dict[str, str]]:
    """
    Select k examples that best demonstrate distinctive morphemes with variety.
    Strategy:
    1. Find examples with shared distinctive morphemes
    2. Rank by distinctiveness score of shared morphemes
    3. Ensure diversity in morpheme combinations
    """
    test_tokens = set(tokenize_gloss(test_item["gloss"]))
    test_distinctives = test_tokens & distinctive_morphemes

    if not test_distinctives:
        # Fallback: select examples with highest overall distinctiveness
        candidates = []
        for train_item in train_data:
            train_tokens = set(tokenize_gloss(train_item["gloss"]))
            distinctive_score = sum(morpheme_distinctiveness.get(token, 0) 
                                  for token in train_tokens if token in distinctive_morphemes)
            if distinctive_score > 0:
                candidates.append((train_item, distinctive_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in candidates[:k]]

    # Group candidates by shared distinctive morphemes
    morpheme_to_examples = defaultdict(list)
    example_scores = {}
    
    for train_item in train_data:
        train_tokens = set(tokenize_gloss(train_item["gloss"]))
        shared_distinctives = test_distinctives & train_tokens
        
        if shared_distinctives:
            # Calculate distinctiveness score for this example
            score = sum(morpheme_distinctiveness.get(morph, 0) for morph in shared_distinctives)
            example_scores[id(train_item)] = score
            
            # Group by the most distinctive shared morpheme
            most_distinctive = max(shared_distinctives, 
                                 key=lambda x: morpheme_distinctiveness.get(x, 0))
            morpheme_to_examples[most_distinctive].append(train_item)

    # Select examples to maximize diversity
    selected = []
    used_morpheme_groups = set()
    
    # Sort morphemes by distinctiveness (most distinctive first)
    sorted_morphemes = sorted(morpheme_to_examples.keys(), 
                            key=lambda x: morpheme_distinctiveness.get(x, 0), 
                            reverse=True)
    
    # First pass: select one example from each distinctive morpheme group
    for morpheme in sorted_morphemes:
        if len(selected) >= k:
            break
        
        candidates = morpheme_to_examples[morpheme]
        # Select the highest scoring example from this group
        best_candidate = max(candidates, key=lambda x: example_scores[id(x)])
        selected.append(best_candidate)
        used_morpheme_groups.add(morpheme)
    
    # Second pass: fill remaining slots with diverse examples
    if len(selected) < k:
        remaining_candidates = []
        for train_item in train_data:
            if train_item not in selected:
                train_tokens = set(tokenize_gloss(train_item["gloss"]))
                shared_distinctives = test_distinctives & train_tokens
                if shared_distinctives:
                    remaining_candidates.append(train_item)
        
        # Sort remaining by diversity they would add
        while len(selected) < k and remaining_candidates:
            best_candidate = None
            best_diversity_gain = -1
            
            for candidate in remaining_candidates:
                # Calculate diversity gain if we add this candidate
                temp_selected = selected + [candidate]
                diversity_score = get_morpheme_diversity_score(temp_selected)
                
                if diversity_score > best_diversity_gain:
                    best_diversity_gain = diversity_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining_candidates.remove(best_candidate)
            else:
                break

    # Pad with random examples if still needed
    if len(selected) < k:
        remaining = [ex for ex in train_data if ex not in selected]
        if remaining:
            selected += random.sample(remaining, min(k - len(selected), len(remaining)))

    return selected[:k]

def analyze_morpheme_patterns(examples: List[Dict[str, str]], 
                            distinctive_morphemes: Set[str]) -> Dict[str, int]:
    """
    Analyze patterns of distinctive morphemes in the selected examples.
    """
    pattern_counts = Counter()
    
    for ex in examples:
        tokens = set(tokenize_gloss(ex["gloss"]))
        distinctive_in_ex = tokens & distinctive_morphemes
        for morpheme in distinctive_in_ex:
            pattern_counts[morpheme] += 1
    
    return dict(pattern_counts)

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

    
    # Load and preprocess data
    test_data = parse_igt_file(input_path)
    train_path = Path(str(input_path).replace("test", "train"))
    train_data = parse_igt_file(train_path)
    print(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples")

    # Compute morpheme stats
    distinctive_morphemes = get_top_distinctive_morphemes(train_data, top_k=50)
    morpheme_distinctiveness = compute_morpheme_distinctiveness(train_data)

    # For each test example, construct prompt with distinctive in-context examples
    for idx, test_example in enumerate(test_data):
        print(f"\n{'='*60}")
        print(f"Test Example {idx+1}")
        print(f"{'='*60}")
        gloss_line = test_example["gloss"]
        translation_line = test_example["translation"]
        test_tokens = tokenize_gloss(gloss_line)
        test_distinctives = set(test_tokens) & distinctive_morphemes

        if not test_distinctives:
            print("No distinctive morphemes found in test example. Skipping.")
            continue

        # Select ICL examples
        icl_examples = select_distinctive_examples(
            test_example, train_data, distinctive_morphemes,
            morpheme_distinctiveness, k=4
        )

        # Format ICL block
        in_context_examples = ""
        for j, ex in enumerate(icl_examples):
            ex_gloss = ex["gloss"]
            ex_translation = ex.get("translation", "N/A")
            in_context_examples += (
                f"Example {j+1}:\n"
                f"Glossed sentence: {ex_gloss}\n"
                f"Translation: {ex_translation}\n"
            )

        # Final prompt
        prompt = (
            f"As a linguistic expert in {lang}, your task is to translate the following interlinear gloss into natural English.\n"
            f"{in_context_examples}"
            f"Glossed sentence: {gloss_line}\n"
            f"Please return only the English translation of the sentence. Do not say anything else."
        )
        
        print(prompt)
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

