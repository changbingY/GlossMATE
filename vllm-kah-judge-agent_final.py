import json
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
import re

import pandas as pd
explain_file = Path("../gloss_multitask/LLM_grammar/Kah/gloss_explanations.txt")

# Load CSV and clean up headers
df = pd.read_csv("../gloss_multitask/LLM_grammar/Kah/kane.csv", encoding="utf-8-sig")
df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True).str.replace('\u200b', '')

# === File Paths ===
kah_file = Path("../gloss_multitask/LLM_grammar/Kah/kah_gloss2en.json")
pronouns_file = Path("../gloss_multitask/LLM_grammar/Kah/pronouns.json")
options_file = Path("../gloss_multitask/LLM_grammar/Kah/new_options.json")
affixes_file = Path("../gloss_multitask/LLM_grammar/Kah/affixes.json")
critique_file = Path("../gloss_multitask/LLM_grammar/Qwen2.5-7B-Kah_result/Kah/baseline+abb+explain/Kanyenkeha_qwen2.5-7b_output_base+explain.txt")
output_file = Path("../gloss_multitask/LLM_grammar/Qwen2.5-7B-Kah_result/Kah/baseline+abb+explain/Kanyenkeha_qwen2.5-7b_output_corrected.txt")

# Load original translations
TRANSL_FILE = Path("../gloss_multitask/LLM_grammar/Qwen2.5-7B-Kah_result/Kah/baseline+abb/Kanyenkeha_qwen2.5-7b_output_base+abb.txt")

def read_candidate_translations(path: Path) -> list[str]:
    """Read candidate translations from file"""
    translations = []
    pattern = re.compile(r"^Qwen2\.5-7B Result:\s*(.*)$")
    
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                translations.append(m.group(1).strip())
    return translations

def parse_critique_file(path: Path) -> list[dict]:
    """Parse the critique file and extract information for each example"""
    critiques = []
    
    with path.open(encoding="utf-8") as f:
        content = f.read()
    
    # Split by the separator pattern
    sections = content.split("============================================================")
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        
        # Find the critic number
        critic_match = re.search(r"--- CRITIC (\d+) ---", section)
        if not critic_match:
            continue
            
        critic_num = int(critic_match.group(1))
        
        # Extract meta information (everything between "### Original Translation Prompt" and "### Candidate translation")
        meta_start = section.find("### Original Translation Prompt")
        meta_end = section.find("### Candidate translation")
        
        if meta_start == -1 or meta_end == -1:
            continue
            
        meta_info = section[meta_start:meta_end].replace("### Original Translation Prompt", "").strip()
        
        # Extract candidate translation
        candidate_start = section.find("### Candidate translation")
        candidate_end = section.find("### Your review")
        
        if candidate_start == -1 or candidate_end == -1:
            continue
            
        candidate_section = section[candidate_start:candidate_end]
        candidate_match = re.search(r'"([^"]*)"', candidate_section)
        candidate = candidate_match.group(1) if candidate_match else ""
        
        # Extract critique
        critique_start = section.find("### Your review")
        if critique_start == -1:
            continue
            
        critique = section[critique_start:].replace("### Your review", "").strip()
        
        critiques.append({
            'number': critic_num,
            'meta_info': meta_info,
            'candidate': candidate,
            'critique': critique
        })
    
    return critiques

translations = read_candidate_translations(TRANSL_FILE)
critiques = parse_critique_file(critique_file)

print(f"Loaded {len(translations)} candidate translations.")
print(f"Loaded {len(critiques)} critiques.")

root_dict = {}
# Define fields to include
placeholders = [
    "label", "root", "red", "blue", "purple",
    "command", "habitual", "punctual", "perf",
    "progr-end", "progr-pron-type",
    "hab-past-end", "hab-fut-end", "stative-pres-other",
    "te-pref", "t-pref", "ni-pref", "ie-pref", "s-pref",
    "translate", "eng-inf", "eng-3", "eng-prog", "eng-perf", "eng-past", "eng-alt-passive"
]

# explanation mapping
pattern_expl = {
    "d. incl." : "we two (you and I) / us two (incl. you)",
    "d. excl." : "we two (someone else and I) / us two (excl. you)",
    "d. m."    : "they two (male) / them two (male)",
    "d. f."    : "they two (female) / them two (female)",
    "d."       : "two people",
    "pl. incl.": "we all (including you) / us all (incl. you)",
    "pl. excl.": "we all (excluding you) / us all (excl. you)",
    "pl. m."   : "they (male) / them (male)",
    "pl. f."   : "they (female) / them (female)",
    "pl."      : "more than three people",
}

# build one big regex, ordering alternatives by descending length
ordered = sorted(pattern_expl.keys(), key=len, reverse=True)
big_regex = re.compile("|".join(map(re.escape, ordered)))

def explain_patterns(text: str) -> str:
    """Replace every pattern with its explanation, longest patterns first."""
    return big_regex.sub(lambda m: f"({pattern_expl[m.group(0)]})", text)

for idx, row in df.iterrows():
    # Extract values
    values = {field: str(row.get(field, "")).strip() for field in placeholders}
    root_dict[values["root"]] = values["translate"]

# === Load JSON Files ===
with open(kah_file, "r", encoding="utf-8") as f:
    kah_data = json.load(f)

with open(pronouns_file, "r", encoding="utf-8") as f:
    pronouns = json.load(f)
    pronoun_map = {p["tag"]: p["en"] for p in pronouns}

with open(options_file, "r", encoding="utf-8") as f:
    options = json.load(f)
    option_map = {opt["tag"]: ", ".join(opt["classes"]) for opt in options}

with open(affixes_file, "r", encoding="utf-8") as f:
    affixes = json.load(f)
    affix_map = {a["tag"]: a["gloss"] for a in affixes}

explain_dict = {}
with open(explain_file, "r", encoding="utf-8") as f:
    for line in f:
        if ":" in line:
            key, val = line.strip().split(":", 1)
            explain_dict[key.strip()] = val.strip()
            
# === vLLM Setup ===
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    dtype="half"
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=64,
    stop=["<|endoftext|>", "<|im_end|>"]
)

# === Qwen Chat Format ===
def format_chat_prompt(content):
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

# === Corrector Prompt Template ===
CORRECTOR_HEADER = """
You are an expert Kanyen'kéha translator. Your task is to provide a corrected English translation based on the original meta information and the critique feedback.

## Task
Given the original translation context and the critique identifying problems, provide a corrected English translation that addresses all the issues mentioned in the critique.

Please return ONLY the corrected English translation. Do not explain your reasoning or include any additional text.
""".strip()

# === Generate Corrected Translations ===
with open(output_file, "w", encoding="utf-8") as fout:
    for i, critique_data in tqdm(enumerate(critiques), total=len(critiques), desc="Generating corrected translations"):
        
        # Get the corresponding example from kah_data
        if i >= len(kah_data):
            continue
            
        ex = kah_data[i]
        output_data = ex["output"]
        
        # Get reference translation
        reference_english = ""
        for item in output_data:
            if item[1] == "" and item[4].strip():
                reference_english = item[4].strip()
                break
        
        # Get meta information and critique from parsed data
        meta_info = critique_data['meta_info']
        candidate = critique_data['candidate']
        critique = critique_data['critique']
        
        # Generate corrected translation
        corrector_prompt = (
            f"{CORRECTOR_HEADER}\n\n"
            f"### Original Meta Information\n{meta_info}\n\n"
            f'### Original Translation\n"{candidate}"\n\n'
            f"### Critique\n{critique}\n\n"
            f"### Corrected Translation\n"
        )

        full_corrector_prompt = format_chat_prompt(corrector_prompt)
        # print('Full prompt '+full_corrector_prompt)
        outputs = llm.generate([full_corrector_prompt], sampling_params)
        corrected_translation = outputs[0].outputs[0].text.strip()
        # print('model output translation '+corrected_translation)

        # Save results
        fout.write(f"--- EXAMPLE {i+1} ---\n")
        fout.write(f"Meta Information:\n{meta_info}\n\n")
        fout.write(f"Original Translation: {candidate}\n")
        fout.write(f"Expected Translation: {explain_patterns(reference_english)}\n")
        fout.write(f"Critique: {critique}\n")
        fout.write(f"Corrected Version Translation: {corrected_translation}\n")
        fout.write("=" * 60 + "\n\n")

print(f"✅ All translations corrected. Output saved to: {output_file}")
