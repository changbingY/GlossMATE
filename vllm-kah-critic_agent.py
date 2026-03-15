import json
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams

import pandas as pd
explain_file = Path("../gloss_multitask/LLM_grammar/Kah/gloss_explanations.txt")

# Load CSV and clean up headers
df = pd.read_csv("../gloss_multitask/LLM_grammar/Kah/kane.csv", encoding="utf-8-sig")
df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True).str.replace('\u200b', '')


import re
from pathlib import Path

TRANSL_FILE = Path("../gloss_multitask/LLM_grammar/Qwen2.5-7B-Kah_result/Kah/baseline+abb/Kanyenkeha_qwen2.5-7b_output_base+abb.txt")

def read_candidate_translations(path: Path) -> list[str]:
    """
    Return:
        translations[idx]  ->  string with the model’s translation for example idx
    Assumes each example in the file contains a line starting with
        'Qwen2.5-7B Result: '
    """
    translations = []
    pattern = re.compile(r"^Qwen2\.5-7B Result:\s*(.*)$")
    
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                translations.append(m.group(1).strip())
    return translations

translations = read_candidate_translations(TRANSL_FILE)
print(f"Loaded {len(translations)} candidate translations.")

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

# pattern_explanations = {
#     # dual (exactly two people)
#     "d. incl." : "1st-person dual **inclusive** → “we two (you and I)” / “us two (incl. you)”",
#     "d. excl." : "1st-person dual **exclusive** → “we two (someone else and I, not you)” / “us two (excl. you)”",
#     "d."       : "Dual number (two people) with no gender/inclusivity info – e.g. “you two” (2nd pers.) or “they two” (3rd pers.)",
#     "d. m."    : "3rd-person dual **masculine** → “they two (male)” / “them two (male)”",
#     "d. f."    : "3rd-person dual **feminine** → “they two (female)” / “them two (female)”",

#     # plural (three or more people)
#     "pl. incl.": "1st-person plural **inclusive** → “we all (including you)” / “us all (incl. you)”",
#     "pl. excl.": "1st-person plural **exclusive** → “we all (excluding you)” / “us all (excl. you)”",
#     "pl."      : "Plural number with no gender/inclusivity info – e.g. “you all” (2nd pers.) or “they” (3rd pers.)",
#     "pl. m."   : "3rd-person plural **masculine** → “they (male)” / “them (male)”",
#     "pl. f."   : "3rd-person plural **feminine** → “they (female)” / “them (female)”"
# }


import re

# explanation mapping (same as before)
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

color_meanings = {
    "b": "passive",
    "p": "transitive", 
    "r": "active"
}
      

# === File Paths ===
kah_file = Path("../gloss_multitask/LLM_grammar/Kah/kah_gloss2en.json")
pronouns_file = Path("../gloss_multitask/LLM_grammar/Kah/pronouns.json")
options_file = Path("../gloss_multitask/LLM_grammar/Kah/new_options.json")
affixes_file = Path("../gloss_multitask/LLM_grammar/Kah/affixes.json")
output_file = Path("../gloss_multitask/LLM_grammar/Qwen2.5-7B-Kah_result/Kah/baseline+abb+explain/Kanyenkeha_qwen2.5-7b_output_base+explain.txt")

# === Load Files ===
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

# === Prompt Template ===
prompt_template = """
You are an expert in translating Kanyen’kéha verb forms into natural English. Based on the root verb, participant roles (agent/patient), grammatical information, and morphological structure, translate the form into fluent English.
Provided Kanyen’kéha gloss: {morphs}
Here are some additional information:
- Verb root translation: {translation}
- Verb class: {color_meaning}
- Agent: {agent} ({agent_en})  {agent_expl}
- Patient: {patient} ({patient_en})  {patient_expl}
- Grammatical Context: {option_tag}
  {option_expl}
Morphological Breakdown:
{morphs_str_breakdown}
Please return a fluent English translation only. Do not explain your reasoning.
""".strip()


# 1️⃣  Put this near the top, after imports
CRITIC_HEADER = """
You are a meticulous translation reviewer.

## Task
Identify **all** problems in the candidate English translation below,
given the full Kanyen’kéha prompt that the translator saw.
For each problem:
- Quote or paraphrase the erroneous span.
- Explain why it is wrong w.r.t. the gloss and meta-information.
If the translation is perfect, output exactly: NO ISSUE
""".strip()



# === Generate and Save Results ===
with open(output_file, "w", encoding="utf-8") as fout:
    for i, ex in tqdm(enumerate(kah_data), total=len(kah_data), desc="Running vLLM"):
        input_data = ex["input"]
        output_data = ex["output"]

        root = input_data.get("root", "")
        root_base = root.split('-')[0]
        color = root.split('-')[1] if '-' in root else ""
        translation = root_dict.get(root_base, "")
        agent = input_data.get("agent", "")
        patient = input_data.get("patient", "")
        option = input_data.get("option", "")

        if not translation:
            continue  # skip if we don't have root translation

        # Get color meaning
        color_meaning = color_meanings.get(color, f"unknown color ({color})")
        
        # Get readable tags
        agent_en = pronoun_map.get(agent, {}).get("agent", agent)
        patient_en = pronoun_map.get(patient, {}).get("patient", patient)
        option_tag = option_map.get(option, option)
        
        agent_expl = f"→ {explain_dict[agent]}" if agent in explain_dict else ""
        patient_expl = f"→ {explain_dict[patient]}" if patient in explain_dict else ""

        # Split compound tags like "HAB,FUT" and explain each
        option_parts = option_tag.split(",")
        option_expl_lines = []
        for part in option_parts:
            part = part.strip()
            if part:
                expl = explain_dict.get(part, "")
                if expl:
                   option_expl_lines.append(f"  - {part}: {expl}")
        option_expl = "\n".join(option_expl_lines)


        # Build morpheme breakdown and gloss line
        morphemes = []
        glosses = []
        morph_lines = []

        for item in output_data:
            if item[1] == "morph":
                tag = " > ".join(item[2]) if item[2] else ""
                morph = item[2][-1] if item[2] else ""
                gloss = item[3].strip()
                if gloss:
                    morphemes.append(morph)
                    glosses.append(gloss)
                    explanation = explain_dict.get(gloss, "")
                    if explanation:
                       morph_lines.append(f"- {tag}: {gloss}  ({explanation})")
                    else:
                       morph_lines.append(f"- {tag}: {gloss}")

            elif item[1] == "" and item[4].strip():
                reference_english = item[4].strip()

        if not glosses:
            continue

        morphs_str = "-".join(glosses)
        morphs_str_breakdown = "\n".join(morph_lines)

        # Fill the template and get model output
        prompt_text = prompt_template.format(
                    morphs=morphs_str,
                    translation=translation,
                    color_meaning=color_meaning,
                    agent=agent,
                    agent_en=agent_en,
                    agent_expl=agent_expl,
                    patient=patient,
                    patient_en=patient_en,
                    patient_expl=patient_expl,
                    option_tag=option_tag,
                    option_expl=option_expl,
                    morphs_str_breakdown=morphs_str_breakdown)


        # full_prompt = format_chat_prompt(prompt_text)
        # outputs = llm.generate([full_prompt], sampling_params)
        # result = outputs[0].outputs[0].text.strip()
        

        # 2️⃣  Inside your FOR-LOOP that iterates over kah_data
        #     (right after you finish computing `prompt_text`)
        # -----------------------------------------------------
        full_translator_prompt = prompt_text               # keep the whole thing
        candidate          = translations[i]               # string from earlier file

        critic_prompt = (
                         f"{CRITIC_HEADER}\n\n"
                         f"### Meta Information\n{full_translator_prompt}\n\n"
                         f'### Candidate translation\n"{candidate}"\n\n'
                         f"### Your review\n"
                        )

        # Wrap for Qwen chat format
        full_prompt = format_chat_prompt(critic_prompt)

        # Now generate the critique
        outputs   = llm.generate([full_prompt], sampling_params)
        critique  = outputs[0].outputs[0].text.strip()

        # 3️⃣  Save or print as you like
        fout.write(f"--- CRITIC {i+1} ---\n")
        fout.write(critic_prompt + critique + "\n")
        fout.write("="*60 + "\n\n")
        # # Save result
        # fout.write(f"--- PROMPT {i + 1} ---\n")
        # fout.write(prompt_text + "\n")
        # fout.write(f"Expected Translation: {explain_patterns(reference_english)}\n")
        # fout.write(f"Qwen2.5-7B Result: {result}\n")
        # fout.write("=" * 50 + "\n\n")

print(f"✅ All prompts processed. Output saved to: {output_file}")
