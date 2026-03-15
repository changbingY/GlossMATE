import os
from pathlib import Path
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

# === CONFIGURATION ===
input_root = Path(".../gloss_multitask/LLM_grammar/Sigmorphon/")
output_root = Path("Qwen2.5-7B-Sigmorphon_result")

languages = [ "Gitksan","Lezgi","Natugu"]
settings = [
    "grammatical_drop_keep1",
    "grammatical_lcs",
    "grammatical_replace_question",
    "random_drop_keep1",
    "random_lcs",
    "random_question"
]

def translate_gloss_file_vllm(llm, sampling_params, input_path: Path, output_path: Path, lang: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')
    prompts = []
    metadata = []

    for idx, block in enumerate(blocks):
        lines = block.strip().split('\n')
        gloss_line = None
        translation_line = "(no reference available)"

        for line in lines:
            if line.startswith('\\g '):
                gloss_line = line[3:].strip()
            elif line.startswith('\\l '):
                translation_line = line[3:].strip()

        if gloss_line:
            prompt = (
                f"You are a linguistic expert in {lang}. "
                "Your task is to translate the given gloss into fluent English.\n"
                f"Gloss: {gloss_line}\n"
                "Please only return the English translation. Do not say anything else."
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
        max_model_len=256,
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

            files = list(input_dir.glob("*.txt"))
            print(f"🔍 Processing {len(files)} files in {lang}/{setting}...")
            for file in tqdm(files, desc=f"{lang}/{setting}", unit="file"):
                output_file = output_dir / file.name.replace(".txt", "_LLM_output.txt")
                translate_gloss_file_vllm(llm, sampling_params, file, output_file, lang)
