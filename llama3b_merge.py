"""
Stage 3 for the Llama-3.2-3B fine-tune: merge the LoRA adapter into the base
weights, repair the config for Ollama, and write a matching Modelfile.

Three things this does that the Qwen run had to learn the hard way:

1. MERGE -- reuses merge_lora_adapter.py by overriding its globals, so the
   procedure is identical to the Qwen merge.

2. CONFIG REPAIR -- transformers v5 writes `rope_parameters` (nested) and
   `dtype`, where Ollama's safetensors->GGUF converter still expects the v4
   layout: top-level `rope_theta`, `rope_scaling`, `torch_dtype`. On Qwen
   this mismatch silently produced a model that emitted only "@@@@@" and
   cost hours to diagnose. Llama-3.2 is worse: its rope_scaling carries four
   extra fields (factor, low/high_freq_factor,
   original_max_position_embeddings) which, if lost, break long-context
   positional encoding. This hoists them back to the top level WITHOUT
   removing anything, so transformers still loads the folder too.

3. MODELFILE -- written automatically with the stop token read from the
   tokenizer that was actually trained with. Llama-3.2-Instruct ends turns
   with <|eot_id|>; hard-coding Qwen's <|im_end|> would make every answer run
   to the token cap. The template is raw passthrough, matching how
   qlora_finetune.py tokenizes (raw prompt, add_special_tokens=False) --
   eval_raw_template.py showed format makes no measurable difference, but
   matching it costs nothing.

Run from the project root, after llama3b_finetune.py finishes:
    python llama3b_merge.py

Then:
    ollama create llama3.2-3b-medqa-lora --quantize q4_K_M -f Modelfile_llama3b
    python eval_llama3b.py
"""

import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from transformers import AutoTokenizer

import merge_lora_adapter as m


ADAPTER_DIR = "qlora_output/llama3.2-3b-medqa-lora"
MERGED_DIR = "qlora_output/llama3.2-3b-medqa-merged"
MODELFILE = "Modelfile_llama3b"

CANDIDATE_REPOS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
    "NousResearch/Llama-3.2-3B-Instruct",
]


def resolve_base():
    """Prefer the base model recorded in the adapter config, so the merge
    uses exactly the weights that were fine-tuned."""

    cfg_path = os.path.join(ADAPTER_DIR, "adapter_config.json")

    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            base = json.load(f).get("base_model_name_or_path")
        if base:
            print(f"Base model from adapter_config.json: {base}")
            return base

    for repo in CANDIDATE_REPOS:
        try:
            AutoTokenizer.from_pretrained(repo)
            print(f"Falling back to: {repo}")
            return repo
        except Exception:
            continue

    print("Could not resolve the base model. Check ADAPTER_DIR.")
    sys.exit(1)


def repair_config(merged_dir):
    """Hoist transformers-v5 nested fields back to the v4 top level that
    Ollama's converter reads. Additive only -- nothing is removed."""

    path = os.path.join(merged_dir, "config.json")

    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    changed = []

    if "torch_dtype" not in cfg and "dtype" in cfg:
        cfg["torch_dtype"] = cfg["dtype"]
        changed.append("torch_dtype")

    rp = cfg.get("rope_parameters")

    if isinstance(rp, dict):

        if "rope_theta" not in cfg and "rope_theta" in rp:
            cfg["rope_theta"] = rp["rope_theta"]
            changed.append("rope_theta")

        # Everything in rope_parameters other than rope_theta describes the
        # scaling scheme. Llama-3.2 needs factor / low_freq_factor /
        # high_freq_factor / original_max_position_embeddings preserved.
        scaling = {k: v for k, v in rp.items() if k != "rope_theta"}

        rope_type = scaling.get("rope_type", "default")

        if "rope_scaling" not in cfg or cfg.get("rope_scaling") is None:
            if rope_type in (None, "default"):
                cfg["rope_scaling"] = None
                changed.append("rope_scaling=None")
            else:
                # v4 readers expect BOTH "type" and "rope_type".
                scaling.setdefault("type", rope_type)
                cfg["rope_scaling"] = scaling
                changed.append(f"rope_scaling({rope_type})")

    # Sanity: the converter needs these or it reads a dimension as 0 and
    # fails with "tensor 'blk.0.ffn_gate.weight' has wrong shape".
    required = ["hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "vocab_size"]
    missing = [k for k in required if k not in cfg]

    if missing:
        print(f"  WARNING: config is missing {missing} -- conversion will fail.")

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"  config.json patched: added {', '.join(changed)}")
    else:
        print("  config.json already in the layout Ollama expects")

    print(f"  rope_theta   = {cfg.get('rope_theta')}")
    print(f"  rope_scaling = {cfg.get('rope_scaling')}")


def write_modelfile(merged_dir, tokenizer):

    eos = tokenizer.eos_token
    abs_dir = os.path.abspath(merged_dir)

    stops = [eos]

    # Llama-3 chat models emit <|eot_id|> at turn end; include both that and
    # <|end_of_text|> when present so generation always terminates.
    for extra in ("<|eot_id|>", "<|end_of_text|>"):
        if extra in tokenizer.get_vocab() and extra not in stops:
            stops.append(extra)

    lines = [f"FROM {abs_dir}", "", 'TEMPLATE """{{ .Prompt }}"""', ""]
    lines += [f'PARAMETER stop "{s}"' for s in stops]

    with open(MODELFILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote {MODELFILE}")
    print(f"  FROM {abs_dir}")
    print(f"  trained eos_token = {eos!r}")
    print(f"  stop parameters   = {stops}")


def main():

    base = resolve_base()

    if not os.path.isdir(ADAPTER_DIR):
        print(f"Adapter not found: {ADAPTER_DIR} -- run llama3b_finetune.py first.")
        sys.exit(1)

    m.BASE_MODEL = base
    m.ADAPTER_DIR = ADAPTER_DIR
    m.MERGED_OUTPUT_DIR = MERGED_DIR

    print(f"\nMerging {ADAPTER_DIR} into {base}")
    print(f"Output: {MERGED_DIR}\n")

    m.main()

    print("\nRepairing config.json for Ollama's converter...")
    repair_config(MERGED_DIR)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    write_modelfile(MERGED_DIR, tokenizer)

    print("\nNext:")
    print("  ollama create llama3.2-3b-medqa-lora --quantize q4_K_M "
          f"-f {MODELFILE}")
    print("  python eval_llama3b.py")


if __name__ == "__main__":
    main()
