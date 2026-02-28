"""
VayuSwarm -- Drone LLM Fine-tuning (Kaggle Notebook)
================================================
Fine-tunes Phi-3-mini-4k-instruct on the VayuSwarm drone sensor
decision dataset using QLoRA (4-bit quantization + LoRA adapters).

DATASET (pulled from GitHub automatically -- no upload needed):
  ved354/swam -> models/llm_data/llm_training_data.jsonl
  5,000 samples: (drone_state, battery, detections) -> action JSON

OUTPUT (auto-pushed to GitHub after training):
  ved354/swam -> models/drone_llm/
    adapter_config.json, adapter_model.safetensors, tokenizer*, Modelfile

KAGGLE SETUP:
  1. Runtime: GPU T4 x2 (or P100/A100)
  2. Add Kaggle Secret named: GITHUB_TOKEN  (with repo write access)
  3. Expected runtime: ~30-45 min on T4
"""

import subprocess
subprocess.check_call(["pip", "install", "-q",
    "transformers>=4.40.0", "datasets>=2.18.0", "peft>=0.10.0",
    "trl>=0.8.6", "bitsandbytes>=0.43.1", "accelerate>=0.29.0",
    "gitpython", "scipy",
])

import json, shutil, os
from pathlib import Path
import torch
import urllib.request

# ==============================================================
# Config
# ==============================================================

GITHUB_TOKEN  = ""            # Set via Kaggle Secret -- do NOT hardcode
GITHUB_REPO   = "ved354/swam"
GITHUB_BRANCH = "main"

BASE_MODEL   = "microsoft/Phi-3-mini-4k-instruct"   # 3.8B -- fits on T4
OUTPUT_DIR   = Path("/kaggle/working/vayuswarm_llm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS       = 3
BATCH_SIZE   = 2
GRAD_ACCUM   = 8       # Effective batch = 16
LR           = 2e-4
MAX_SEQ_LEN  = 512
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

# Load GITHUB_TOKEN from Kaggle Secrets (preferred) or env
try:
    from kaggle_secrets import UserSecretsClient
    GITHUB_TOKEN = GITHUB_TOKEN or UserSecretsClient().get_secret("GITHUB_TOKEN")
    if GITHUB_TOKEN:
        print("GITHUB_TOKEN loaded from Kaggle Secrets")
    else:
        print("WARNING: Kaggle Secret GITHUB_TOKEN is empty -- check your secret value")
except Exception as _secret_err:
    print(f"Kaggle Secrets unavailable ({_secret_err}), trying environment...")
    GITHUB_TOKEN = GITHUB_TOKEN or os.environ.get("GITHUB_TOKEN", "")
    print("GITHUB_TOKEN from env" if GITHUB_TOKEN else "WARNING: no GITHUB_TOKEN -- auto-push will be skipped")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | GPUs: {torch.cuda.device_count()} | Model: {BASE_MODEL}")

# ==============================================================
# Download Dataset from GitHub
# ==============================================================

DATA_URL  = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/models/llm_data/llm_training_data.jsonl"
DATA_PATH = Path("/kaggle/working/llm_training_data.jsonl")

print("\nDownloading dataset from GitHub...")
urllib.request.urlretrieve(DATA_URL, str(DATA_PATH))

samples = []
with open(DATA_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))

print(f"Loaded {len(samples)} training samples")
print(f"  Input sample:  {samples[0]['input'][:100]}")
print(f"  Output sample: {samples[0]['output'][:100]}")

# ==============================================================
# Format for Phi-3 Chat Template
# ==============================================================

from datasets import Dataset

def format_phi3(sample):
    sys_tok  = "<" + "|system|" + ">"
    end_tok  = "<" + "|end|" + ">"
    user_tok = "<" + "|user|" + ">"
    asst_tok = "<" + "|assistant|" + ">"
    eos_tok  = "<" + "|endoftext|" + ">"
    return {
        "text": (
            f"{sys_tok}\n{sample['instruction']}{end_tok}\n"
            f"{user_tok}\n{sample['input']}{end_tok}\n"
            f"{asst_tok}\n{sample['output']}{end_tok}\n{eos_tok}"
        )
    }

ds = Dataset.from_list(samples).map(format_phi3, remove_columns=["instruction", "input", "output"])
print(f"\nDataset ready: {len(ds)} samples")
print(f"  Example: {ds[0]['text'][:200]}")

# ==============================================================
# Load Model + Tokenizer (4-bit QLoRA)
# ==============================================================

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token     = tokenizer.eos_token
tokenizer.padding_side  = "right"

print("Loading model in 4-bit...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Fix: Phi-3-mini-4k-instruct uses standard RoPE (no scaling).
# The cached modeling_phi3.py on Kaggle's environment has a bug where it
# reads rope_scaling even when it shouldn't, and the config is missing keys.
# Setting rope_scaling=None forces standard RoPE and bypasses all of this.
from transformers import AutoConfig
model_cfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
if hasattr(model_cfg, "rope_scaling") and model_cfg.rope_scaling is not None:
    print(f"  rope_scaling found: {list(model_cfg.rope_scaling.keys())} -- setting to None (standard RoPE)")
    model_cfg.rope_scaling = None

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    config=model_cfg,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
print("Model loaded")

# ==============================================================
# LoRA Config
# ==============================================================

lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ==============================================================
# Train
# ==============================================================

import inspect as _inspect
import trl as _trl

_trl_version = tuple(int(x) for x in _trl.__version__.split(".")[:2])
print(f"trl version: {_trl.__version__}")

# Build SFTConfig args — adapt to whatever trl version Kaggle has
_sft_kwargs = dict(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none",
)

# warmup_ratio vs warmup_steps
_sft_sig = _inspect.signature(SFTConfig.__init__)
if "warmup_steps" in _sft_sig.parameters:
    _sft_kwargs["warmup_steps"] = 30
else:
    _sft_kwargs["warmup_ratio"] = 0.03

# max_seq_length — some versions want it in SFTConfig, others in SFTTrainer
if "max_seq_length" in _sft_sig.parameters:
    _sft_kwargs["max_seq_length"] = MAX_SEQ_LEN
if "dataset_text_field" in _sft_sig.parameters:
    _sft_kwargs["dataset_text_field"] = "text"

sft_cfg = SFTConfig(**_sft_kwargs)

# Build SFTTrainer args
_trainer_sig = _inspect.signature(SFTTrainer.__init__)
_trainer_kwargs = dict(
    model=model,
    train_dataset=ds,
    args=sft_cfg,
)

# tokenizer vs processing_class
if "processing_class" in _trainer_sig.parameters:
    _trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in _trainer_sig.parameters:
    _trainer_kwargs["tokenizer"] = tokenizer

# max_seq_length in trainer (older trl)
if "max_seq_length" in _trainer_sig.parameters and "max_seq_length" not in _sft_kwargs:
    _trainer_kwargs["max_seq_length"] = MAX_SEQ_LEN
if "dataset_text_field" in _trainer_sig.parameters and "dataset_text_field" not in _sft_kwargs:
    _trainer_kwargs["dataset_text_field"] = "text"

trainer = SFTTrainer(**_trainer_kwargs)


print("\nStarting training...")
trainer.train()
print("Training complete!")

# Save adapter + tokenizer
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"Model saved to {OUTPUT_DIR}")

# ==============================================================
# Create Modelfile for Ollama
# ==============================================================

MODELFILE = """FROM phi3:3.8b
ADAPTER ./adapter_model.safetensors

SYSTEM You are the tactical AI for a VayuSwarm autonomous surveillance drone. You receive structured sensor data (detections, battery, state) and output tactical decisions in JSON format. Never engage targets. Surveillance only. Prioritize safety.

PARAMETER temperature 0.2
PARAMETER num_predict 256
"""

(OUTPUT_DIR / "Modelfile").write_text(MODELFILE)
print("Modelfile written for Ollama")

# ==============================================================
# Auto-Push to GitHub
# ==============================================================

if GITHUB_TOKEN and GITHUB_REPO:
    print(f"\nPushing to https://github.com/{GITHUB_REPO}/tree/{GITHUB_BRANCH}/models/drone_llm ...")
    try:
        import git
        clone_dir = Path("/kaggle/working/repo_clone")
        if clone_dir.exists():
            shutil.rmtree(clone_dir)

        repo_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
        try:
            repo = git.Repo.clone_from(repo_url, str(clone_dir), branch=GITHUB_BRANCH)
        except Exception:
            repo = git.Repo.clone_from(repo_url, str(clone_dir))
            repo.git.checkout("-b", GITHUB_BRANCH)

        # Copy trained model files (adapters + tokenizer + Modelfile)
        target = clone_dir / "models" / "drone_llm"
        target.mkdir(parents=True, exist_ok=True)

        for f in OUTPUT_DIR.iterdir():
            if f.is_file():
                shutil.copy2(str(f), str(target / f.name))
                print(f"  Copied: {f.name}")

        repo.config_writer().set_value("user", "name",  "VayuSwarm-Bot").release()
        repo.config_writer().set_value("user", "email", "bot@vayuswarm.ai").release()
        repo.git.add(A=True)

        if repo.is_dirty() or repo.untracked_files:
            repo.index.commit("Add fine-tuned VayuSwarm LLM (Phi-3 QLoRA adapters)")
            repo.remote("origin").push(GITHUB_BRANCH)
            print(f"\nPushed to: https://github.com/{GITHUB_REPO}/tree/{GITHUB_BRANCH}/models/drone_llm")
        else:
            print("No changes to push (already up to date).")

        shutil.rmtree(clone_dir, ignore_errors=True)

    except Exception as e:
        print(f"Push failed: {e}")
        print(f"Model saved locally at: {OUTPUT_DIR}")
else:
    print("No GITHUB_TOKEN -- model saved locally only.")
    print(f"Output: {OUTPUT_DIR}")

print("\nDone! To use with Ollama on the drone:")
print("  ollama create vayuswarm -f models/drone_llm/Modelfile")
print("  ollama run vayuswarm")
