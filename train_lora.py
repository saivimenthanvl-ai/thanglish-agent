"""LoRA fine-tune a public causal LM on the Thanglish -> Tamil dataset.

Usage (GPU recommended, e.g. Colab T4):
    pip install -r requirements-lora.txt
    python train_lora.py                                   # Qwen2.5-0.5B-Instruct, 3 epochs
    python train_lora.py --base Qwen/Qwen2.5-1.5B-Instruct

Only the loss on the Tamil answer is trained (the prompt tokens are masked).
The 90 held-out records from thanglish_split.py are excluded from training.
Writes the adapter to --out and run info (parameter counts, timings) to results/train_info.json.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from thanglish_split import fields, load_records, split

HERE = Path(__file__).resolve().parent


def build_example(tok, record, max_len):
    system, thanglish, tamil = fields(record)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": thanglish}]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tok(tamil + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + answer_ids
    if len(ids) > max_len:
        return None
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=str(HERE / "adapters" / "thanglish_lora"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=-1, help="smoke tests only")
    args = ap.parse_args()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    train_records, _ = split(load_records())
    examples = [build_example(tok, r, args.max_len) for r in train_records]
    kept = [e for e in examples if e is not None]
    print(f"training examples: {len(kept)} of {len(train_records)} (skipped {len(examples) - len(kept)} longer than {args.max_len} tokens)")

    try:                                    # `dtype` in recent transformers, `torch_dtype` in older ones
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    total = sum(p.numel() for p in model.parameters())
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"base parameters: {total:,}  trainable (LoRA): {trainable:,} ({100 * trainable / total:.2f}%)")

    steps_per_epoch = max(1, len(kept) // (args.batch * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(0.05 * total_steps))   # integer steps: works across transformers versions

    targs = TrainingArguments(
        output_dir=str(HERE / "adapters" / "_trainer"), per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        max_steps=args.max_steps, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=warmup_steps, logging_steps=10, save_strategy="no", report_to="none",
        bf16=use_bf16, remove_unused_columns=False, seed=42)
    trainer = Trainer(model=model, args=targs, train_dataset=kept, data_collator=Collator(tok.pad_token_id))

    t0 = time.time()
    result = trainer.train()
    seconds = time.time() - t0

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    info = {"base": args.base, "base_parameters": total, "lora_trainable_parameters": trainable,
            "lora_rank": args.rank, "train_examples": len(kept), "epochs": args.epochs,
            "learning_rate": args.lr, "final_train_loss": result.training_loss,
            "train_seconds": round(seconds, 1), "dtype": str(dtype),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
