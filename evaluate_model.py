"""Evaluate Thanglish -> Tamil generation on the 90 held-out records, with or without a LoRA adapter.

Usage:
    python evaluate_model.py                               # base model, no adapter (baseline)
    python evaluate_model.py --adapter adapters/thanglish_lora

Greedy decoding, one record at a time. Reports:
  exact match      fraction of outputs identical to the Tamil target
  CER              character error rate = edit distance / reference length (lower is better)
  latency          mean seconds per record; tokens/sec is generated tokens over generation time
Writes results/eval_<base|lora>.json. Run both and compare.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from thanglish_split import fields, load_records, split

HERE = Path(__file__).resolve().parent


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference, hypothesis):
    return edit_distance(reference, hypothesis) / max(1, len(reference))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter from train_lora.py")
    ap.add_argument("--n", type=int, default=None, help="limit records (smoke tests)")
    ap.add_argument("--max_new_tokens", type=int, default=600)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32)

    tok = AutoTokenizer.from_pretrained(args.base)
    try:                                    # `dtype` in recent transformers, `torch_dtype` in older ones
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    _, test_records = split(load_records())
    if args.n:
        test_records = test_records[:args.n]

    rows, gen_tokens, gen_seconds = [], 0, 0.0
    for rec in test_records:
        system, thanglish, tamil = fields(rec)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": thanglish}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        pred = tok.decode(new_ids, skip_special_tokens=True).strip()
        gen_tokens += len(new_ids)
        gen_seconds += dt
        rows.append({"thanglish": thanglish, "reference": tamil, "prediction": pred,
                     "exact": pred == tamil.strip(), "cer": cer(tamil.strip(), pred), "seconds": dt})

    summary = {
        "base": args.base, "adapter": args.adapter, "records": len(rows), "device": device,
        "exact_match": sum(r["exact"] for r in rows) / len(rows),
        "mean_cer": statistics.mean(r["cer"] for r in rows),
        "median_cer": statistics.median(r["cer"] for r in rows),
        "mean_latency_s": gen_seconds / len(rows),
        "tokens_per_second": gen_tokens / gen_seconds if gen_seconds else None,
    }
    print(json.dumps(summary, indent=2))
    for r in rows[:3]:
        print("\nTHANGLISH :", r["thanglish"], "\nREFERENCE :", r["reference"], "\nPREDICTION:", r["prediction"])

    (HERE / "results").mkdir(exist_ok=True)
    name = "lora" if args.adapter else "base"
    (HERE / "results" / f"eval_{name}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
