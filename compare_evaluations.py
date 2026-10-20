"""Compare full base-model and LoRA evaluations without inventing results.

Run both evaluate_model.py commands first. The comparison fails if either file
is missing, evaluated on a subset, or used a different base model or dataset.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("rows"), list) or not isinstance(data.get("summary"), dict):
        raise ValueError(f"{path}: expected summary and per-example rows")
    return data

def compare(base, tuned, expected=90):
    a, b = load(base), load(tuned)
    sa, sb = a["summary"], b["summary"]
    if sa["base"] != sb["base"]:
        raise ValueError("Base-model identifiers differ; rerun both evaluations with --base set identically")
    if sa.get("adapter") or not sb.get("adapter"):
        raise ValueError("Expected an unadapted baseline and an adapted LoRA evaluation")
    if len(a["rows"]) != expected or len(b["rows"]) != expected:
        raise ValueError(f"Expected {expected} held-out examples in both runs; smoke tests are not reportable")
    keys = ("thanglish", "reference")
    if [tuple(r[k] for k in keys) for r in a["rows"]] != [tuple(r[k] for k in keys) for r in b["rows"]]:
        raise ValueError("Evaluations have mismatched examples or ordering")
    metrics = ("exact_match", "mean_cer", "median_cer", "mean_latency_s", "tokens_per_second")
    result = {"base_model": sa["base"], "held_out_records": expected,
              "baseline_device": sa.get("device"), "adapter_device": sb.get("device"),
              "metrics": {m: {"baseline": sa[m], "lora": sb[m]} for m in metrics},
              "evaluation_note": "Held-out synthetic transliterations from the same rule engine, not independent real-world Thanglish."}
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, default=ROOT / "results/eval_base.json")
    p.add_argument("--adapted", type=Path, default=ROOT / "results/eval_lora.json")
    p.add_argument("--output", type=Path, default=ROOT / "results/comparison.json")
    args = p.parse_args()
    result = compare(args.baseline, args.adapted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
