"""Verify and summarize train.jsonl (Thanglish -> Tamil chat records).

Usage:
    python verify_dataset.py            # schema, uniqueness and length statistics
    python verify_dataset.py --regen    # also re-run the transliterator (needs open-tamil)

--regen takes each Tamil target, runs thanglish_transliterator.to_thanglish on it and
checks the result equals the Thanglish input stored in the record. This shows the
dataset is reproducible from the rules; it does not measure model quality.
Exit code is 1 if any check fails.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = 903


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_schema(rec):
    p, c = rec["prompt"], rec["completion"]
    return (
        [m["role"] for m in p] == ["system", "user"]
        and [m["role"] for m in c] == ["assistant"]
        and all(m["content"].strip() for m in p + c)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(HERE / "train.jsonl"))
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()

    recs = load(args.file)
    tamil = [r["completion"][0]["content"] for r in recs]
    thang = [r["prompt"][1]["content"] for r in recs]
    lens = [len(t) for t in tamil]

    results = {
        f"record count == {EXPECTED}": len(recs) == EXPECTED,
        "schema valid (system+user -> assistant, non-empty)": all(map(check_schema, recs)),
        "Tamil targets unique": len(set(tamil)) == len(tamil),
        "Thanglish inputs unique": len(set(thang)) == len(thang),
        "single system prompt": len({r["prompt"][0]["content"] for r in recs}) == 1,
    }

    print(f"records: {len(recs)}")
    print(
        "Tamil chars  min/mean/median/max: "
        f"{min(lens)} / {statistics.mean(lens):.1f} / {statistics.median(lens)} / {max(lens)}"
    )
    print(f"Thanglish chars mean: {statistics.mean(len(t) for t in thang):.1f}")

    if args.regen:
        from thanglish_transliterator import to_thanglish

        bad = [i for i, (ta, th) in enumerate(zip(tamil, thang)) if to_thanglish(ta) != th]
        results[f"transliterator reproduces all Thanglish inputs ({len(recs) - len(bad)}/{len(recs)})"] = not bad
        if bad:
            print("first mismatches (record index):", bad[:10])

    for name, ok in results.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
