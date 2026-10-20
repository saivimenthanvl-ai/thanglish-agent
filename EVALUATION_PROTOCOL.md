# Reproducible LLM evaluation

The repository already includes `train_lora.py`, `evaluate_model.py`, and a fixed 813/90 split. These files do **not** yet provide a published full training result.

Run on a GPU from the repository root:

```bash
pip install -r requirements.txt -r requirements-lora.txt
python evaluate_model.py --base Qwen/Qwen2.5-0.5B-Instruct
python train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct
python evaluate_model.py --base Qwen/Qwen2.5-0.5B-Instruct --adapter adapters/thanglish_lora
python compare_evaluations.py
python -m unittest test_compare_evaluations.py
```

`compare_evaluations.py` checks that **both evaluations include the same 90 held-out records** and the same base model, refusing to summarize two-record smoke tests or mismatched experiments. Commit `results/comparison.json` together with the two raw evaluation files only after running the commands, ensuring no personal or sensitive data is included.

Report exact match, mean CER, latency per example and tokens/s, hardware, model version, training cost and total time. Latency is environment-dependent and should use the same GPU for before/after.

**Interpretation:** All 903 examples come from deterministic transliteration rules applied to Tamil source sentences. A held-out score tests inversion of those rules, not accuracy on naturally written conversational Thanglish. Keep a separately sourced, human-checked evaluation set before making a real-world performance claim. A public LoRA run on Qwen 1.5B does not establish the origin or quality of any separate private Deffotech model.
