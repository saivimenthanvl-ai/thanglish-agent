# Current implementation and evidence

This project publishes the reproducible 903-pair deterministic dataset pipeline, an 813/90 split, LoRA training scripts, evaluation scripts and comparison tooling.

## Verified scope

- The synthetic dataset pipeline regenerates 903/903 inputs from Tamil targets. This establishes internal consistency, **not** ML model accuracy.
- CPU smoke testing of Qwen2.5-0.5B-Instruct covered two training steps and two evaluation examples. It establishes that the code executes; it is **not** a reliable quality benchmark.
- `compare_evaluations.py` validates compatible evaluation JSON files and refuses to report two-example smoke tests as complete 90-record experiments.

## Commands

```bash
pip install -r requirements.txt -r requirements-lora.txt
python verify_dataset.py --regen
python -m unittest test_compare_evaluations.py
```

For usage of optional model scripts, see `python train_lora.py --help` and `python evaluate_model.py --help`.

## Limitations

Synthetic Thanglish from deterministic rules is not equivalent to real-world colloquial user input. This repository is independent of any proprietary Deffotech model. No unpublished training benchmark or private checkpoint is presented as evidence.
