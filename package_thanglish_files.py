"""Package the 903 Thanglish prompt/completion dataset files into a verified ZIP.

Run in Windows PowerShell:
    python package_thanglish_files.py

The output is a ZIP placed beside your 'output' folder.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from zipfile import ZipFile, ZIP_DEFLATED
import json
import os
import re

DATA_DIR = Path(r"C:\Users\saivi\Downloads\thanglish data\output")
ZIP_PATH = DATA_DIR.parent / "thanglish_903_prompt_completion_dataset.zip"
EXPECTED_COUNT = 903
NAME_PATTERN = re.compile(r"^thanglish_pair_\d+\.(jsonl|txt)$")


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    files = sorted(
        p for p in DATA_DIR.glob("thanglish_pair_*.*")
        if p.is_file() and NAME_PATTERN.fullmatch(p.name)
    )
    if len(files) != EXPECTED_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_COUNT} Thanglish dataset files, "
            f"but found {len(files)} in {DATA_DIR}. "
            "No incomplete ZIP was created."
        )

    bad_files = []
    rows = []
    for file in files:
        contents = file.read_bytes()
        try:
            line = contents.decode("utf-8-sig").strip()
            record = json.loads(line)
            prompt = record.get("prompt") if isinstance(record, dict) else None
            completion = record.get("completion") if isinstance(record, dict) else None
            if not isinstance(prompt, list) or not isinstance(completion, list):
                bad_files.append(f"{file.name}: missing prompt/completion lists")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            bad_files.append(f"{file.name}: {exc}")

        rows.append({
            "filename": file.name,
            "bytes": len(contents),
            "sha256": sha256(contents).hexdigest(),
        })

    if bad_files:
        raise RuntimeError(
            "Found dataset files that are not valid prompt/completion JSON. "
            "No ZIP was created.\n" + "\n".join(bad_files[:30])
        )

    manifest = {
        "dataset_name": "Thanglish to Tamil 903 Prompt-Completion Dataset",
        "file_count": len(files),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "prompt-completion",
        "note": "Fine-tuning prompt/completion pairs for Thanglish to Tamil conversion.",
        "files": rows,
    }

    temp_path = ZIP_PATH.with_suffix(".zip.part")
    try:
        with ZipFile(temp_path, mode="w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for file in files:
                archive.write(file, arcname=file.name)
            archive.writestr(
                "MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2)
            )

        with ZipFile(temp_path, mode="r") as archive:
            if archive.testzip() is not None:
                raise RuntimeError("ZIP CRC verification failed")
            archived_count = sum(name.startswith("thanglish_pair_") for name in archive.namelist())
            if archived_count != EXPECTED_COUNT:
                raise RuntimeError(f"ZIP contains {archived_count} files, expected {EXPECTED_COUNT}")

        os.replace(temp_path, ZIP_PATH)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Created: {ZIP_PATH}")
    print(f"Files: {len(files)}")
    print(f"ZIP size: {ZIP_PATH.stat().st_size:,} bytes")
    print("Integrity: verified")


if __name__ == "__main__":
    main()