import json
import tempfile
import unittest
from pathlib import Path

from compare_evaluations import compare

class ComparisonTests(unittest.TestCase):
    def test_rejects_smoke_test(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "base.json", Path(d) / "lora.json"
            row = {"thanglish": "vanakkam", "reference": "வணக்கம்"}
            for path, adapter in ((a, None), (b, "adapter")):
                path.write_text(json.dumps({"summary": {"base": "qwen", "adapter": adapter}, "rows": [row]}))
            with self.assertRaisesRegex(ValueError, "held-out"):
                compare(a, b)
if __name__ == "__main__":
    unittest.main()
