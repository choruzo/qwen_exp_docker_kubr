import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_training_repetition.py"
SPEC = importlib.util.spec_from_file_location("audit_training_repetition", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


class TrainingRepetitionAuditTests(unittest.TestCase):
    def test_assistant_text_excludes_non_assistant_turns(self) -> None:
        record = {"messages": [
            {"role": "user", "content": "secret prompt"},
            {"role": "assistant", "content": "answer one"},
            {"role": "assistant", "content": "answer two"},
        ]}
        self.assertEqual(audit.assistant_text(record), "answer one\nanswer two")

    def test_repeated_ngram_ratio(self) -> None:
        self.assertEqual(audit.repeated_ngram_ratio("a b c d", 2), 0.0)
        self.assertAlmostEqual(audit.repeated_ngram_ratio("a b a b", 2), 1 / 3)


if __name__ == "__main__":
    unittest.main()
