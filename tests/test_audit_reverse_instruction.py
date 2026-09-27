import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_reverse_instruction.py"
SPEC = importlib.util.spec_from_file_location("audit_reverse_instruction", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)

THRESHOLDS = {
    "bullets": 10,
    "length_ratio": 15.0,
    "term_recall": 0.3,
    "versions": 10,
    "narrow_list_bullets": 5,
}


def _record(question: str, answer: str, *, category: str = "concepto") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sistema"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "category": category,
            "content_hash": "h",
            "normalization": "reverse_instruction",
            "source": "docker_docs",
        },
    }


CHANGELOG = "\n".join(
    ["#### For all platforms", ""]
    + [f"- Fixed issue {n}. Fixes [docker/for-mac#{n}](https://x/{n})." for n in range(6)]
)


class ReverseInstructionAuditTests(unittest.TestCase):
    def test_changelog_chunk_for_narrow_question_is_flagged(self) -> None:
        record = _record(
            "¿Qué cambio se realizó en Docker Init para las aplicaciones Java con Spring Boot?",
            CHANGELOG,
        )
        found = audit.flags(audit.measure(record), THRESHOLDS)
        self.assertIn("release_notes", found)
        self.assertIn("narrow_question_list_answer", found)
        self.assertIn("ends_in_structure", found)

    def test_enumeration_question_is_not_narrow(self) -> None:
        record = _record("Enumera todos los cambios de la versión.", CHANGELOG)
        self.assertNotIn(
            "narrow_question_list_answer", audit.flags(audit.measure(record), THRESHOLDS)
        )

    def test_yaml_sequences_are_not_list_or_chunk_defects(self) -> None:
        answer = "kind: ClusterRole\nrules:\n- apiGroups:\n  - \"\"\n  verbs:\n  - get\n  - list"
        record = _record("Genera un ClusterRole de solo lectura.", answer, category="generacion_yaml")
        self.assertEqual(audit.flags(audit.measure(record), THRESHOLDS), [])

    def test_natural_endings_without_punctuation_are_not_structural(self) -> None:
        self.assertFalse(audit.ends_in_structure("See https://github.com/org/repo/issues/1"))
        self.assertTrue(audit.ends_in_structure("Intro.\n\n| a | b |"))
        self.assertTrue(audit.ends_in_structure("Options:\n- first"))

    def test_template_residue_and_context_reference(self) -> None:
        record = _record(
            "Según la información proporcionada, ¿qué hace el campo?",
            "## {{% heading \"whatsnext\" %}}\n\nRead [Pods](/docs/concepts/pods/).",
        )
        found = audit.flags(audit.measure(record), THRESHOLDS)
        self.assertIn("template_residue", found)
        self.assertIn("context_reference", found)

    def test_language_mismatch_requires_enough_evidence(self) -> None:
        self.assertIsNone(audit.spanish_share("docker run"))
        record = _record(
            "¿Cómo se configura el límite de memoria de un contenedor en la red de Docker?",
            "You can set the memory limit of the container with the flag, and it is applied to the process.",
        )
        self.assertIn("language_mismatch", audit.flags(audit.measure(record), THRESHOLDS))

    def test_audit_writes_hashes_and_flags_without_text(self) -> None:
        record = _record("¿Qué se corrigió en Docker Desktop para Mac?", CHANGELOG)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "train.jsonl"
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")
            flagged = Path(tmp) / "out" / "flags.jsonl"
            report = audit.audit(source, thresholds=THRESHOLDS, flagged_output=flagged)
            rows = [json.loads(line) for line in flagged.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(report["flagged_records"], 1)
        self.assertEqual(set(rows[0]), {"content_hash", "normalization", "source", "category", "flags"})
        self.assertNotIn("Fixed issue", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
