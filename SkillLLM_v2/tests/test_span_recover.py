import unittest

from src.postprocess.span_recover import SpanRecoverer


class SpanRecovererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recoverer = SpanRecoverer(threshold=0.85)

    def test_exact_indices_are_preserved(self) -> None:
        text = "Python programming and stakeholder communication are required."
        predicted = [
            {
                "text": "Python programming",
                "label": "hard_skill",
                "start": 0,
                "end": 18,
            }
        ]
        recovered, trace = self.recoverer.recover_example("exact", text, predicted)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["recovery_type"], "exact")
        self.assertFalse(trace["rejected"])

    def test_whitespace_normalized_recovery(self) -> None:
        text = "Hands-on  stakeholder   communication is valued."
        predicted = [{"text": "stakeholder communication", "label": "soft_skill"}]
        recovered, _ = self.recoverer.recover_example("normalized", text, predicted)
        self.assertEqual(recovered[0]["text"], "stakeholder   communication")
        self.assertEqual(recovered[0]["recovery_type"], "normalized")

    def test_case_insensitive_recovery(self) -> None:
        text = "Experience with SQL is expected."
        predicted = [{"text": "sql", "label": "hard_skill"}]
        recovered, _ = self.recoverer.recover_example("casefold", text, predicted)
        self.assertEqual(recovered[0]["text"], "SQL")
        self.assertEqual(recovered[0]["recovery_type"], "normalized")

    def test_fuzzy_recovery(self) -> None:
        text = "Experience with stakeholder communication is expected."
        predicted = [{"text": "stakeholder comunication", "label": "soft_skill"}]
        recovered, _ = self.recoverer.recover_example("fuzzy", text, predicted)
        self.assertEqual(recovered[0]["text"], "stakeholder communication")
        self.assertEqual(recovered[0]["recovery_type"], "fuzzy")

    def test_same_label_overlap_resolution_keeps_better_span(self) -> None:
        text = "Python programming"
        predicted = [
            {"text": "Python programming", "label": "hard_skill"},
            {"text": "programming", "label": "hard_skill"},
        ]
        recovered, _ = self.recoverer.recover_example("overlap", text, predicted)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["text"], "Python programming")


if __name__ == "__main__":
    unittest.main()
