import unittest

from premo_study.model_compare import matched_evaluations


class ModelCompareTests(unittest.TestCase):
    def test_rejects_unmatched_cohort(self):
        records = [{"id": "a"}]
        qwen = {"complete_ids": ["a"], "adapter": {"backend": "ollama"}, "predictions": []}
        claude = {"complete_ids": ["b"], "adapter": {"backend": "claude"}, "predictions": []}
        with self.assertRaises(ValueError):
            matched_evaluations(records, qwen, claude, iterations=0)


if __name__ == "__main__":
    unittest.main()
