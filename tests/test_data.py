import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from premo_study.data import audit_dataset, load_dataset


def source_row(index=1, emotions=None):
    emotions = emotions if emotions is not None else [["joy"], ["joy", "anger"], ["sadness"]]
    answers = []
    for position, selected in enumerate(emotions):
        annotation = {"polarity": "neutral", "emotions": selected, "confidence": 4, "positive_intensity": 1, "negative_itensity": 1, "comments": "retained"}
        answers.append({"user_type": "neuro" if position == 0 else "comp", "part1": copy.deepcopy(annotation), "part2": copy.deepcopy(annotation)})
    return {"project": "repo", "raw_message": "Thanks, but I am worried.", "message_url": f"https://github.com/owner/repo/pull/{index}#issuecomment-{index}", "part1_aggregate": {"polarity": "neutral", "avg_confidence": 4, "agreement_type": "all"}, "part2_aggregate": {"polarity": "neutral", "avg_confidence": 4, "agreement_type": "all"}, "individual_answers": answers}


class DatasetTests(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            return load_dataset(path)

    def test_independent_per_emotion_vote_and_lossless_original(self):
        row = source_row()
        row["individual_answers"][0]["part2"]["emotions"] = ["fear", "anger"]
        row["individual_answers"][1]["part2"]["emotions"] = ["fear", "anger"]
        record = self.load([row])[0]
        self.assertEqual(record["consensus"]["part1"], ["joy"])
        self.assertEqual(set(record["consensus"]["part2"]), {"fear", "anger"})
        self.assertEqual(record["source_record"], row)
        self.assertIn("negative_itensity", record["source_record"]["individual_answers"][0]["part1"])
        self.assertEqual(record["id"], hashlib.sha256(row["message_url"].encode()).hexdigest()[:16])

    def test_empty_majority_does_not_imply_neutral_or_unanimous_absence(self):
        rows = [source_row(1, [["joy"], ["anger"], ["fear"]]), source_row(2, [[], [], []])]
        records = self.load(rows)
        self.assertEqual(records[0]["consensus"]["part1"], [])
        self.assertFalse(records[0]["all_annotators_no_emotion"]["part1"])
        self.assertTrue(records[1]["all_annotators_no_emotion"]["part1"])
        self.assertNotIn("neutral", records[0]["consensus"]["part1"])

    def test_bare_url_is_candidate_pr_body(self):
        row = source_row()
        row["message_url"] = row["message_url"].split("#")[0]
        self.assertEqual(self.load([row])[0]["target_kind"], "pr_body")

    def test_rejects_corrupt_schemas(self):
        mutations = [
            lambda row: row["individual_answers"].pop(),
            lambda row: row["individual_answers"][0]["part1"].update(emotions=["neutral"]),
            lambda row: row["individual_answers"][0]["part1"].update(emotions=["joy", "joy"]),
            lambda row: row["individual_answers"][0]["part2"].update(confidence=True),
            lambda row: row["individual_answers"][0].pop("part2"),
            lambda row: row["individual_answers"][0].update(user_type="comp"),
            lambda row: row.update(message_url="https://evil.example/owner/repo/pull/1"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                row = source_row()
                mutate(row)
                with self.assertRaises(ValueError):
                    self.load([row])
        with self.assertRaises(ValueError):
            self.load([source_row(), source_row()])

    def test_audit_reports_expected_discrepancy_without_fabricating_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps([source_row()]), encoding="utf-8")
            audit = audit_dataset(path)
            self.assertEqual(audit["comments"], 1)
            self.assertEqual(audit["expected_characteristic_discrepancies"]["comments"]["expected"], 1791)
            self.assertEqual(audit["passes"]["part2"]["neutral_polarity_with_emotion"], 3)


if __name__ == "__main__":
    unittest.main()
