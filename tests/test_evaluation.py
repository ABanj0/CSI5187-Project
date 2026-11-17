import copy
import unittest

from premo_study.evaluation import QUALITATIVE_CODEBOOK, binary_kappa, evaluate, exact_mcnemar, human_agreement, multilabel_metrics


def record(identifier, truth, pr="1", text="Thanks"):
    answer = {"part1": {"emotions": truth}, "part2": {"emotions": truth}, "user_type": "comp"}
    return {"id": identifier, "project": "repo", "message_url": f"https://github.com/owner/repo/pull/{pr}#issuecomment-{identifier}", "pr_url": f"https://github.com/owner/repo/pull/{pr}", "target_kind": "issue_comment", "raw_message": text, "consensus": {"part1": truth, "part2": truth}, "source_record": {"individual_answers": [dict(copy.deepcopy(answer), user_type="neuro"), copy.deepcopy(answer), copy.deepcopy(answer)]}}


def predictions(identifier, before, after):
    return [{"id": identifier, "condition": "comment_only", "emotions": before}, {"id": identifier, "condition": "context", "emotions": after}]


class EvaluationTests(unittest.TestCase):
    def test_metrics_against_hand_calculated_multilabel_example(self):
        result = multilabel_metrics([{"joy", "anger"}, set()], [{"joy", "fear"}, {"love"}])
        self.assertAlmostEqual(result["micro_f1"], 2 / 5)
        self.assertAlmostEqual(result["hamming_loss"], 3 / 14)
        self.assertEqual(result["subset_accuracy"], 0)
        self.assertEqual(result["per_emotion"]["joy"]["tp"], 1)
        self.assertEqual(result["per_emotion"]["fear"]["fp"], 1)
        self.assertEqual(result["per_emotion"]["anger"]["fn"], 1)
        self.assertEqual(result["per_emotion"]["joy"]["kappa"], 1)

    def test_undefined_kappa_and_empty_cohort_are_not_perfect(self):
        self.assertIsNone(binary_kappa([False, False], [False, False]))
        self.assertEqual(binary_kappa([False, True], [True, False]), -1)
        self.assertIsNone(multilabel_metrics([], [])["micro_f1"])
        result = evaluate([record("a", [])], [], bootstrap_iterations=0)
        self.assertEqual(result["paired_n"], 0)
        self.assertIsNone(result["comparisons"]["part2"]["context"]["micro_f1"])
        self.assertEqual(result["coverage"]["context"]["missing_n"], 1)

    def test_fixed_reference_prevents_label_shift_from_confounded_delta(self):
        row = record("a", ["joy"])
        row["consensus"]["part1"] = ["anger"]
        result = evaluate([row], predictions("a", ["joy"], ["joy"]), bootstrap_iterations=3)
        self.assertEqual(result["comparisons"]["part2"]["delta_context_minus_comment_only"]["micro_f1"], 0)
        self.assertEqual(result["comparisons"]["part1"]["delta_context_minus_comment_only"]["micro_f1"], 0)
        self.assertEqual(result["comparisons"]["part2"]["context"]["micro_f1"], 1)
        self.assertEqual(result["comparisons"]["part1"]["context"]["micro_f1"], 0)

    def test_pairing_reports_explicit_failures_and_absent_predictions(self):
        rows = [record("a", ["joy"]), record("b", ["anger"]), record("c", [])]
        preds = predictions("a", [], ["joy"]) + [{"id": "b", "condition": "context", "status": "error"}, {"id": "b", "condition": "comment_only", "emotions": []}]
        result = evaluate(rows, preds, bootstrap_iterations=0)
        self.assertEqual(result["paired_ids"], ["a"])
        self.assertEqual(result["coverage"]["context"]["failed_n"], 1)
        self.assertEqual(result["coverage"]["context"]["missing_ids"], ["c"])
        self.assertEqual(result["error_counts"]["improved"], 1)
        self.assertIn("Incomplete coverage", result["warning"])

    def test_rejects_duplicate_unknown_or_invalid_predictions(self):
        row = record("a", ["joy"])
        bad_lists = [predictions("a", [], []) * 2, predictions("unknown", [], []), predictions("a", ["positive"], []), [{"id": "a", "condition": "context"}], [{"id": "a", "condition": "unknown", "emotions": []}]]
        for bad in bad_lists:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    evaluate([row], bad, bootstrap_iterations=0)

    def test_bootstrap_resamples_entire_pr_clusters_and_is_deterministic(self):
        rows = [record("a", ["joy"], pr="1"), record("b", ["anger"], pr="1")]
        preds = predictions("a", [], ["joy"]) + predictions("b", [], ["anger"])
        first = evaluate(rows, preds, bootstrap_iterations=12, seed=9)
        second = evaluate(rows, preds, bootstrap_iterations=12, seed=9)
        self.assertEqual(first, second)
        uncertainty = first["uncertainty"]["part2"]
        self.assertEqual(uncertainty["clusters"], 1)
        self.assertEqual(uncertainty["intervals"]["micro_f1"], [1, 1])
        self.assertEqual(uncertainty["arm_intervals"]["comment_only"]["micro_f1"], [0, 0])
        self.assertEqual(uncertainty["arm_intervals"]["context"]["micro_f1"], [1, 1])
        self.assertEqual(uncertainty["arm_intervals"]["context"]["macro_f1"], [2 / 7, 2 / 7])
        self.assertEqual(uncertainty["arm_intervals"]["context"]["f1_anger"], [1, 1])
        self.assertEqual(first["comparisons"]["part2"]["delta_context_minus_comment_only"]["hamming_loss"], -1 / 7)

    def test_human_agreement_pools_roles_without_global_identity(self):
        result = human_agreement([record("a", ["joy"]), record("b", [])])
        self.assertEqual(result["part2"]["comp+neuro"]["joy"]["pairs"], 4)
        self.assertEqual(result["part2"]["comp+comp"]["joy"]["pairs"], 2)
        self.assertEqual(result["part2"]["all_within_comment_pairs"]["joy"]["kappa"], 1)

    def test_model_individual_kappa_uses_humans_not_only_consensus(self):
        rows = [record("a", ["joy"]), record("b", [])]
        rows[0]["source_record"]["individual_answers"][0]["part2"]["emotions"] = []
        result = evaluate(rows, predictions("a", [], ["joy"]) + predictions("b", [], []), bootstrap_iterations=0)
        agreement = result["model_individual_agreement"]["part2"]["context"]
        self.assertEqual(agreement["comp"]["joy"]["pairs"], 4)
        self.assertEqual(agreement["comp"]["joy"]["kappa"], 1)
        self.assertEqual(agreement["neuro"]["joy"]["kappa"], 0)
        self.assertEqual(agreement["local_position_0"]["joy"]["pairs"], 2)
        self.assertAlmostEqual(agreement["all_individual_annotations"]["joy"]["kappa"], 2 / 3)

    def test_exact_mcnemar_matches_known_binomial_probability(self):
        result = exact_mcnemar([False] * 6 + [True], [True] * 6 + [False])
        self.assertEqual(result["comment_only_correct_context_wrong"], 1)
        self.assertEqual(result["comment_only_wrong_context_correct"], 6)
        self.assertEqual(result["discordant_pairs"], 7)
        self.assertEqual(result["two_sided_exact_p"], .125)
        self.assertEqual(exact_mcnemar([True], [True])["two_sided_exact_p"], 1)
        self.assertIsNone(exact_mcnemar([], [])["two_sided_exact_p"])
        with self.assertRaises(ValueError):
            exact_mcnemar([True], [])

    def test_per_label_mcnemar_includes_correct_negatives_and_subset(self):
        result = evaluate([record("a", ["joy"]), record("b", [])], predictions("a", [], ["joy"]) + predictions("b", ["joy"], []), bootstrap_iterations=0)
        tests = result["descriptive_mcnemar"]["passes"]["part2"]
        self.assertEqual(tests["joy"]["comment_only_wrong_context_correct"], 2)
        self.assertEqual(tests["subset_exact_match"]["two_sided_exact_p"], .5)
        self.assertEqual(tests["anger"]["discordant_pairs"], 0)
        self.assertIn("PR dependence", result["descriptive_mcnemar"]["limitations"])

    def test_qualitative_samples_are_deterministic_capped_and_unreviewed(self):
        rows = [record(str(i), ["joy"]) for i in range(60)]
        preds = [prediction for row in rows for prediction in predictions(row["id"], [], [])]
        first = evaluate(rows, preds, bootstrap_iterations=0, seed=7)
        second = evaluate(list(reversed(rows)), list(reversed(preds)), bootstrap_iterations=0, seed=7)
        self.assertEqual(first["error_samples"], second["error_samples"])
        samples = [sample for group in first["error_samples"].values() for sample in group]
        self.assertEqual(len(samples), 50)
        self.assertEqual(first["qualitative_review"]["sample_size"], 50)
        self.assertEqual(set(QUALITATIVE_CODEBOOK), {"speaker_confusion", "missing_context", "politeness_convention", "sarcasm", "technical_ambiguity", "human_disagreement", "other"})
        self.assertTrue(all(row["manual_category"] is None and row["manual_notes"] is None and row["review_status"] == "unreviewed" for row in samples))
        self.assertTrue(all(row["outcome"] == "unchanged_error" for row in samples))


if __name__ == "__main__":
    unittest.main()
