import unittest

from premo_study.pilot import select_pilot


class PilotTests(unittest.TestCase):
    def test_seed_pair_eligibility_and_small_pool(self):
        records = [{"id": str(i), "consensus": {"part1": [], "part2": ["joy"] if i % 2 else []}} for i in range(100)]
        first = select_pilot(records)
        self.assertEqual(first, select_pilot(list(reversed(records))))
        self.assertEqual(first["selected_size"], 75)
        self.assertEqual(first["human_sensitive_count"], 25)
        self.assertEqual(len(set(first["ids"])), 75)
        eligible = select_pilot(records, eligible_ids=["1", "2"])
        self.assertEqual(set(eligible["ids"]), {"1", "2"})
        self.assertEqual(select_pilot(records, eligible_ids=[])["ids"], [])


if __name__ == "__main__":
    unittest.main()
