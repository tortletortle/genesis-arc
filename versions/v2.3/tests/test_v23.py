import unittest
import numpy as np
from active import query_information, select_query
from open_set import identify_open_set
from latent import (assess_identifiability, compose, gauge_transform, inverse,
                    pairwise_composites, recover_first_factor, recover_second_factor)
from online import VersionedDictionary

class LatentDiscoveryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.ps = [tuple(map(int, rng.permutation(8))) for _ in range(4)]

    def test_composition_and_inverse(self):
        ident = tuple(range(8))
        for p in self.ps:
            self.assertEqual(compose(p, inverse(p)), ident)
            self.assertEqual(compose(inverse(p), p), ident)

    def test_recover_each_factor_when_other_is_anchored(self):
        a, b = self.ps[:2]
        c = compose(a, b)
        self.assertEqual(recover_second_factor(c, a), b)
        self.assertEqual(recover_first_factor(c, b), a)

    def test_anchor_identifies_all_latent_operators(self):
        result = assess_identifiability(pairwise_composites(self.ps), 4, (0, self.ps[0]))
        self.assertTrue(result.unique_original_primitives_identifiable)
        self.assertEqual(result.canonical_model, tuple(self.ps))

    def test_unanchored_system_only_identifies_gauge_class(self):
        result = assess_identifiability(pairwise_composites(self.ps), 4)
        self.assertFalse(result.unique_original_primitives_identifiable)
        self.assertEqual(result.gauge_equivalence_class_size_lower_bound, 40320)
        self.assertTrue(result.pairwise_consistent)

    def test_incomplete_pairwise_evidence_is_rejected(self):
        composites = pairwise_composites(self.ps)
        composites.pop((1, 2))
        with self.assertRaises(ValueError):
            assess_identifiability(composites, 4)

    def test_gauge_transform_preserves_all_observations(self):
        h = tuple(map(int, np.random.default_rng(9).permutation(8)))
        self.assertEqual(pairwise_composites(self.ps),
                         pairwise_composites(gauge_transform(self.ps, h)))

    def test_query_chooses_an_input_that_separates_candidates(self):
        candidates = [tuple(range(8)), (0,2,1,3,4,5,6,7),
                      (0,1,2,4,3,5,6,7), (0,1,2,3,4,6,5,7)]
        posterior = [0.25] * 4
        x = select_query(candidates, posterior, noise=0)
        self.assertIsNotNone(x)
        self.assertNotEqual(x, 0)
        self.assertGreater(query_information(candidates, posterior, x, 0), 0)

    def test_open_set_accepts_member_and_rejects_novel_map(self):
        candidates = [tuple(range(8)), (0,2,1,3,4,5,6,7),
                      (0,1,2,4,3,5,6,7), (0,1,2,3,4,6,5,7)]
        novel = (0,2,3,4,5,6,7,1)
        accepted = identify_open_set(candidates, candidates[2], noise=0, strategy="active",
                                     validation_inputs=4, verification_repeats=3)
        rejected = identify_open_set(candidates, novel, noise=0, strategy="active",
                                     validation_inputs=4, verification_repeats=3)
        self.assertEqual(accepted.decision, "identified")
        self.assertTrue(accepted.correct)
        self.assertEqual(rejected.decision, "rejected_unknown")
        discovery = [x for x in accepted.history if x["phase"] == "discovery"]
        validation = [x for x in accepted.history if x["phase"] == "validation"]
        self.assertTrue(all(x["x"] < 4 for x in discovery))
        self.assertTrue(all(x["x"] >= 4 for x in validation))

class OnlineMemoryTests(unittest.TestCase):
    def setUp(self):
        self.p1 = (1,2,3,4,5,6,7,0)
        self.p2 = (0,2,1,4,3,6,5,7)
        self.c12 = compose(self.p1, self.p2)
        self.c21 = compose(self.p2, self.p1)
        self.corpus = [self.p1, self.p2, self.c12, self.c21]

    def test_version_pinning_preserves_old_records(self):
        memory = VersionedDictionary([self.p1], max_entries=4)
        record = memory.encode(self.c12)
        old_version = record.version_id
        self.assertTrue(memory.add_candidate(self.p2, self.corpus))
        self.assertNotEqual(memory.active_version, old_version)
        self.assertEqual(memory.decode(record), self.c12)
        self.assertEqual(memory.records[record.record_id].version_id, old_version)

    def test_atomic_migration_reencodes_and_roundtrips(self):
        memory = VersionedDictionary([self.p1], max_entries=4)
        record = memory.encode(self.c12)
        self.assertTrue(memory.add_candidate(self.p2, self.corpus))
        self.assertTrue(memory.migrate_all_atomically())
        migrated = memory.records[record.record_id]
        self.assertEqual(migrated.version_id, memory.active_version)
        self.assertEqual(memory.decode(migrated), self.c12)
        self.assertEqual(memory.audit()["roundtrip_rate"], 1.0)

    def test_prune_to_budget_migrates_records_atomically(self):
        memory = VersionedDictionary([self.p1, self.p2], max_entries=4)
        memory.encode(self.c12)
        selected = memory.prune_to_budget(self.corpus, budget=1, migrate=True)
        self.assertLessEqual(len(selected), 1)
        self.assertEqual(memory.audit()["roundtrip_rate"], 1.0)
        self.assertTrue(all(r.version_id == memory.active_version for r in memory.records.values()))

    def test_failed_migration_does_not_replace_records(self):
        memory = VersionedDictionary([self.p1], max_entries=4)
        record = memory.encode(self.c12)
        self.assertTrue(memory.add_candidate(self.p2, self.corpus))
        memory.records[record.record_id] = type(record)(
            record.record_id, record.key, 999, record.kind, record.payload)
        corrupted = dict(memory.records)
        self.assertFalse(memory.migrate_all_atomically())
        self.assertEqual(memory.records, corrupted)

if __name__ == "__main__":
    unittest.main()
