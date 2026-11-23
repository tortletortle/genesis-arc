import unittest
import numpy as np
from dictionary import (N, all_programs, apply_program, dataset_cost,
                        fit_dictionary, roundtrip)
from experiment import make_family, observe_and_estimate

class SelfOrganizingDictionaryTests(unittest.TestCase):
    def test_program_round_trip(self):
        rng = np.random.default_rng(2)
        d = [tuple(map(int, rng.permutation(N))) for _ in range(4)]
        for depth in (1, 2, 3):
            for seq in [(0,) * depth, tuple(range(depth))]:
                key = apply_program(d, seq)
                self.assertEqual(apply_program(d, all_programs(d)[key]), key)

    def test_pair_lookahead_finds_jointly_useful_primitives(self):
        rng = np.random.default_rng(2026)
        hidden, train, held = make_family(rng)
        keys = list(hidden) + [k for _, k in train]
        fit = fit_dictionary(keys)
        self.assertEqual(set(fit.dictionary), set(hidden))
        self.assertTrue(fit.bits < 40 * len(keys))
        decoded = roundtrip(keys, fit.dictionary, fit.encodings)
        self.assertEqual(decoded, {k: k for k in keys})

    def test_singleton_only_ablation_has_zero_gain_on_pair_compositions(self):
        rng = np.random.default_rng(2027)
        hidden, train, _ = make_family(rng)
        keys = list(hidden) + [k for _, k in train]
        fit = fit_dictionary(keys, pair_lookahead=False)
        self.assertEqual(fit.dictionary, [])
        self.assertEqual(fit.bits, 40 * len(keys))

    def test_dictionary_uses_observed_maps_only(self):
        rng = np.random.default_rng(2028)
        hidden, train, held = make_family(rng)
        train_keys = list(hidden) + [k for _, k in train]
        held_keys = [k for _, k in held]
        fit = fit_dictionary(train_keys)
        self.assertTrue(set(fit.dictionary).issubset(set(train_keys)))
        _, held_enc = dataset_cost(held_keys, fit.dictionary)
        decoded = roundtrip(held_keys, fit.dictionary, held_enc)
        self.assertEqual(decoded, {k: k for k in held_keys})

    def test_noisy_estimation_is_separate_from_lossless_codec(self):
        rng = np.random.default_rng(2029)
        key = tuple(map(int, rng.permutation(N)))
        estimate, counts = observe_and_estimate(key, .2, 5, rng)
        self.assertEqual(counts.shape, (N, N))
        fit = fit_dictionary([estimate])
        self.assertEqual(roundtrip([estimate], fit.dictionary, fit.encodings)[estimate], estimate)

if __name__ == '__main__':
    unittest.main()
