import unittest
import numpy as np
import v19_baseline as b
from memory import BitMemory, Record

class MemoryCodecTests(unittest.TestCase):
    def test_program_round_trip_all_dsl_maps(self):
        mem = BitMemory(100000, 'program')
        for i, key in enumerate(b.KEYS):
            mem.data[key] = Record(.9, i, i, .5)
        self.assertTrue(mem.validate())
        raw = BitMemory(100000, 'raw')
        raw.data.update(mem.data)
        self.assertLess(mem.bits_used, raw.bits_used)

    def test_delta_round_trip_mixed_maps(self):
        rng = np.random.default_rng(20261123)
        keys = list(b.KEYS[:30])
        while len(keys) < 70:
            key = tuple(map(int, rng.integers(0, b.N, size=b.N)))
            if key not in keys:
                keys.append(key)
        mem = BitMemory(100000, 'delta')
        for i, key in enumerate(keys):
            mem.data[key] = Record(.8, i, i, .4)
        self.assertTrue(mem.validate())
        records, code_bits = mem._encoded_records()
        self.assertEqual(sum(r[-1] for r in records), code_bits)

    def test_budget_is_enforced_for_all_codecs(self):
        for codec in ('raw', 'program', 'delta'):
            mem = BitMemory(480, codec)
            agent = b.Agent('utility_gate', .1, {})
            target = b.v.TIERS[0][0]
            for x in b.X:
                for _ in range(3):
                    agent.update(0, int(x), int(target[x]))
            mem.update(agent, 0)
            self.assertLessEqual(mem.bits_used, mem.cap)
            self.assertTrue(mem.validate())

    def test_raw_cost_is_exact(self):
        mem = BitMemory(10000, 'raw')
        for i, key in enumerate(b.KEYS[:10]):
            mem.data[key] = Record(.8, i, i, .4)
        self.assertEqual(mem.bits_used, 10 * 40)

if __name__ == '__main__':
    unittest.main()
