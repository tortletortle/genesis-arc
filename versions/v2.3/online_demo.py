"""Versioned dictionary online-update demonstration."""
from online import VersionedDictionary
from dictionary import dataset_cost
from latent import compose

def run_online_demo():
    p1 = (1, 2, 3, 4, 5, 6, 7, 0)
    p2 = (0, 2, 1, 4, 3, 6, 5, 7)
    c12, c21 = compose(p1, p2), compose(p2, p1)
    corpus = [p1, p2, c12, c21]
    memory = VersionedDictionary([p1], max_entries=4)
    old_record, second_old_record = memory.encode(c12), memory.encode(c21)
    old_cost, _ = dataset_cost(corpus, [p1])
    gain = memory.addition_gain(p2, corpus)
    accepted = memory.add_candidate(p2, corpus)
    new_cost, _ = dataset_cost(corpus, list(memory.active.entries))
    old_ok = memory.decode(old_record) == c12
    second_old_ok = memory.decode(second_old_record) == c21
    migrated = memory.migrate_all_atomically()
    return {
        "old_cost_bits": old_cost, "addition_gain_bits": gain,
        "candidate_accepted": accepted, "new_cost_bits": new_cost,
        "old_record_still_decodes_before_migration": old_ok,
        "second_old_record_still_decodes_before_migration": second_old_ok,
        "atomic_migration_succeeded": migrated,
        "record_versions_after_migration": sorted({r.version_id for r in memory.records.values()}),
        "audit": memory.audit(),
    }
