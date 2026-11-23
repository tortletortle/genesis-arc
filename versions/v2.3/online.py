"""Immutable dictionary snapshots and atomic record migration for online updates."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import sys
from typing import Iterable

_V22_PATH = Path(__file__).resolve().parents[1] / "v2.2"
if str(_V22_PATH) not in sys.path:
    sys.path.insert(0, str(_V22_PATH))
from dictionary import Map, Program, all_programs, dataset_cost, encode_one, decode_encoding, MAX_DEPTH

@dataclass(frozen=True)
class DictionaryVersion:
    version_id: int
    entries: tuple[Map, ...]
    parent_id: int | None

@dataclass(frozen=True)
class EncodedRecord:
    record_id: int
    key: Map
    version_id: int
    kind: str
    payload: Program | None

class VersionedDictionary:
    """Append-only version registry; records pin the exact snapshot they use."""
    def __init__(self, initial_entries: Iterable[Map] = (), max_entries: int = 8,
                 max_depth: int = MAX_DEPTH):
        entries = tuple(dict.fromkeys(tuple(x) for x in initial_entries))
        if len(entries) > max_entries:
            raise ValueError("initial dictionary exceeds max_entries")
        self.max_entries = max_entries
        self.max_depth = max_depth
        self.versions = {0: DictionaryVersion(0, entries, None)}
        self.active_version = 0
        self.records: dict[int, EncodedRecord] = {}
        self._next_record_id = 1

    @property
    def active(self) -> DictionaryVersion:
        return self.versions[self.active_version]

    def _encode_key(self, key: Map, entries: tuple[Map, ...] | list[Map]):
        entries = list(entries)
        return encode_one(tuple(key), entries, all_programs(entries, self.max_depth))

    def encode(self, key: Map) -> EncodedRecord:
        kind, payload, _ = self._encode_key(tuple(key), self.active.entries)
        record = EncodedRecord(self._next_record_id, tuple(key), self.active_version, kind, payload)
        self.records[record.record_id] = record
        self._next_record_id += 1
        return record

    def decode(self, record: EncodedRecord) -> Map:
        if record.version_id not in self.versions:
            raise KeyError(f"missing dictionary version {record.version_id}")
        version = self.versions[record.version_id]
        return decode_encoding(record.kind, record.payload, record.key, list(version.entries))

    def addition_gain(self, candidate: Map, corpus: Iterable[Map]) -> int:
        candidate = tuple(candidate)
        corpus = list(dict.fromkeys(tuple(x) for x in corpus))
        if candidate not in corpus:
            raise ValueError("dictionary candidates must be observed in the corpus")
        if candidate in self.active.entries or len(self.active.entries) >= self.max_entries:
            return 0
        old_cost, _ = dataset_cost(corpus, list(self.active.entries), max_depth=self.max_depth)
        new_cost, _ = dataset_cost(corpus, [*self.active.entries, candidate], max_depth=self.max_depth)
        return old_cost - new_cost

    def add_candidate(self, candidate: Map, corpus: Iterable[Map], min_gain: int = 1) -> bool:
        candidate = tuple(candidate)
        if min_gain < 0:
            raise ValueError("min_gain must be nonnegative")
        if self.addition_gain(candidate, corpus) < min_gain:
            return False
        version_id = max(self.versions) + 1
        self.versions[version_id] = DictionaryVersion(
            version_id, (*self.active.entries, candidate), self.active_version)
        self.active_version = version_id
        return True

    def migrate_all_atomically(self) -> bool:
        """Stage every replacement; commit only if all round trips succeed."""
        staged: dict[int, EncodedRecord] = {}
        active_entries = list(self.active.entries)
        try:
            for record_id, old in self.records.items():
                original = self.decode(old)
                kind, payload, _ = self._encode_key(original, active_entries)
                candidate = EncodedRecord(record_id, original, self.active_version, kind, payload)
                if decode_encoding(kind, payload, original, active_entries) != original:
                    return False
                staged[record_id] = candidate
        except Exception:
            return False
        self.records = staged
        return True

    def prune_to_budget(self, corpus: Iterable[Map], budget: int,
                        migrate: bool = True) -> tuple[Map, ...]:
        """Choose a minimum-cost subset of current entries up to budget."""
        if budget < 0 or budget > self.max_entries:
            raise ValueError("invalid dictionary budget")
        corpus = list(dict.fromkeys(tuple(x) for x in corpus))
        entries = self.active.entries
        best_entries: tuple[Map, ...] = ()
        best_cost = None
        for size in range(min(budget, len(entries)) + 1):
            for subset in combinations(entries, size):
                cost, _ = dataset_cost(corpus, list(subset), max_depth=self.max_depth)
                if best_cost is None or cost < best_cost or (cost == best_cost and subset < best_entries):
                    best_cost, best_entries = cost, subset
        if best_entries != entries:
            parent_id = self.active_version
            version_id = max(self.versions) + 1
            staged_records = None
            if migrate:
                staged_records = {}
                try:
                    for record_id, old in self.records.items():
                        original = self.decode(old)
                        kind, payload, _ = self._encode_key(original, best_entries)
                        if decode_encoding(kind, payload, original, list(best_entries)) != original:
                            return entries
                        staged_records[record_id] = EncodedRecord(
                            record_id, original, version_id, kind, payload)
                except Exception:
                    return entries
            self.versions[version_id] = DictionaryVersion(version_id, best_entries, parent_id)
            self.active_version = version_id
            if staged_records is not None:
                self.records = staged_records
        return best_entries

    def audit(self) -> dict:
        failures = []
        for record_id, record in self.records.items():
            try:
                if self.decode(record) != record.key:
                    failures.append(record_id)
            except Exception:
                failures.append(record_id)
        return {"record_count": len(self.records), "version_count": len(self.versions),
                "active_version": self.active_version, "roundtrip_failures": failures,
                "roundtrip_rate": 1.0 if not self.records else 1.0 - len(failures) / len(self.records)}
