import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "batch_watchdog.py"
SPEC = importlib.util.spec_from_file_location("batch_watchdog", MODULE_PATH)
batch_watchdog = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(batch_watchdog)


class TestBatchWatchdog(unittest.TestCase):
    def test_find_stale_heartbeats_filters_by_phase_and_age(self):
        heartbeats = [
            {"rank": 0, "phase": "chat_start", "updated_at": 10.0},
            {"rank": 1, "phase": "batch_done", "updated_at": 10.0},
            {"rank": 2, "phase": "encode_start", "updated_at": 95.0},
        ]

        stale = batch_watchdog.find_stale_heartbeats(
            heartbeats,
            now=120.0,
            timeout_seconds=20.0,
            stale_phases={"encode_start", "chat_start"},
        )

        self.assertEqual([entry["rank"] for entry in stale], [0, 2])
        self.assertGreater(stale[0]["age_seconds"], 100.0)
        self.assertGreater(stale[1]["age_seconds"], 20.0)

    def test_write_timeout_snapshot_persists_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "timeout_snapshot.json"
            stale = [{"rank": 7, "phase": "chat_start", "updated_at": 0.0, "age_seconds": 901.0}]
            all_heartbeats = stale + [{"rank": 8, "phase": "batch_done", "updated_at": 899.0}]

            batch_watchdog.write_timeout_snapshot(
                snapshot_path,
                detected_at=1000.0,
                timeout_seconds=900.0,
                stale_heartbeats=stale,
                all_heartbeats=all_heartbeats,
            )

            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["timeout_seconds"], 900.0)
            self.assertEqual(payload["stale_heartbeats"][0]["rank"], 7)
            self.assertEqual(len(payload["all_heartbeats"]), 2)


if __name__ == "__main__":
    unittest.main()
