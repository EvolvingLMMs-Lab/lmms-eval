"""Unit and real-codec tests. Run with optional frame dependencies installed."""

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lmms_eval.tasks.omnicompass._core import extract_frames as sampler


class SamplingTests(unittest.TestCase):
    def test_avcc_validation_rejects_partial_and_invalid_nals(self):
        valid = b"\x00\x00\x00\x02\x65\x80"
        self.assertTrue(sampler.valid_avcc_packet(valid))
        self.assertTrue(sampler.valid_avcc_packet(valid * 2))
        for data in (b"", valid[:-1], valid + b"\0", b"\0\0\0\0", b"\0\0\0\x02\xe5\x80", b"\0\0\0\x02\x00\x80"):
            with self.subTest(data=data):
                self.assertFalse(sampler.valid_avcc_packet(data))

    def test_packet_recovery_is_unique_bounded_and_read_only(self):
        import av

        valid = b"\x00\x00\x00\x03\x65\x80\x55"
        raw = b"\xff" * 8 + valid + b"\xff" * 8
        for shift in (-2, -1, 1, 2):
            with self.subTest(shift=shift):
                position = 8 - shift
                packet = av.Packet(raw[position : position + len(valid)])
                packet.pts, packet.dts, packet.duration = 100, 98, 2
                packet.time_base = Fraction(1, 25)
                packet.is_keyframe = True
                # Packet.pos is demuxer-owned; proxy it while testing real update().
                proxy_type = type(
                    "PacketProxy",
                    (),
                    {
                        "__bytes__": lambda self: bytes(packet),
                        "pos": position,
                        "update": lambda self, data: packet.update(data),
                    },
                )
                reader = object.__new__(sampler.VideoReader)
                reader.packet_source = io.BytesIO(raw)
                reader.packet_corrections = {}
                reader.recover_packet(proxy_type())
                self.assertEqual(bytes(packet), valid)
                self.assertEqual((packet.pts, packet.dts, packet.duration, packet.time_base, packet.is_keyframe), (100, 98, 2, Fraction(1, 25), True))
                self.assertEqual(reader.packet_corrections, {position: shift})
                self.assertEqual(reader.packet_source.getvalue(), raw)

    def test_packet_recovery_rejects_ambiguous_and_unrecoverable_bytes(self):
        class Packet:
            pos = 4
            updated = False

            def __bytes__(self):
                return b"\xff" * 8

            def update(self, data):
                self.updated = True

        reader = object.__new__(sampler.VideoReader)
        reader.packet_source = io.BytesIO(b"\xff" * 32)
        reader.packet_corrections = {}
        packet = Packet()
        with self.assertRaisesRegex(ValueError, "unrecoverable"):
            reader.recover_packet(packet)
        with patch.object(sampler, "valid_avcc_packet", side_effect=[False, True, True, False, False]):
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                reader.recover_packet(packet)
        reader.packet_source = io.BytesIO(b"\x00" * 32)
        with self.assertRaisesRegex(ValueError, "do not match"):
            reader.recover_packet(packet)
        self.assertFalse(packet.updated)
        self.assertEqual(reader.packet_corrections, {})

    def test_invalid_packet_recovery_keeps_strict_mode_and_other_errors(self):
        import av

        frame = object()

        class Packet:
            def __init__(self, pos, error=None):
                self.pos, self.error = pos, error

            def decode(self):
                if self.error:
                    raise self.error
                return [frame]

        packets = [Packet(1, av.error.InvalidDataError(1094995529, "bad packet")), Packet(2)]
        reader = object.__new__(sampler.VideoReader)
        reader.stream = object()
        reader.packet_source = object()
        reader.failed_packets = {}
        reader.recover_packet = lambda packet: None
        reader.container = SimpleNamespace(demux=lambda stream: iter(packets))
        with self.assertRaises(av.error.InvalidDataError):
            list(reader.decoded_frames())
        self.assertEqual(list(reader.decoded_frames(skip_decode_errors=True)), [frame])
        self.assertEqual(set(reader.failed_packets), {1})
        packets[:] = [Packet(3, RuntimeError("unexpected decoder failure"))]
        with self.assertRaises(RuntimeError):
            list(reader.decoded_frames(skip_decode_errors=True))

    def test_corrupt_frames_are_rejected_or_explicitly_skipped(self):
        reader = object.__new__(sampler.VideoReader)
        reader.origin, reader.time_base, reader.stream = 0, Fraction(1), object()
        reader.packet_source = None
        reader.skipped_corrupt_pts = set()
        reader.container = SimpleNamespace(seek=lambda *args, **kwargs: None)
        frames = [
            SimpleNamespace(pts=1, time_base=Fraction(1), is_corrupt=True),
            SimpleNamespace(pts=2, time_base=Fraction(1), is_corrupt=False, to_image=lambda: "valid"),
        ]
        reader.decoded_frames = lambda skip_decode_errors=False: iter(frames)
        with self.assertRaisesRegex(ValueError, "corrupt"):
            reader.frame_at(1)
        self.assertEqual(reader.frame_at(1, skip_decode_errors=True), ("valid", 2, 2, "1"))
        self.assertEqual(reader.skipped_corrupt_pts, {1})

    def test_known_source_decode_recovery_is_recorded_and_bounded(self):
        class Reader:
            duration = 30
            packet_source = object()

            def frame_at(self, seconds, *, skip_decode_errors=False):
                if not skip_decode_errors:
                    raise ValueError("invalid packet")
                return None, seconds + 0.08, 27, "1/25"

        _, row = sampler.sample(Reader(), 1, 0, "nearest")
        self.assertTrue(row["decode_recovery"])
        self.assertEqual(row["attempts"], [])
        self.assertEqual(row["fallback_offset_sec"], 0)
        with self.assertRaises(ValueError):
            sampler.sample(Reader(), 1, 0, "fail")

    def test_sequential_recovery_is_forward_streaming_and_can_rewind(self):
        reader = object.__new__(sampler.VideoReader)
        reader.origin, reader.time_base, reader.stream = 0, Fraction(1, 25), object()
        reader.packet_source = object()
        reader.sequential_frames = reader.sequential_frame = None
        reader.sequential_target = -math.inf
        reader.skipped_corrupt_pts = set()
        seeks, decoded = [], []
        reader.container = SimpleNamespace(seek=lambda *a, **k: seeks.append(a[0]))

        def frames(skip_decode_errors=False):
            self.assertTrue(skip_decode_errors)
            for pts in range(25):
                decoded.append(pts)
                yield SimpleNamespace(pts=pts, time_base=Fraction(1, 25), duration=1, is_corrupt=pts == 3, to_image=lambda: "image")

        reader.decoded_frames = frames
        for target, expected in ((0, 0), (0.04, 1), (0.12, 4), (0.13, 4), (0.5, 13), (0.99, 24)):
            self.assertEqual(reader.frame_at(target, skip_decode_errors=True)[2], expected)
        self.assertEqual(decoded, list(range(25)))
        self.assertEqual(seeks, [0])
        self.assertEqual(reader.skipped_corrupt_pts, {3})
        self.assertEqual(reader.frame_at(0.04, skip_decode_errors=True)[2], 1)
        self.assertEqual(seeks, [0, 0])
        with self.assertRaisesRegex(ValueError, "No decodable frame"):
            reader.frame_at(1.1, skip_decode_errors=True)

    def test_fixed_budget(self):
        self.assertEqual(sampler.targets(10, nframe=4), [2, 4, 6, 8])
        self.assertEqual(sampler.targets(10, nframe=1), [5])

    def test_every_supported_frame_budget_has_an_ordered_complete_schedule(self):
        durations = (0.07, 1 / 29.97, 3.0, 1482.966667, 2425.64)
        for count in range(1, 10001):
            duration = durations[count % len(durations)]
            with self.subTest(count=count, duration=duration):
                schedule = sampler.targets(duration, nframe=count)
                self.assertEqual(len(schedule), count)
                self.assertGreater(schedule[0], 0)
                self.assertLess(schedule[-1], duration)
                self.assertTrue(all(a < b for a, b in zip(schedule, schedule[1:])))
                self.assertAlmostEqual(schedule[0], duration / (count + 1))
                self.assertAlmostEqual(schedule[-1], duration * count / (count + 1))

    def test_cadence_endpoint_and_short_video(self):
        self.assertEqual(sampler.targets(3, fps=1), [0, 1, 2])
        self.assertEqual(sampler.targets(2.1, fps=1), [0, 1, 2])
        self.assertEqual(sampler.targets(0.1, fps=1), [0])
        self.assertEqual(len(sampler.targets(0.07, fps=100)), 7)

    def test_invalid_modes_and_limits(self):
        for kwargs in ({}, {"nframe": 2, "fps": 1}, {"nframe": 0}, {"nframe": 1.5}, {"fps": 0}, {"fps": float("nan")}, {"fps": float("inf")}, {"nframe": 101, "max_frames": 100}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sampler.targets(10, **kwargs)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                sampler.targets(value, nframe=1)

    def test_cli_requires_one_mode(self):
        for flags in ([], ["--nframe", "2", "--fps", "1"], ["--fps", "nan"], ["--nframe", "0"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sampler.parser().parse_args(["--video", "x", "--cache-dir", "c", *flags])

    def test_fallback_and_fail_policy(self):
        class Reader:
            duration = 10

            def frame_at(self, time):
                if time == 4:
                    raise ValueError("bad packet")
                return None, time, int(time * 100), "1/100"

        _, record = sampler.sample(Reader(), 4, 3, "nearest")
        self.assertEqual(record["fallback_offset_sec"], -0.25)
        self.assertEqual(len(record["attempts"]), 1)
        with self.assertRaises(ValueError):
            sampler.sample(Reader(), 4, 3, "fail")
        _, record = sampler.sample(Reader(), 4, 4, "nearest")
        self.assertEqual(record["fallback_offset_sec"], 0.25)

    def test_no_silent_large_jump(self):
        class Reader:
            duration = 30

            def frame_at(self, time):
                return None, 20, 20, "1"

        with self.assertRaises(ValueError):
            sampler.sample(Reader(), 2, 0, "nearest")

    def test_previously_decoded_frame_is_only_a_bounded_last_resort(self):
        image = object()
        prior = (image, {"timestamp_sec": 3, "pts": 75, "time_base": "1/25", "decode_recovery": False})

        class Reader:
            duration = 30

            def frame_at(self, seconds):
                raise ValueError("Cannot seek into damaged GOP")

        returned, row = sampler.sample(Reader(), 4, 3, "nearest", prior)
        self.assertIs(returned, image)
        self.assertEqual(row["timestamp_sec"], 3)
        self.assertEqual((row["pts"], row["time_base"]), (75, "1/25"))
        self.assertEqual(row["fallback_offset_sec"], -1)
        self.assertTrue(row["reused_previous_frame"])
        self.assertGreater(len(row["attempts"]), 1)
        for requested, previous, policy in ((9, 3, "nearest"), (4, 3, "fail"), (4, 5, "nearest")):
            with self.subTest(requested=requested, previous=previous, policy=policy):
                with self.assertRaises(ValueError):
                    sampler.sample(Reader(), requested, previous, policy, prior)
        with patch.object(Reader, "frame_at", return_value=("new", 4, 100, "1/25")):
            returned, row = sampler.sample(Reader(), 4, 3, "nearest", prior)
        self.assertEqual(returned, "new")
        self.assertNotIn("reused_previous_frame", row)

    def test_dense_sampling_retries_between_old_offsets(self):
        class Reader:
            duration = 384

            def frame_at(self, seconds):
                if 374.5 <= seconds < 378:
                    raise ValueError("Undecodable GOP")
                pts = min(math.ceil(seconds * 25), int(self.duration * 25) - 1)
                return None, pts / 25, pts, "1/25"

        _, row = sampler.sample(Reader(), 375.5, 374.44, "nearest")
        self.assertEqual(row["fallback_offset_sec"], 2.5)
        self.assertEqual(row["timestamp_sec"], 378)
        previous = 374.44
        for i in range(850):
            requested = 375.5 + i / 100
            _, row = sampler.sample(Reader(), requested, previous, "nearest")
            self.assertGreaterEqual(row["timestamp_sec"], previous)
            self.assertLessEqual(abs(row["timestamp_sec"] - requested), 5)
            previous = row["timestamp_sec"]
        self.assertEqual(previous, 383.96)

    def test_fractional_targets_do_not_miss_valid_first_frame(self):
        class Reader:
            duration = 20

            def frame_at(self, seconds):
                if 0 < seconds < 6.5:
                    raise ValueError("Damaged first GOP")
                return None, seconds, round(seconds * 1000), "1/1000"

        for requested in (0.1482818384828184, 1.4814851814851815):
            _, row = sampler.sample(Reader(), requested, -math.inf, "nearest")
            self.assertEqual(row["timestamp_sec"], 0)
            self.assertEqual(row["fallback_offset_sec"], -requested)
        previous = -math.inf
        for i in range(1, 400):
            requested = i / 30
            _, row = sampler.sample(Reader(), requested, previous, "nearest")
            self.assertGreaterEqual(row["timestamp_sec"], previous)
            self.assertLessEqual(abs(row["timestamp_sec"] - requested), 5)
            previous = row["timestamp_sec"]
        with self.assertRaises(ValueError):
            sampler.sample(Reader(), 0.148, -math.inf, "fail")

    def test_writer_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with sampler.cache_lock(root, "key"):
                with self.assertRaises(ValueError):
                    with sampler.cache_lock(root, "key"):
                        pass
            self.assertFalse((root / ".key.lock").exists())


class CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import av
        from PIL import Image

        cls.av = av
        cls.Image = Image
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.video = cls.root / "fixture.mp4"
        cls.make_video(cls.video)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def make_video(cls, path, irregular=False, start=0):
        with cls.av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            stream.time_base = Fraction(1, 1000)
            stream.codec_context.time_base = Fraction(1, 1000)
            ticks = [0, 100, 400, 700, 1100, 1400, 1800, 2100] if irregular else range(0, 3000, 100)
            for i, time in enumerate(ticks):
                image = cls.Image.new("RGB", (64, 48), (i * 7 % 256, 80, 120))
                frame = cls.av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = time + start, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def test_real_nframe_and_cache_no_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path, result, hit = sampler.extract_video(self.video, directory, nframe=4, max_side=32)
            self.assertFalse(hit)
            self.assertEqual(len(result["frames"]), 4)
            self.assertEqual([round(f["timestamp_sec"], 2) for f in result["frames"]], [0.6, 1.2, 1.8, 2.4])
            self.assertEqual((result["frames"][0]["width"], result["frames"][0]["height"]), (32, 24))
            before = {p.name: p.stat().st_mtime_ns for p in path.iterdir()}
            with patch.object(sampler.VideoReader, "frame_at", side_effect=AssertionError("decoded")):
                same, cached, hit = sampler.extract_video(self.video, directory, nframe=4, max_side=32)
            self.assertTrue(hit)
            self.assertEqual(path, same)
            self.assertEqual(result, cached)
            self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in path.iterdir()})
            self.assertNotIn("packet_offset_recovery", result)

    def test_unknown_source_never_uses_offset_recovery(self):
        for source_hash in (None, "unknown"):
            reader = sampler.VideoReader(self.video, source_sha256=source_hash)
            try:
                self.assertIsNone(reader.packet_source)
            finally:
                reader.close()

    def test_real_h264_offset_fault_preserves_pixels_timestamps_and_source(self):
        from _omnicompass_mp4 import video_offsets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.mp4"
            with self.av.open(str(original), "w") as container:
                stream = container.add_stream("libx264", rate=10)
                stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
                for i in range(30):
                    image = self.Image.new("RGB", (64, 48), (i * 7 % 256, 80, 120))
                    for packet in stream.encode(self.av.VideoFrame.from_image(image)):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            data = bytearray(original.read_bytes())
            with original.open("rb") as handle:
                for offset, position, width in video_offsets(handle, len(data)):
                    data[position : position + width] = (offset + 1).to_bytes(width, "big")
            damaged = root / "misaligned.mp4"
            damaged.write_bytes(data)
            before = sampler.sha(damaged)
            with self.assertRaisesRegex(ValueError, "Cannot sample"):
                sampler.extract_video(damaged, root / "cache", nframe=32, on_error="fail")
            selections = [{"nframe": count} for count in (*range(1, 130), 257, 511, 512, 1024, 10000)]
            selections += [{"fps": fps} for fps in (0.1, 0.5, 1, 10, 30)]
            for selection in selections:
                with self.subTest(**selection):
                    _, expected, _ = sampler.extract_video(original, root / "cache", **selection, on_error="fail")
                    for policy in ("fail", "nearest"):
                        with patch.object(sampler, "OFFSET_RECOVERY_SOURCES", {before}):
                            _, recovered, hit = sampler.extract_video(damaged, root / "cache", **selection, on_error=policy)
                            self.assertFalse(hit)
                            _, repeated, hit = sampler.extract_video(damaged, root / "cache", **selection, on_error=policy)
                        self.assertTrue(hit)
                        self.assertEqual(repeated, recovered)
                        self.assertEqual(expected["frames"], [dict(row, decode_recovery=False) for row in recovered["frames"]])
                        self.assertEqual(recovered["fallback_frames"], 0)
                        self.assertGreater(recovered["packet_offset_recovery"]["corrected_packets"], 0)
            self.assertEqual(sampler.sha(damaged), before)

    def test_real_fps_and_no_upscale(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result, _ = sampler.extract_video(self.video, directory, fps=2)
            self.assertEqual(len(result["frames"]), 6)
            self.assertEqual([round(f["timestamp_sec"], 3) for f in result["frames"]], [0, 0.5, 1, 1.5, 2, 2.5])
            self.assertEqual((result["frames"][0]["width"], result["frames"][0]["height"]), (64, 48))

    def test_vfr_and_nonzero_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "vfr.mp4"
            self.make_video(video, irregular=True, start=5000)
            _, result, _ = sampler.extract_video(video, Path(directory) / "cache", fps=2)
            self.assertAlmostEqual(result["video"]["stream_start_seconds"], 5)
            actual = [round(f["timestamp_sec"], 3) for f in result["frames"]]
            self.assertEqual(actual, [0, 0.7, 1.1, 1.8, 2.1])
            for row in result["frames"]:
                self.assertAlmostEqual(row["timestamp_sec"], row["pts"] * float(Fraction(row["time_base"])) - 5)

    def test_cache_keys_options_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, _, _ = sampler.extract_video(self.video, root / "cache", nframe=2)
            b, _, _ = sampler.extract_video(self.video, root / "cache", fps=1)
            c, _, _ = sampler.extract_video(self.video, root / "cache", nframe=2, max_side=32)
            d, _, _ = sampler.extract_video(self.video, root / "cache", nframe=2, jpeg_quality=80)
            copied = root / "another.mp4"
            copied.write_bytes(self.video.read_bytes())
            same, _, hit = sampler.extract_video(copied, root / "cache", nframe=2)
            self.assertEqual(a, same)
            self.assertTrue(hit)
            copied.write_bytes(copied.read_bytes() + b"\x00" * 8)
            changed, _, hit = sampler.extract_video(copied, root / "cache", nframe=2)
            self.assertFalse(hit)
            self.assertEqual(len({a, b, c, d, changed}), 5)

    def test_delayed_video_keeps_shared_audio_clock(self):
        stream = SimpleNamespace(time_base=Fraction(1, 1000), start_time=2000, duration=3000, width=64, height=48, codec_context=SimpleNamespace())
        container = SimpleNamespace(streams=SimpleNamespace(video=[stream]), start_time=1000000, close=lambda: None)
        with patch.object(self.av, "open", return_value=container):
            reader = sampler.VideoReader(self.video)
        self.assertEqual(reader.origin, 1)
        self.assertEqual(reader.duration, 4)
        self.assertEqual(reader.info["stream_start_seconds"], 2)

    def test_missing_or_corrupt_cache_rejected(self):
        for damage in ("delete", "corrupt", "manifest"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                path, result, _ = sampler.extract_video(self.video, directory, nframe=2)
                image = path / result["frames"][0]["path"]
                if damage == "delete":
                    image.unlink()
                elif damage == "corrupt":
                    image.write_bytes(b"bad")
                else:
                    (path / "sampling.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "Invalid cache"):
                    sampler.extract_video(self.video, directory, nframe=2)

    def test_failure_does_not_publish_partial_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sampler.VideoReader, "frame_at", side_effect=ValueError("bad packet")):
                with self.assertRaises(ValueError):
                    sampler.extract_video(self.video, directory, nframe=2)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_duplicates_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result, _ = sampler.extract_video(self.video, directory, nframe=40)
            self.assertEqual(len(result["frames"]), 40)
            self.assertGreater(result["duplicate_frames"], 0)

    def test_cached_previous_pixels_survive_failed_reseeks(self):
        with tempfile.TemporaryDirectory() as directory:
            original = sampler.VideoReader.frame_at
            prior = []

            def fail_after_first(reader, seconds):
                if prior:
                    raise ValueError("Damaged tail")
                result = original(reader, seconds)
                prior.append(result)
                return result

            with patch.object(sampler.VideoReader, "frame_at", new=fail_after_first):
                path, result, _ = sampler.extract_video(self.video, directory, nframe=3, max_side=32)
            self.assertEqual(len(result["frames"]), 3)
            self.assertEqual(result["reused_previous_frames"], 2)
            self.assertEqual(result["duplicate_frames"], 2)
            self.assertEqual(result["fallback_frames"], 2)
            self.assertEqual(len({f["sha256"] for f in result["frames"]}), 1)
            self.assertEqual(len({f["pts"] for f in result["frames"]}), 1)
            self.assertEqual(prior[0][0].size, (64, 48))
            for frame in result["frames"]:
                with self.Image.open(path / frame["path"]) as image:
                    image.load()
                    self.assertEqual(image.size, (32, 24))

    def test_batch_filter_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "good.mp4").write_bytes(self.video.read_bytes())
            rows = [{"video_id": "oc_001", "video_path": "good.mp4", "tracks": ["VL", "AV"]}, {"video_id": "oc_002", "video_path": "missing.mp4", "tracks": ["VL"]}]
            (root / "videos.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            flags = ["--data", str(root), "--nframe", "2", "--cache-dir", str(root / "cache")]
            with redirect_stdout(io.StringIO()) as output:
                code = sampler.main([*flags, "--track", "AV"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["video_id"], "oc_001")
            with redirect_stdout(io.StringIO()) as output:
                code = sampler.main(flags)
            self.assertEqual(code, 1)
            self.assertEqual([r["status"] for r in map(json.loads, output.getvalue().splitlines())], ["ok", "error"])


if __name__ == "__main__":
    unittest.main()
