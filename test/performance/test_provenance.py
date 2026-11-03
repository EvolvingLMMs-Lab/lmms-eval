import hashlib
import subprocess
from dataclasses import FrozenInstanceError
from importlib import metadata as importlib_metadata
from types import SimpleNamespace

import pytest

import lmms_eval._performance.provenance as provenance
from lmms_eval._performance.json_v1 import canonical_json_bytes


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _init_git(repo):
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")


def test_source_revision_hashes_exact_tracked_and_untracked_source_bytes(tmp_path, monkeypatch):
    _init_git(tmp_path)
    (tmp_path / "lmms_eval").mkdir()
    (tmp_path / "lmms_eval" / "tracked.py").write_bytes(b"tracked\r\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (tmp_path / ".gitignore").write_text("ignored.py\n")
    _git(tmp_path, "add", "lmms_eval/tracked.py", "pyproject.toml", ".gitignore")
    _git(tmp_path, "commit", "-qm", "initial source")
    (tmp_path / "lmms_eval" / "untracked.bin").write_bytes(b"untracked\x00bytes")
    (tmp_path / "lmms_eval" / "ignored.py").write_text("ignored")

    manifest = [
        {"path": "lmms_eval/tracked.py", "sha256": hashlib.sha256(b"tracked\r\n").hexdigest()},
        {"path": "lmms_eval/untracked.bin", "sha256": hashlib.sha256(b"untracked\x00bytes").hexdigest()},
        {"path": "pyproject.toml", "sha256": hashlib.sha256(b"[project]\nname = 'test'\n").hexdigest()},
    ]
    seen = []
    real_canonical_json_bytes = canonical_json_bytes
    monkeypatch.setattr(provenance, "canonical_json_bytes", lambda value: seen.append(value) or real_canonical_json_bytes(value))

    captured = provenance.capture_baseline_provenance(tmp_path)
    expected = hashlib.sha256(b"lmms-eval/SourceTreeV1/manifest\0" + real_canonical_json_bytes(manifest)).hexdigest()

    assert seen == [manifest]
    assert captured.source_commit == _git(tmp_path, "rev-parse", "HEAD")
    assert captured.source_tree_digest == f"sha256:{expected}"


def test_source_revision_reads_unmocked_git_nul_listing(tmp_path):
    _init_git(tmp_path)
    (tmp_path / "lmms_eval").mkdir()
    (tmp_path / "lmms_eval" / "source.py").write_text("source")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    _git(tmp_path, "add", "lmms_eval/source.py", "pyproject.toml")
    _git(tmp_path, "commit", "-qm", "source")

    commit, digest = provenance._source_revision(tmp_path)

    assert commit == _git(tmp_path, "rev-parse", "HEAD")
    assert digest.startswith("sha256:")


def test_source_revision_requires_the_repository_root(tmp_path):
    _init_git(tmp_path)
    (tmp_path / "lmms_eval").mkdir()
    (tmp_path / "lmms_eval" / "source.py").write_text("source")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    _git(tmp_path, "add", "lmms_eval/source.py", "pyproject.toml")
    _git(tmp_path, "commit", "-qm", "source")

    assert provenance._source_revision(tmp_path)[0] == _git(tmp_path, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="repository root"):
        provenance._source_revision(tmp_path / "lmms_eval")


def test_source_revision_fails_closed_outside_a_git_repository(tmp_path):
    outside_repository = tmp_path / "not-a-repository"
    outside_repository.mkdir()

    with pytest.raises(subprocess.CalledProcessError):
        provenance.capture_baseline_provenance(outside_repository)


def test_source_revision_fails_closed_without_commit_or_readable_source(tmp_path):
    _init_git(tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        provenance.capture_baseline_provenance(tmp_path)

    (tmp_path / "lmms_eval").mkdir()
    (tmp_path / "lmms_eval" / "broken.py").symlink_to("missing.py")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    _git(tmp_path, "add", "lmms_eval/broken.py", "pyproject.toml")
    _git(tmp_path, "commit", "-qm", "broken source")
    with pytest.raises(FileNotFoundError):
        provenance.capture_baseline_provenance(tmp_path)

    (tmp_path / "lmms_eval" / "broken.py").unlink()
    external = tmp_path.parent / f"{tmp_path.name}-external.py"
    external.write_bytes(b"outside repository")
    (tmp_path / "lmms_eval" / "external.py").symlink_to(external)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "external source link")
    with pytest.raises(ValueError, match="escapes repository root"):
        provenance.capture_baseline_provenance(tmp_path)


def test_environment_digest_hashes_sorted_casefolded_installed_distributions(monkeypatch):
    distributions = [
        SimpleNamespace(metadata={"Name": "Zoo"}, version="2.0RC1"),
        SimpleNamespace(metadata={"Name": "Alpha"}, version="1.0"),
    ]
    monkeypatch.setattr(importlib_metadata, "distributions", lambda: distributions)
    expected = hashlib.sha256(b"alpha==1.0\nzoo==2.0rc1").hexdigest()
    assert provenance._environment_lock_digest() == f"sha256:{expected}"


@pytest.mark.parametrize(("platform_name", "expected"), [("linux", 3 * 1024), ("darwin", 3)])
def test_peak_host_rss_normalizes_platform_units(monkeypatch, platform_name, expected):
    fake_resource = SimpleNamespace(RUSAGE_SELF=1, getrusage=lambda scope: SimpleNamespace(ru_maxrss=3))
    monkeypatch.setattr(provenance, "_resource", fake_resource, raising=False)
    monkeypatch.setattr(provenance.sys, "platform", platform_name)
    assert provenance.peak_host_rss_bytes() == expected


def test_peak_host_rss_is_unavailable_when_resource_module_is_missing(monkeypatch):
    monkeypatch.setattr(provenance, "_resource", None, raising=False)
    assert provenance.peak_host_rss_bytes() is None


def test_capture_baseline_provenance_is_frozen_and_contains_all_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "_source_revision", lambda repo_root: ("commit", "tree"))
    monkeypatch.setattr(provenance, "_environment_lock_digest", lambda: "environment")
    monkeypatch.setattr(provenance, "_hardware_description", lambda: "hardware")

    captured = provenance.capture_baseline_provenance(tmp_path)

    assert captured == provenance.BaselineProvenance("commit", "tree", "environment", "hardware")
    with pytest.raises(FrozenInstanceError):
        captured.hardware = "other"
