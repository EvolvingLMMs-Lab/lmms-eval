# Neptune Task Notes

## Data Hydration Status

Neptune videos are stored in Hugging Face chunk archives (`data_chunk_01.zip` to
`data_chunk_05.zip`) and hydrated locally into `HF_HOME/neptune`.

Hydrate directly from HF chunk archives:

```bash
python - <<'PY'
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

repo = "lmms-lab/GoogleDeepMind-NEPTUNE"
hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
out_dir = hf_home / "neptune"
out_dir.mkdir(parents=True, exist_ok=True)

required = {Path(str(row["video_path"])).name for row in load_dataset(repo, "full", split="test", token=True)}
remaining = set(required)

for idx in range(1, 6):
    chunk_name = f"data_chunk_{idx:02d}.zip"
    chunk_path = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=chunk_name))
    with zipfile.ZipFile(chunk_path, "r") as zf:
        member_by_basename = {Path(member).name: member for member in zf.namelist() if not member.endswith("/")}
        hits = sorted(remaining & set(member_by_basename))
        for basename in hits:
            dst = out_dir / basename
            if dst.exists():
                continue
            with zf.open(member_by_basename[basename], "r") as src, tempfile.NamedTemporaryFile("wb", delete=False, dir=out_dir) as tmp:
                shutil.copyfileobj(src, tmp)
                tmp_name = tmp.name
            os.replace(tmp_name, dst)
        remaining -= set(hits)

print(f"hydrated={len(required) - len(remaining)} missing={len(remaining)}")
if remaining:
    print("missing:", sorted(remaining)[:20])
PY
```

## Known Missing Videos (Full Split)

As of 2026-02-23, two videos from `full/test` are still unavailable after
chunk extraction plus YouTube fallback:

- `HbmW0wlrGcA.mp4` - private YouTube video
- `oP3-Uw-tUCw.mp4` - YouTube video unavailable

## Impact

- `neptune_mma_*` and `neptune_mmh_*` can be fully hydrated and evaluated.
- `neptune_full_*` may fail if evaluation samples include either missing video.
