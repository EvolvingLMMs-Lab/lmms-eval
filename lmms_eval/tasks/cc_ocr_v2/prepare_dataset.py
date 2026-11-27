"""Prepare the CC-OCR V2 test split from the authors' pinned Hugging Face release.

Run python -m lmms_eval.tasks.cc_ocr_v2.prepare_dataset --output-dir PATH.
Set HF_HUB_OFFLINE=1 to reuse a complete, cached snapshot without network access.
"""

import argparse
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from huggingface_hub import snapshot_download
from PIL import Image as PILImage

REPO_ID = "Eioss/CC-OCR-V2"
REVISION = "f80226d958d748daa0bf805ed6f69c2d41823e80"
# SHA-256 of the sorted, LF-terminated relative paths of all question, answer and image files.
RELEASE_PATHS_SHA256 = "4ca70eaeda8d52ba69b0f1f0a29fd710f885bfc15afc288a6bdb164f6551bae5"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FEATURES = Features(
    {
        "source_key": Value("string"),
        "track": Value("string"),
        "sub_task": Value("string"),
        "scenario": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "images": Sequence(Image()),
        "image_widths": Sequence(Value("int32")),
        "image_heights": Sequence(Value("int32")),
    }
)


@dataclass(frozen=True)
class Sample:
    key: str
    question: Path
    answer: Path
    pages: tuple[Path, ...]


def natural_key(name: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def index_files(root: Path, kind: str, extensions: set[str]):
    index = defaultdict(list)
    for path in root.glob(f"*/*/{kind}/**/*"):
        if path.suffix.lower() not in extensions or not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if len(parts) == 5:
            stem = path.stem
        elif len(parts) == 6 and kind == "images":
            stem = parts[4]  # Multi-page document directory.
        else:
            raise ValueError(f"Unexpected {kind} path: {path}")
        track, sub_task, _, scenario = parts[:4]
        index[f"{track}/{sub_task}/{scenario}/{stem}"].append(path)
    return index


def collect_samples(root: Path) -> list[Sample]:
    questions = index_files(root, "question", {".txt"})
    answers = index_files(root, "answer", {".txt", ".json"})
    images = index_files(root, "images", IMAGE_EXTENSIONS)
    if not questions:
        raise ValueError(f"No CC-OCR V2 questions found in {root}")
    for kind, files in (("answer", answers), ("images", images)):
        unmatched = questions.keys() ^ files.keys()
        if unmatched:
            raise ValueError(f"Unmatched question/{kind} key: {min(unmatched)}")

    samples = []
    for key in sorted(questions, key=natural_key):
        if len(questions[key]) != 1 or len(answers[key]) != 1:
            raise ValueError(f"Expected one question and one answer for {key}")
        pages = tuple(sorted(images[key], key=lambda path: natural_key(path.name)))
        samples.append(Sample(key, questions[key][0], answers[key][0], pages))
    return samples


def validate_release(root: Path, samples: list[Sample]) -> None:
    files = (path for sample in samples for path in (sample.question, sample.answer, *sample.pages))
    paths = sorted(path.relative_to(root).as_posix() for path in files)
    paths_sha256 = hashlib.sha256(("\n".join(paths) + "\n").encode("utf-8")).hexdigest()
    if paths_sha256 != RELEASE_PATHS_SHA256:
        counts = f"{len(samples)} samples, {len(paths)} files"
        raise ValueError(f"Incomplete or unexpected {REPO_ID}@{REVISION} file inventory ({counts})")


def iter_rows(samples: tuple[Sample, ...]):
    for sample in samples:
        track, sub_task, scenario, _ = sample.key.split("/")
        images, widths, heights = [], [], []
        for page in sample.pages:
            with PILImage.open(page) as image, BytesIO() as buffer:
                image.convert("RGB").save(buffer, format="JPEG", quality=95, optimize=True)
                images.append({"bytes": buffer.getvalue(), "path": None})
                widths.append(image.width)
                heights.append(image.height)
        yield {
            "source_key": sample.key,
            "track": track,
            "sub_task": sub_task,
            "scenario": scenario,
            "question": sample.question.read_text(encoding="utf-8"),
            "answer": sample.answer.read_bytes().decode("utf-8"),
            "images": images,
            "image_widths": widths,
            "image_heights": heights,
        }


def save_dataset(samples: list[Sample], output_dir: Path) -> None:
    """Stream rows to a temporary Arrow cache and publish the finished DatasetDict atomically."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent, prefix=f".{output_dir.name}-") as work:
        work = Path(work)
        prepared = work / "prepared"
        with Dataset.from_generator(
            iter_rows,
            gen_kwargs={"samples": tuple(samples)},
            features=FEATURES,
            cache_dir=str(work / "arrow-cache"),
            writer_batch_size=8,
        ) as test:
            test.info.description = f"CC-OCR V2 ({REPO_ID}@{REVISION})"
            DatasetDict({"test": test}).save_to_disk(str(prepared))
        prepared.rename(output_dir)


def prepare_dataset(output_dir: Path, cache_dir: Path | None = None) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    root = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            cache_dir=cache_dir,
            allow_patterns=["*/*/question/*", "*/*/answer/*", "*/*/images/*"],
        )
    )
    samples = collect_samples(root)
    validate_release(root, samples)
    save_dataset(samples, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the prepared DatasetDict")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face Hub cache for the pinned source snapshot")
    args = parser.parse_args()
    prepare_dataset(args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
