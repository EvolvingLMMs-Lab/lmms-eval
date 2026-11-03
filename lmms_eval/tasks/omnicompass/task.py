"""Native task loading; no generated task files or separate preparation command."""

import hashlib
import json
import os

from datasets import Dataset, DatasetDict

from lmms_eval.api.task import ConfigurableMessagesTask
from lmms_eval.tasks.omnicompass.utils import protocol


class OmniCompassTask(ConfigurableMessagesTask):
    """Load paired text and text-image requests from the pinned release."""

    MODEL_NAME_AWARE = True

    def __init__(self, *args, config=None, model_name=None, **kwargs):
        config = dict(config or {})
        config.pop("class", None)
        super().__init__(*args, config=config, model_name=model_name, **kwargs)

    def download(self, dataset_kwargs: dict | None = None) -> None:
        config = dataset_kwargs or {}
        track, task = config["track"], config["task"]
        adapter = protocol()
        video_ids = None
        if os.getenv("OMNICOMPASS_VIDEO_IDS"):
            requested = {value.strip() for value in os.environ["OMNICOMPASS_VIDEO_IDS"].split(",")}
            if not requested <= set(adapter.videos):
                raise ValueError("Unknown OMNICOMPASS_VIDEO_IDS")
            video_ids = sorted(requested & {q["video_id"] for q in adapter.questions if q["track"] == track})
        docs = adapter.documents(track, (task,), video_ids=video_ids)
        self.dataset = DatasetDict({"test": Dataset.from_list(docs)})
        self.dataset_no_image = self.dataset.copy()
        self.config.metadata = {
            **(self.config.metadata or {}),
            "protocol_id": adapter.protocol_id,
            "dataset_revision": adapter.store.revision if not adapter.store.local else "local-verified",
            # doc_id is positional: bind cache fingerprints to the ordered selection.
            "selection_sha256": hashlib.sha256(json.dumps([(doc["question_id"], doc["setting"]) for doc in docs], separators=(",", ":")).encode()).hexdigest(),
        }
