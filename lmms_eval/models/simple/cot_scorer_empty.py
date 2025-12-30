from typing import Dict, List, Tuple

import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils import FrameCoTScorer


@register_model("cot_scorer_empty")
class CoTScorerEmpty(lmms):
    """
    Empty model that runs FrameCoTScorer to cache reasoning/score per task.
    """

    def __init__(
        self,
        *,
        scorer_type: int = 2,
        reasoner_name: str,
        scorer_name: str,
        candidates: int = 8,
        device: str = "cuda:0",
        cache_root: str = "./score_cache",
        use_cache: bool = True,
        reasoner_max_new_tokens: int = 512,
        scorer_max_new_tokens: int = 16,
        temperature: float = 0.1,
        llava_conv_template: str = "qwen_1_5",
        llava_attn_implementation: str = "eager",
        debug_print: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            eval_logger.warning(f"Ignoring unexpected kwargs: {list(kwargs.keys())}")

        self.scorer_type = scorer_type
        self.reasoner_name = reasoner_name
        self.scorer_name = scorer_name
        self.candidates = candidates
        self.device = device
        self.cache_root = cache_root
        self.use_cache = use_cache
        self.reasoner_max_new_tokens = reasoner_max_new_tokens
        self.scorer_max_new_tokens = scorer_max_new_tokens
        self.temperature = temperature
        self.llava_conv_template = llava_conv_template
        self.llava_attn_implementation = llava_attn_implementation
        self.debug_print = debug_print
        self._scorers: Dict[str, FrameCoTScorer] = {}

    def _get_scorer(self, task_name: str) -> FrameCoTScorer:
        if task_name in self._scorers:
            return self._scorers[task_name]
        scorer = FrameCoTScorer(
            scorer_type=self.scorer_type,
            reasoner_name=self.reasoner_name,
            scorer_name=self.scorer_name,
            candidates=self.candidates,
            task_name=task_name,
            cache_root=self.cache_root,
            use_cache=self.use_cache,
            device=self.device,
            reasoner_max_new_tokens=self.reasoner_max_new_tokens,
            scorer_max_new_tokens=self.scorer_max_new_tokens,
            temperature=self.temperature,
            llava_conv_template=self.llava_conv_template,
            llava_attn_implementation=self.llava_attn_implementation,
            debug_print=self.debug_print,
        )
        self._scorers[task_name] = scorer
        return scorer

    def _normalize_query(self, query) -> str:
        if isinstance(query, (list, tuple)):
            return "\n".join(str(item) for item in query)
        return "" if query is None else str(query)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="CoT Scoring")
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            query = self._normalize_query(contexts)
            doc = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(doc)

            video_path = None
            if isinstance(visuals, str):
                video_path = visuals
            elif isinstance(visuals, list) and visuals:
                if isinstance(visuals[0], str):
                    video_path = visuals[0]

            if video_path is not None:
                scorer = self._get_scorer(task)
                scorer.score_video(video_path=video_path, query=query, query_id=str(doc_id))
            else:
                frames = visuals if isinstance(visuals, list) else [visuals]

                def _load_frames(_, num_frames):
                    processed = []
                    for frame in frames:
                        if isinstance(frame, Image.Image):
                            processed.append(np.array(frame))
                        elif isinstance(frame, np.ndarray):
                            processed.append(frame)
                    if not processed:
                        return np.empty((0,)), [], "unknown"
                    return np.stack(processed, axis=0), ["unknown"] * len(processed), "unknown"

                scorer = FrameCoTScorer(
                    scorer_type=self.scorer_type,
                    reasoner_name=self.reasoner_name,
                    scorer_name=self.scorer_name,
                    candidates=self.candidates,
                    task_name=task,
                    cache_root=self.cache_root,
                    use_cache=self.use_cache,
                    video_loader=_load_frames,
                    device=self.device,
                    reasoner_max_new_tokens=self.reasoner_max_new_tokens,
                    scorer_max_new_tokens=self.scorer_max_new_tokens,
                    temperature=self.temperature,
                    llava_conv_template=self.llava_conv_template,
                    llava_attn_implementation=self.llava_attn_implementation,
                    debug_print=self.debug_print,
                )
                scorer.score_video(video_path=f"{task}_{doc_id}", query=query, query_id=str(doc_id))

            res.append("")
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not supported for cot_scorer_empty")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        return self.generate_until(requests)
