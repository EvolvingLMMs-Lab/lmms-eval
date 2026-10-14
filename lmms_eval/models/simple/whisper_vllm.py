import copy
import importlib
import re
from typing import Any, List, Optional, Sequence, Tuple, Union

from tqdm import tqdm
from vllm import LLM, SamplingParams

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# HuggingFace Whisper generate_with_fallback defaults. Used when vLLM exposes
# WhisperLLM (vllm-project/vllm#58274). Stock LLM.generate ignores these kwargs.
_HF_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_HF_COMPRESSION_RATIO_THRESHOLD = 1.35
_HF_LOGPROB_THRESHOLD = -1.0

# LibriSpeech-only duration-retry when the first pass emitted no timestamps.
# English speech is ~2 w/s. Not in HF generate_with_fallback (HF skips silence).
_MIN_RETRY_DURATION_S = 2.0
_EXPECTED_WPS = 2.0
_SHORT_FRACTION = 0.5
_KEEP_RECOVERED_WPS = 0.5
_TOKENS_PER_SEC = 1.5

# LibriSpeech unread-tail (HF long-form seek analog). vLLM LLM.generate has no
# mel-frame cursor; STT serving vllm-project/vllm#53145 is a different path.
_TS_RE = re.compile(r"<\|(\d+\.\d+)\|>")
_MIN_GAP_S = 2.0
_PREROLL_S = 0.5
_MIN_DENSITY_TAIL_S = 4.0
_MAX_TAIL_PASSES = 3
_LIBRI_TS_PROMPT = "<|startoftranscript|><|en|><|transcribe|>"


def _is_librispeech_task(task: str) -> bool:
    return str(task).strip().startswith("librispeech")


def _duration_s(audio: dict) -> float:
    arr, sr = audio["array"], float(audio["sampling_rate"])
    return len(arr) / max(sr, 1.0)


def _too_short(text: str, duration_s: float) -> bool:
    n = len(text.split())
    if duration_s < _MIN_RETRY_DURATION_S:
        return n == 0
    if n == 0:
        return True
    return n < _SHORT_FRACTION * duration_s * _EXPECTED_WPS


def _keep_retry(old: str, new: str, duration_s: float) -> bool:
    nw, ow = new.split(), old.split()
    if len(nw) <= len(ow):
        return False
    if duration_s > 0 and (len(nw) / duration_s) < _KEEP_RECOVERED_WPS:
        return False
    return True


def _strip_ts(text: str) -> str:
    return " ".join(_TS_RE.sub(" ", text or "").split())


def _last_timestamp_s(text: str, token_ids=None, tokenizer=None) -> Optional[float]:
    if token_ids is not None and tokenizer is not None:
        last = None
        for tid in token_ids:
            tok = tokenizer.convert_ids_to_tokens(int(tid))
            match = _TS_RE.fullmatch(tok) if tok else None
            if match:
                last = float(match.group(1))
        if last is not None:
            return last
    hits = [float(x) for x in _TS_RE.findall(text or "")]
    return hits[-1] if hits else None


def _slice_audio(audio: dict, start_s: float) -> dict:
    arr, sr = audio["array"], float(audio["sampling_rate"])
    i = max(int(start_s * sr), 0)
    return {"array": arr[i:], "sampling_rate": audio["sampling_rate"]}


def _seam_dedup(head: str, tail: str, max_overlap: int = 12) -> str:
    hw, tw = head.split(), tail.split()
    for n in range(min(max_overlap, len(hw), len(tw)), 0, -1):
        if hw[-n:] == tw[:n]:
            tw = tw[n:]
            break
    return " ".join(hw + tw)


def _keep_tail(tail_text: str, tail_dur: float) -> bool:
    n = len(tail_text.split())
    if n == 0:
        return False
    if tail_dur >= _MIN_DENSITY_TAIL_S and (n / tail_dur) < _KEEP_RECOVERED_WPS:
        return False
    return True


def _audio_prompt(audio: dict, prompt_text: str) -> dict:
    return {
        "prompt": prompt_text,
        "multi_modal_data": {"audio": (audio["array"], audio["sampling_rate"])},
    }


def _optional_whisper_llm():
    """WhisperLLM if vLLM #58274 is present; otherwise stock LLM."""
    for mod_name in (
        "vllm.entrypoints.speech_to_text.whisper",
        "vllm.entrypoints.whisper",
    ):
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, "WhisperLLM", None)
            if cls is not None:
                return cls, mod
        except ImportError:
            continue
    return LLM, None


def _generate_with_fallback(llm, prompts, sampling_params, whisper_mod=None, use_tqdm: bool = False):
    """Pass HF long-form kwargs only when WhisperLLM.generate accepts them."""
    prompts_list = prompts if isinstance(prompts, (list, tuple)) else [prompts]
    gen_kwargs: dict[str, Any] = dict(use_tqdm=use_tqdm)
    if whisper_mod is not None:
        gen_kwargs.update(
            temperature=getattr(whisper_mod, "TEMPERATURES", _HF_TEMPERATURES),
            compression_ratio_threshold=getattr(whisper_mod, "COMPRESSION_RATIO_THRESHOLD", _HF_COMPRESSION_RATIO_THRESHOLD),
            logprob_threshold=getattr(whisper_mod, "LOGPROB_THRESHOLD", _HF_LOGPROB_THRESHOLD),
        )
    try:
        return llm.generate(prompts_list, sampling_params, **gen_kwargs)
    except TypeError:
        return llm.generate(prompts_list, sampling_params, use_tqdm=use_tqdm)


@register_model("whisper_vllm")
class WhisperVllm(lmms):
    """Whisper via vLLM.

    LibriSpeech recovers early-EOT (empty / short / mid-clause prefix). Fleurs
    and other tasks keep the original one-shot prompts.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: Optional[Union[int, str]] = 1,
        **model_kwargs,
    ) -> None:
        super().__init__()
        self._batch_size = int(batch_size)
        llm_cls, self._whisper_mod = _optional_whisper_llm()
        self._model = llm_cls(
            model=pretrained,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **model_kwargs,
        )

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        raise NotImplementedError()

    @property
    def tokenizer(self):
        return self._model.get_tokenizer()

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Whisper")

    def flatten(self, input):  # noqa: A002 - Preserve the existing keyword argument name.
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _generate(self, prompts, sampling_params):
        return _generate_with_fallback(self.model, prompts, sampling_params, whisper_mod=self._whisper_mod, use_tqdm=False)

    def _recover_unread_tails(
        self,
        transcriptions: List[str],
        outputs,
        durations: Sequence[float],
        audios: Sequence[dict],
        recover: Sequence[bool],
        sampling_params,
        tokenizer,
    ) -> Tuple[List[str], List[Optional[float]]]:
        covered: List[Optional[float]] = []
        for i, text in enumerate(transcriptions):
            ids = list(outputs[i].outputs[0].token_ids)
            covered.append(_last_timestamp_s(text, ids, tokenizer))

        for _ in range(_MAX_TAIL_PASSES):
            retry_idx, retry_prompts, retry_starts = [], [], []
            for i, (dur, cov) in enumerate(zip(durations, covered)):
                if not recover[i] or cov is None or (dur - cov) < _MIN_GAP_S:
                    continue
                start = max(cov - _PREROLL_S, 0.0)
                tail_audio = _slice_audio(audios[i], start)
                if len(tail_audio["array"]) == 0:
                    covered[i] = dur
                    continue
                retry_idx.append(i)
                retry_starts.append(start)
                retry_prompts.append(_audio_prompt(tail_audio, _LIBRI_TS_PROMPT))
            if not retry_idx:
                break
            tail_out = self._generate(retry_prompts, sampling_params)
            for j, i in enumerate(retry_idx):
                tail_raw = tail_out[j].outputs[0].text
                tail_txt = _strip_ts(tail_raw)
                tail_dur = durations[i] - retry_starts[j]
                if not _keep_tail(tail_txt, tail_dur):
                    covered[i] = durations[i]
                    continue
                transcriptions[i] = _seam_dedup(_strip_ts(transcriptions[i]), tail_txt)
                t2 = _last_timestamp_s(tail_raw, list(tail_out[j].outputs[0].token_ids), tokenizer)
                covered[i] = durations[i] if t2 is None else retry_starts[j] + t2
        return transcriptions, covered

    def _retry_short_hyps(
        self,
        transcriptions: List[str],
        batched_prompts: Sequence[dict],
        durations: Sequence[float],
        covered: Sequence[Optional[float]],
        recover: Sequence[bool],
        sampling_params,
    ) -> List[str]:
        retry_idx, retry_prompts, retry_sp = [], [], []
        for i, (text, dur, cov) in enumerate(zip(transcriptions, durations, covered)):
            if not recover[i] or cov is not None:
                continue
            if not _too_short(_strip_ts(text), dur):
                continue
            sp = sampling_params.clone() if hasattr(sampling_params, "clone") else copy.copy(sampling_params)
            floor = int(dur * _TOKENS_PER_SEC)
            sp.min_tokens = min(max(floor, 1), sp.max_tokens or 256)
            retry_idx.append(i)
            retry_prompts.append(batched_prompts[i])
            retry_sp.append(sp)

        if not retry_idx:
            return transcriptions

        groups: dict = {}
        for j, sp in enumerate(retry_sp):
            groups.setdefault(sp.min_tokens, []).append(j)
        retry_out = [None] * len(retry_idx)
        for js in groups.values():
            outs = self._generate([retry_prompts[j] for j in js], retry_sp[js[0]])
            for k, j in enumerate(js):
                retry_out[j] = outs[k]
        for j, i in enumerate(retry_idx):
            new = retry_out[j].outputs[0].text
            if _keep_retry(_strip_ts(transcriptions[i]), _strip_ts(new), durations[i]):
                transcriptions[i] = new
        return transcriptions

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batched_requests = [requests[i : i + self.batch_size] for i in range(0, len(requests), self.batch_size)]
        for batch_requests in batched_requests:
            batched_prompts = []
            durations = []
            audios = []
            recover = []
            sampling_params = None
            for idx in range(len(batch_requests)):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx].arguments

                sampling_params = SamplingParams(
                    temperature=gen_kwargs.get("temperature", 0),
                    top_p=gen_kwargs.get("top_p", 0),
                    max_tokens=gen_kwargs.get("max_new_tokens", 256),
                )

                audio = doc_to_visual(self.task_dict[task][split][doc_id])
                assert len(audio) == 1
                audio = audio[0]

                pre_prompt = gen_kwargs.get("pre_prompt", "")
                post_prompt = gen_kwargs.get("post_prompt", "")

                task_name = str(task).strip()
                use_libri_recover = _is_librispeech_task(task_name)
                if task_name.startswith("fleurs"):
                    language = self.task_dict[task][split][doc_id]["language"]
                    language_to_token = {"Mandarin Chinese": "zh", "Cantonese Chinese": "zh", "English": "en"}
                    if language in language_to_token:
                        token = language_to_token[language]
                        prompt_text = f"{pre_prompt}<|startoftranscript|><|{token}|><|transcribe|><|notimestamps|>{post_prompt}"
                    else:
                        prompt_text = f"{pre_prompt}Please recognize the speech and only output the recognized content:{post_prompt}"
                elif use_libri_recover:
                    prompt_text = _LIBRI_TS_PROMPT
                else:
                    prompt_text = "<|startoftranscript|>"

                batched_prompts.append(_audio_prompt(audio, prompt_text))
                durations.append(_duration_s(audio))
                audios.append(audio)
                recover.append(use_libri_recover)

            outputs = self._generate(batched_prompts, sampling_params)
            transcriptions = [output.outputs[0].text for output in outputs]
            tok = self.model.get_tokenizer()
            transcriptions, covered = self._recover_unread_tails(transcriptions, outputs, durations, audios, recover, sampling_params, tok)
            transcriptions = self._retry_short_hyps(transcriptions, batched_prompts, durations, covered, recover, sampling_params)
            answers = [tok.normalize(_strip_ts(t)) for t in transcriptions]

            assert len(answers) == len(batch_requests)
            res.extend(answers)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
