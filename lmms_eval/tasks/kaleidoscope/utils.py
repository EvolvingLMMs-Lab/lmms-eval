"""Utilities for Kaleidoscope, a massively multilingual multimodal exam benchmark.

Paper: https://arxiv.org/abs/2504.07072
Code:  https://github.com/israfelsr/kaleidoscope
Data:  https://huggingface.co/datasets/CohereLabs/kaleidoscope

Prompt construction, answer extraction and metric definitions mirror the
reference implementation so that scores stay comparable with the paper.  The
two prompting regimes reported in the paper are both available:

``direct``
    English JSON-format system message plus in-language ``Question``/``Options``
    keywords.  Used for the open-weight models in the paper.
``cot``
    In-language zero-shot chain-of-thought system message; the answer is read
    back from ``<ANSWER> X </ANSWER>`` tags.  Used for the closed models.

See ``README.md`` in this directory for the deviations from the reference code.
"""

import ast
import io
import json
import os
import re
import threading
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import datasets
import yaml
from loguru import logger as eval_logger
from PIL import Image

HF_DATASET_REPO = "CohereLabs/kaleidoscope"
HF_IMAGE_ARCHIVE = "data.zip"
# Members inside data.zip are prefixed with the archive's top-level folder,
# i.e. ``final_data/<value of the "image" column>``.
ZIP_ROOT_PREFIX = "final_data/"

with open(Path(__file__).parent / "_default_template_yaml", "r", encoding="utf-8") as f:
    raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
    config = yaml.safe_load("".join(safe_data))

# ---------------------------------------------------------------------------
# Benchmark constants (kept verbatim from the reference implementation)
# ---------------------------------------------------------------------------

LANGUAGES: Dict[str, str] = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "lt": "Lithuanian",
    "ne": "Nepali",
    "nl": "Dutch; Flemish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sr": "Serbian",
    "te": "Telugu",
    "uk": "Ukrainian",
}

RESOURCE_LEVEL: Dict[str, str] = {
    "Arabic": "High",
    "Croatian": "High",
    "Dutch; Flemish": "High",
    "English": "High",
    "French": "High",
    "German": "High",
    "Hindi": "High",
    "Hungarian": "High",
    "Persian": "High",
    "Portuguese": "High",
    "Russian": "High",
    "Serbian": "High",
    "Spanish": "High",
    "Bengali": "Mid/Low",
    "Lithuanian": "Mid/Low",
    "Nepali": "Mid/Low",
    "Telugu": "Mid/Low",
    "Ukrainian": "Mid/Low",
}

LATIN_SCRIPT: Dict[str, str] = {
    "Croatian": "Latin",
    "Dutch; Flemish": "Latin",
    "English": "Latin",
    "French": "Latin",
    "German": "Latin",
    "Hungarian": "Latin",
    "Lithuanian": "Latin",
    "Portuguese": "Latin",
    "Spanish": "Latin",
    "Arabic": "Non-Latin",
    "Bengali": "Non-Latin",
    "Hindi": "Non-Latin",
    "Nepali": "Non-Latin",
    "Persian": "Non-Latin",
    "Russian": "Non-Latin",
    "Serbian": "Non-Latin",
    "Telugu": "Non-Latin",
    "Ukrainian": "Non-Latin",
}

# In-language words for "Question" / "Options" / "Answer" used to lay out the prompt.
KEYWORDS: Dict[str, Dict[str, str]] = {
    "en": {"question": "Question", "options": "Options", "answer": "Answer"},
    "es": {"question": "Pregunta", "options": "Opciones", "answer": "Respuesta"},
    "hi": {"question": "प्रश्न", "options": "विकल्प", "answer": "उत्तर"},
    "hu": {"question": "Kérdés", "options": "Lehetőségek", "answer": "Válasz"},
    "hr": {"question": "Pitanje", "options": "Opcije", "answer": "Odgovor"},
    "uk": {"question": "Питання", "options": "Варіанти", "answer": "Відповідь"},
    "pt": {"question": "Pergunta", "options": "Opções", "answer": "Resposta"},
    "bn": {"question": "প্রশ্ন", "options": "বিকল্প", "answer": "উত্তর"},
    "te": {"question": "ప్రశ్న", "options": "ఎంపికలు", "answer": "సమాధానం"},
    "ne": {"question": "प्रश्न", "options": "विकल्पहरू", "answer": "उत्तर"},
    "sr": {"question": "Pitanje", "options": "Opcije", "answer": "Odgovor"},
    "nl": {"question": "Vraag", "options": "Opties", "answer": "Antwoord"},
    "ar": {"question": "السؤال", "options": "الخيارات", "answer": "الإجابة"},
    "ru": {"question": "Вопрос", "options": "Варианты", "answer": "Ответ"},
    "fr": {"question": "Question", "options": "Options", "answer": "Réponse"},
    "fa": {"question": "سؤال", "options": "گزینه‌ها", "answer": "پاسخ"},
    "de": {"question": "Frage", "options": "Optionen", "answer": "Antwort"},
    "lt": {"question": "Klausimas", "options": "Pasirinkimai", "answer": "Atsakymas"},
}

# Zero-shot chain-of-thought system message, translated into every evaluation language.
INSTRUCTIONS_COT: Dict[str, str] = {
    "en": "The following is a multiple-choice question. Think step by step and then provide your FINAL answer between the tags <ANSWER> X </ANSWER> where X is ONLY the correct letter of your choice. Do not write additional text between the tags.",
    "es": "Lo siguiente es una pregunta de opción múltiple. Piensa paso a paso y luego proporciona tu RESPUESTA FINAL entre las etiquetas <ANSWER> X </ANSWER>, donde X es ÚNICAMENTE la letra correcta de tu elección. No escribas texto adicional entre las etiquetas.",
    "hi": "निम्नलिखित एक बहुविकल्पीय प्रश्न है। चरणबद्ध सोचें और फिर <ANSWER> X </ANSWER> टैग के बीच अपना अंतिम उत्तर प्रदान करें, जहाँ X केवल आपके चयन का सही अक्षर है। टैग के बीच अतिरिक्त कोई पाठ न लिखें।",
    "hu": "A következő egy feleletválasztós kérdés. Gondolkodj lépésről lépésre, majd add meg a VÉGSŐ válaszodat a <ANSWER> X </ANSWER> címkék között, ahol X CSAK a választott helyes betű. Ne írj további szöveget a címkék közé.",
    "hr": "Sljedeće je pitanje s višestrukim izborom. Razmislite korak po korak, a zatim dajte svoj ZAVRŠNI odgovor između oznaka <ANSWER> X </ANSWER> gdje je X SAMO ispravno slovo vašeg izbora. Nemojte pisati dodatni tekst između oznaka.",
    "uk": "Наступне — це питання з множинним вибором. Думайте крок за кроком, а потім надайте вашу ОСТАННЮ відповідь між тегами <ANSWER> X </ANSWER>, де X — ЛИШЕ правильна літера за вашим вибором. Не пишіть додаткового тексту між тегами.",
    "pt": "A seguir, temos uma questão de múltipla escolha. Pense passo a passo e depois forneça sua RESPOSTA FINAL entre as tags <ANSWER> X </ANSWER>, onde X é SOMENTE a letra correta da sua escolha. Não escreva texto adicional entre as tags.",
    "bn": "নিম্নলিখিতটি একটি বহু-বিকল্প প্রশ্ন। ধাপে ধাপে চিন্তা করুন এবং তারপর <ANSWER> X </ANSWER> ট্যাগের মধ্যে আপনার চূড়ান্ত উত্তর প্রদান করুন, যেখানে X শুধুমাত্র আপনার পছন্দের সঠিক অক্ষর। ট্যাগগুলির মধ্যে অতিরিক্ত কোনো লেখা লিখবেন না।",
    "te": "కింద ఇచ్చినది ఒక బహుళ ఎంపిక ప్రశ్న. దశల వారీగా ఆలోచించి, <ANSWER> X </ANSWER> ట్యాగ్లలో మీ తుది సమాధానాన్ని ఇవ్వండి, ఇక్కడ X మీ ఎంపికలోని సరైన అక్షరం మాత్రమే. ట్యాగ్లలో అదనపు వచనం రాయవద్దు.",
    "ne": "तलको प्रश्न बहुविकल्पीय छ। चरणबद्ध सोच्नुहोस् र त्यसपछि <ANSWER> X </ANSWER> ट्यागहरूबीच आफ्नो अन्तिम उत्तर प्रदान गर्नुहोस्, जहाँ X केवल तपाईंको रोजाइको सही अक्षर हो। ट्यागहरूबीच अतिरिक्त पाठ नलेख्नुहोस्।",
    "sr": "Sledeće je pitanje sa višestrukim izborom. Razmislite korak po korak, a zatim dajte svoj KONAČNI odgovor između oznaka <ANSWER> X </ANSWER>, gde je X SAMO tačno slovo vašeg izbora. Nemojte pisati dodatni tekst između oznaka.",
    "nl": "Het volgende is een meerkeuzevraag. Denk stap voor stap na en geef dan je UITEINDLIJKE antwoord tussen de tags <ANSWER> X </ANSWER>, waarbij X ALLEEN de juiste letter van je keuze is. Schrijf geen extra tekst tussen de tags.",
    "ar": "التالي هو سؤال اختيار من متعدد. فكر خطوة بخطوة ثم قدم إجابتك النهائية بين الوسوم <ANSWER> X </ANSWER> حيث X هي الحرف الصحيح فقط من اختيارك. لا تكتب نصًا إضافيًا بين الوسوم.",
    "ru": "Следующее — это вопрос с выбором ответа. Думайте шаг за шагом, а затем предоставьте ваш ОКОНЧАТЕЛЬНЫЙ ответ между тегами <ANSWER> X </ANSWER>, где X — ТОЛЬКО правильная буква вашего выбора. Не пишите дополнительный текст между тегами.",
    "fr": "Ce qui suit est une question à choix multiple. Réfléchissez étape par étape, puis donnez votre RÉPONSE FINALE entre les balises <ANSWER> X </ANSWER>, où X est UNIQUEMENT la lettre correcte de votre choix. N'écrivez pas de texte supplémentaire entre les balises.",
    "fa": "متن زیر یک سوال چندگزینه‌ای است. مرحله به مرحله فکر کنید و سپس پاسخ نهایی خود را بین تگ‌های <ANSWER> X </ANSWER> قرار دهید، جایی که X تنها حرف صحیح انتخاب شماست. متن اضافی بین تگ‌ها ننویسید.",
    "de": "Im Folgenden ist eine Multiple-Choice-Frage. Denken Sie Schritt für Schritt nach und geben Sie dann Ihre ENDGÜLTIGE Antwort zwischen den Tags <ANSWER> X </ANSWER> an, wobei X NUR der korrekte Buchstabe Ihrer Wahl ist. Schreiben Sie keinen zusätzlichen Text zwischen den Tags.",
    "lt": "Toliau pateikiamas klausimas su keliomis pasirinkimo galimybėmis. Mąstykite žingsnis po žingsnio ir pateikite savo GALUTINĮ atsakymą tarp žymų <ANSWER> X </ANSWER>, kur X yra TIK teisinga jūsų pasirinkta raidė. Nerašykite jokio papildomo teksto tarp žymų.",
}

# English-only "direct answer" system message, used for the open-weight models.
# The reference implementation writes ``\n\ONLY`` here; the stray backslash is an
# escape-sequence typo (see the fidelity notes in README.md) and is dropped.
SYS_MESSAGE_DIRECT = 'You are a helpful assistant who answers multiple-choice questions. For each question, output your final answer in JSON format with the following structure:\n\n{"choice": "The correct option (e.g., A, B, C, or D)"}\n\nONLY output this format exactly. Do not include any additional text or explanations outside the JSON structure.'
INSTRUCTION_DIRECT = "Output your choice in the specified JSON format."

# Choice letters that models sometimes emit in the script of the question language.
MAP_NON_LATIN: Dict[str, str] = {
    "Б": "B",  # Ukrainian
    "Ц": "C",  # Ukrainian
    "Д": "D",  # Ukrainian
    "Г": "D",
    "أ": "A",  # Arabic (Alif)
    "ب": "B",  # Arabic (Ba)
    "ج": "C",  # Arabic (Jeem)
    "د": "D",  # Arabic (Dal)
}

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_ANSWER_TAG_RE = re.compile(r"<ANSWER>\s*([^\s<>]+)\s*</ANSWER>", re.IGNORECASE)
_CHOICE_JSON_RE = re.compile(r"\{\s*\"choice\"\s*:\s*.*?\s*\}", re.DOTALL)

# ---------------------------------------------------------------------------
# Image resolution
#
# The parquet table only stores *relative paths* (e.g.
# ``data/GATE_2022_Multimodal/images/xl_question_11.png``); the pixels live in a
# companion ``data.zip`` on the Hub.  Three resolution strategies are supported:
#
#   1. ``KALEIDOSCOPE_DATA_ROOT=/path/to/extracted`` - a directory that already
#      contains ``data/...`` (i.e. the contents of ``final_data/``).
#   2. default - download ``data.zip`` once through the HF cache and read
#      members straight out of the archive (no extraction, no second copy).
#   3. ``KALEIDOSCOPE_STREAM_ZIP=1`` - read members over HTTP range requests
#      without downloading the ~1 GB archive.  Handy for smoke tests.
# ---------------------------------------------------------------------------

_zip_lock = threading.Lock()
_zip_handle: Optional[zipfile.ZipFile] = None
_zip_names: Optional[frozenset] = None


def _data_root() -> Optional[str]:
    """Return a local directory holding the extracted images, if configured."""
    root = os.getenv("KALEIDOSCOPE_DATA_ROOT")
    if not root:
        return None
    if not os.path.isdir(root):
        raise FileNotFoundError(f"KALEIDOSCOPE_DATA_ROOT={root!r} is not a directory")
    return root


def _image_size() -> Optional[int]:
    """Return the square edge length images are resized to, or ``None`` for native."""
    raw = os.getenv("KALEIDOSCOPE_IMAGE_SIZE")
    if raw is None:
        raw = config.get("metadata", {}).get("image_size", 512)
    if raw in (None, "", "none", "None", 0, "0"):
        return None
    return int(raw)


def _get_zip() -> Tuple[zipfile.ZipFile, frozenset]:
    """Open ``data.zip`` (downloading or streaming it once) and cache the handle."""
    global _zip_handle, _zip_names
    if _zip_handle is None:
        if os.getenv("KALEIDOSCOPE_STREAM_ZIP", "").lower() in ("1", "true", "yes"):
            from huggingface_hub import HfFileSystem

            eval_logger.info(f"Kaleidoscope: streaming images from {HF_DATASET_REPO}/{HF_IMAGE_ARCHIVE} over range requests")
            handle = HfFileSystem().open(f"datasets/{HF_DATASET_REPO}/{HF_IMAGE_ARCHIVE}", "rb")
        else:
            from huggingface_hub import hf_hub_download

            eval_logger.info(f"Kaleidoscope: resolving images from {HF_DATASET_REPO}/{HF_IMAGE_ARCHIVE} (~1 GB, cached by huggingface_hub)")
            handle = hf_hub_download(repo_id=HF_DATASET_REPO, filename=HF_IMAGE_ARCHIVE, repo_type="dataset")
        _zip_handle = zipfile.ZipFile(handle)
        _zip_names = frozenset(_zip_handle.namelist())
    return _zip_handle, _zip_names


def _read_bytes(relative_path: str) -> bytes:
    """Read one image from the extracted directory or from ``data.zip``."""
    root = _data_root()
    if root is not None:
        with open(os.path.join(root, relative_path), "rb") as handle:
            return handle.read()

    with _zip_lock:
        archive, names = _get_zip()
        for candidate in (ZIP_ROOT_PREFIX + relative_path, relative_path):
            if candidate in names:
                return archive.read(candidate)
    raise KeyError(f"{relative_path!r} not found in {HF_IMAGE_ARCHIVE}")


def _normalize_relative_path(value: str) -> str:
    """Reject absolute paths and ``..`` segments before touching the filesystem."""
    cleaned = value.replace("\\", "/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if cleaned.startswith("/") or os.path.isabs(cleaned) or ".." in cleaned.split("/"):
        raise ValueError(f"Kaleidoscope image path must be relative and contain no '..': {value!r}")
    return cleaned


def _load_image(value: str) -> Image.Image:
    """Load one benchmark image as RGB, resized to the configured edge length."""
    image = Image.open(io.BytesIO(_read_bytes(_normalize_relative_path(value)))).convert("RGB")
    size = _image_size()
    if size is not None:
        image = image.resize((size, size))
    return image


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------


def _options(doc: Dict[str, Any]) -> List[str]:
    """Return the option list, tolerating the stringified form some mirrors use."""
    options = doc.get("options") or []
    if isinstance(options, str):
        options = ast.literal_eval(options)
    return [str(option) for option in options]


def _is_image_option(option: str) -> bool:
    return option.lower().endswith(_IMAGE_SUFFIXES)


def _has_question_image(doc: Dict[str, Any]) -> bool:
    value = doc.get("image")
    return bool(value) and str(value).lower() != "none"


def _language_name(doc: Dict[str, Any]) -> str:
    code = doc.get("language", "")
    return LANGUAGES.get(code, code)


def _prompt_type(lmms_eval_specific_kwargs: Optional[Dict[str, Any]]) -> str:
    if not lmms_eval_specific_kwargs:
        return "direct"
    return lmms_eval_specific_kwargs.get("prompt_type", "direct")


def _system_message(doc: Dict[str, Any], prompt_type: str) -> str:
    """Return the system message for this document under the given regime."""
    if prompt_type == "cot":
        code = doc.get("language", "en")
        if code not in INSTRUCTIONS_COT:
            raise ValueError(f"No chain-of-thought instruction for language {code!r}")
        return INSTRUCTIONS_COT[code]
    return SYS_MESSAGE_DIRECT


def _question_block(doc: Dict[str, Any], prompt_type: str) -> str:
    """Render the question stem plus lettered options, following the reference code."""
    options = _options(doc)
    if prompt_type == "cot":
        # Closed-model layout: question, then a plain English "Options:" header.
        lines = [doc["question"], "Options:"]
        lines += [f"{chr(65 + index)}. {option}" for index, option in enumerate(options)]
        return "\n".join(lines)

    keyword = KEYWORDS.get(doc.get("language", "en"), KEYWORDS["en"])
    block = f"\n{INSTRUCTION_DIRECT}\n\n{keyword['question']}: {doc['question']}\n{keyword['options']}:\n"
    for index, option in enumerate(options):
        block += f"{chr(65 + index)}. {option}\n"
    return block + "\nANSWER:"


# ---------------------------------------------------------------------------
# Task interface
# ---------------------------------------------------------------------------


def kaleidoscope_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    """Return the question image followed by any options that are themselves images."""
    visuals: List[Image.Image] = []
    if _has_question_image(doc):
        visuals.append(_load_image(str(doc["image"])))
    for option in _options(doc):
        if _is_image_option(option):
            visuals.append(_load_image(option))
    return visuals


def kaleidoscope_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Build the flat prompt used by the simple (non-chat) model interface.

    Simple wrappers substitute their own system prompt, so the benchmark's system
    message is folded into the user turn here.  Chat wrappers get it as a real
    ``system`` message via :func:`kaleidoscope_doc_to_messages`.
    """
    prompt_type = _prompt_type(lmms_eval_specific_kwargs)
    return f"{_system_message(doc, prompt_type)}\n{_question_block(doc, prompt_type)}"


def kaleidoscope_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Build interleaved chat messages, placing option images next to their letters."""
    prompt_type = _prompt_type(lmms_eval_specific_kwargs)
    options = _options(doc)
    content: List[Dict[str, Any]] = []

    if prompt_type == "cot":
        # Closed-model layout: question text, question image, then the options.
        content.append({"type": "text", "text": doc["question"]})
        if _has_question_image(doc):
            content.append({"type": "image", "url": _load_image(str(doc["image"]))})
        content.append({"type": "text", "text": "Options:"})
    else:
        # Open-model layout: the question image leads, then the whole text block.
        if _has_question_image(doc):
            content.append({"type": "image", "url": _load_image(str(doc["image"]))})
        keyword = KEYWORDS.get(doc.get("language", "en"), KEYWORDS["en"])
        content.append({"type": "text", "text": f"\n{INSTRUCTION_DIRECT}\n\n{keyword['question']}: {doc['question']}\n{keyword['options']}:"})

    for index, option in enumerate(options):
        letter = chr(65 + index)
        if _is_image_option(option):
            content.append({"type": "text", "text": f"{letter}."})
            content.append({"type": "image", "url": _load_image(option)})
        else:
            content.append({"type": "text", "text": f"{letter}. {option}"})

    if prompt_type != "cot":
        content.append({"type": "text", "text": "\nANSWER:"})

    return [
        {"role": "system", "content": [{"type": "text", "text": _system_message(doc, prompt_type)}]},
        {"role": "user", "content": content},
    ]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def _letter_to_index(letter: str, num_options: int) -> Optional[int]:
    letter = MAP_NON_LATIN.get(letter, letter).strip().upper()
    if len(letter) != 1 or not ("A" <= letter <= "Z"):
        return None
    index = ord(letter) - ord("A")
    return index if 0 <= index < num_options else None


def _choice_from_json(response: str, num_options: int) -> Optional[int]:
    """Extract ``{"choice": "B"}``, mirroring ``format_answer.py`` in the reference code."""
    payload = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    candidates: List[str] = []

    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict) and isinstance(parsed.get("choice"), str):
            candidates.append(parsed["choice"])
    except (json.JSONDecodeError, TypeError):
        pass

    if not candidates:
        match = _CHOICE_JSON_RE.search(payload)
        if match:
            try:
                parsed = ast.literal_eval(match.group())
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("choice"), str):
                candidates.append(parsed["choice"])

    for candidate in candidates:
        choice = candidate.strip().upper()
        index = _letter_to_index(choice, num_options)
        if index is not None:
            return index
        # ``{"choice": "B. Ribosome"}`` - split on "." exactly like the reference
        # ``map_to_choice`` and accept only when a single segment *is* a letter.
        segments = [segment.strip() for segment in choice.split(".")]
        valid = [_letter_to_index(segment, num_options) for segment in segments]
        valid = [index for index in valid if index is not None]
        if len(valid) == 1:
            return valid[0]
    return None


def _choice_from_tags(response: str, num_options: int) -> Optional[int]:
    """Extract ``<ANSWER> X </ANSWER>``, mirroring ``format_answer`` in ``model_zoo.py``."""
    matches = _ANSWER_TAG_RE.findall(response)
    for raw in reversed(matches):
        token = raw.strip().strip(".)(").upper()
        index = _letter_to_index(token, num_options)
        if index is not None:
            return index
        if token.isdigit() and 1 <= int(token) <= num_options:
            return int(token) - 1
    return None


def extract_choice(response: str, num_options: int, prompt_type: str = "direct", lenient: bool = False) -> Optional[int]:
    """Parse a model response into a zero-based option index.

    Both documented output formats are tried regardless of ``prompt_type`` - it
    only decides which one is attempted first - because models occasionally
    answer in the other one.  ``None`` means the answer could not be extracted
    and the sample counts towards the format-error rate.

    Args:
        response: Raw model output.
        num_options: Number of options for this question.
        prompt_type: ``"direct"`` (JSON) or ``"cot"`` (``<ANSWER>`` tags).
        lenient: Also fall back to the shared loose MCQ extractor.  Off by
            default because the paper's format-error rate depends on strict
            parsing.

    Returns:
        Zero-based option index, or ``None`` when nothing could be extracted.
    """
    if not response or not response.strip():
        return None

    order = (_choice_from_tags, _choice_from_json) if prompt_type == "cot" else (_choice_from_json, _choice_from_tags)
    for extractor in order:
        index = extractor(response, num_options)
        if index is not None:
            return index

    stripped = response.strip()
    if len(stripped) == 1:
        index = _letter_to_index(stripped, num_options)
        if index is not None:
            return index
        if stripped.isdigit() and 1 <= int(stripped) <= num_options:
            return int(stripped) - 1

    if lenient:
        from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

        letter = extract_mcq_answer(response, [chr(65 + i) for i in range(num_options)])
        if letter:
            return _letter_to_index(letter, num_options)
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _lenient_extraction() -> bool:
    """Whether to fall back to the loose MCQ extractor for unparseable answers."""
    raw = os.getenv("KALEIDOSCOPE_LENIENT_EXTRACTION")
    if raw is None:
        raw = config.get("metadata", {}).get("lenient_extraction", False)
    return str(raw).lower() in ("1", "true", "yes")


def _process_results(doc: Dict[str, Any], results: Sequence[str], prompt_type: str) -> Dict[str, Dict[str, Any]]:
    """Score one document and emit the record shared by all three metrics."""
    options = _options(doc)
    prediction = extract_choice(results[0] if results else "", len(options), prompt_type, _lenient_extraction())
    valid = prediction is not None

    record = {
        "language": _language_name(doc),
        "language_code": doc.get("language", ""),
        "country": doc.get("country", ""),
        "level": doc.get("level", ""),
        "category": doc.get("category_en", ""),
        "general_category": doc.get("general_category_en", ""),
        "image_type": doc.get("image_type", ""),
        "image_information": doc.get("image_information", ""),
        "is_multimodal": _has_question_image(doc),
        "answer": doc.get("answer"),
        "prediction": prediction,
        "valid": valid,
        "correct": bool(valid and prediction == doc.get("answer")),
    }
    return {
        "kaleidoscope_acc": record,
        "kaleidoscope_valid_acc": record,
        "kaleidoscope_format_error": record,
    }


def kaleidoscope_process_results(doc: Dict[str, Any], results: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Score a ``direct`` (JSON answer format) response."""
    return _process_results(doc, results, prompt_type="direct")


def kaleidoscope_process_results_cot(doc: Dict[str, Any], results: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Score a ``cot`` (``<ANSWER>`` tagged) response."""
    return _process_results(doc, results, prompt_type="cot")


def _macro_average(records: Sequence[Dict[str, Any]], key: str, valid_only: bool) -> float:
    """Average per-group accuracy with equal weight per group, as the paper does."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record.get(key) or "unknown"].append(record)

    scores = []
    for group in groups.values():
        correct = sum(1 for record in group if record["correct"])
        denominator = sum(1 for record in group if record["valid"]) if valid_only else len(group)
        if denominator:
            scores.append(correct / denominator)
    return 100 * sum(scores) / len(scores) if scores else 0.0


def _breakdown(records: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record.get(key) or "unknown"].append(record)

    rows = []
    for name in sorted(groups):
        group = groups[name]
        correct = sum(1 for record in group if record["correct"])
        valid = sum(1 for record in group if record["valid"])
        rows.append(
            {
                "name": name,
                "num": len(group),
                "acc": 100 * correct / len(group),
                "valid_acc": 100 * correct / valid if valid else 0.0,
                "format_error": 100 * (len(group) - valid) / len(group),
            }
        )
    return rows


def _log_breakdown(title: str, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [f"{'':2}{title}", f"{'':2}{'group':<24}{'n':>7}{'acc':>9}{'valid_acc':>11}{'fmt_err':>9}"]
    for row in rows:
        lines.append(f"{'':2}{row['name'][:24]:<24}{row['num']:>7}{row['acc']:>9.2f}{row['valid_acc']:>11.2f}{row['format_error']:>9.2f}")
    eval_logger.info("\n".join(lines))


def kaleidoscope_aggregate_accuracy(results: List[Dict[str, Any]]) -> float:
    """Macro-averaged accuracy over languages, counting format errors as wrong."""
    if not results:
        return 0.0

    _log_breakdown("Kaleidoscope by language", _breakdown(results, "language"))
    _log_breakdown("Kaleidoscope by general category", _breakdown(results, "general_category"))
    if any(record["is_multimodal"] for record in results):
        multimodal = [record for record in results if record["is_multimodal"]]
        _log_breakdown("Kaleidoscope by image type (multimodal only)", _breakdown(multimodal, "image_type"))

    valid = sum(1 for record in results if record["valid"])
    eval_logger.info(
        f"Kaleidoscope overall: n={len(results)} "
        f"acc={_macro_average(results, 'language', valid_only=False):.2f} "
        f"valid_acc={_macro_average(results, 'language', valid_only=True):.2f} "
        f"format_error={100 * (len(results) - valid) / len(results):.2f}"
    )
    return _macro_average(results, "language", valid_only=False)


def kaleidoscope_aggregate_valid_accuracy(results: List[Dict[str, Any]]) -> float:
    """Macro-averaged accuracy over languages, ignoring unparseable responses."""
    return _macro_average(results, "language", valid_only=True) if results else 0.0


def kaleidoscope_aggregate_format_error(results: List[Dict[str, Any]]) -> float:
    """Share of responses (over all samples) whose answer could not be extracted."""
    if not results:
        return 0.0
    return 100 * sum(1 for record in results if not record["valid"]) / len(results)


# ---------------------------------------------------------------------------
# Split filters
# ---------------------------------------------------------------------------


def kaleidoscope_filter_multimodal(dataset: datasets.Dataset) -> datasets.Dataset:
    """Keep only questions that carry an image."""
    return dataset.filter(_has_question_image)


def kaleidoscope_filter_text_only(dataset: datasets.Dataset) -> datasets.Dataset:
    """Keep only questions that have no image."""
    return dataset.filter(lambda doc: not _has_question_image(doc))
