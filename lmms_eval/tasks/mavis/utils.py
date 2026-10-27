"""MAVIS: multimodal source attribution in long-form visual question answering.

Paper: Song, Park and Kim, "MAVIS: A Benchmark for Multimodal Source Attribution in
Long-form Visual Question Answering", AAAI 2026. https://arxiv.org/abs/2511.12142
Reference code: https://github.com/seokwon99/MAVIS

This task implements the paper's GT-RAG setting: the model receives the input image, the
question and the human-annotated supporting documents (text passages and images, in the
fixed random order used in the paper), and writes a long-form answer that cites the
documents as [1][2]. The answer is scored by an LLM judge (GPT-4.1 in the paper):

- Groundedness: citation recall and precision of every answer sentence (paper Sec. 4.2).
- Informativeness: completeness over the annotated atomic facts and relevance of every
  sentence to the question and image (paper Sec. 4.1).

Each pair is combined into a per-answer F1 and averaged, as in the reference
`experiment/eval/summary.py`. Prompts, sentence splitting and label scoring follow the
reference implementation. Fluency (MAUVE) is not computed here.

Judge configuration (environment variables):
    MAVIS_JUDGE_API_TYPE   judge provider for lmms_eval.llm_judge (default: "openai")
    MAVIS_JUDGE_MODEL      judge model (default: "gpt-4.1", as in the paper)
    MAVIS_JUDGE_WORKERS    concurrent judge requests per answer (default: 16)
"""

import base64
import hashlib
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
from loguru import logger as eval_logger

# --------------------------------------------------------------------------------------
# Prompts (verbatim from the reference implementation)
# --------------------------------------------------------------------------------------

VANILLA_INSTRUCTION = """Based on the documents, provide a helpful answer in paragraph form. Your answer must be supported by the content in the citations.
You should cite the passage number (indices) in the format of [1][2][3] at the end of each sentence.
Do not include sentences that are not supported by the documents."""

DOC_PREFIX = "\n[{}]: "

GROUNDEDNESS_PROMPT = """Instruction:
1. You will be given a question, a statement, and an external document.
2. First, extract all subclaims within the statement that need verification.
3. Assess how well each subclaim is supported by the document.
4. Assign one of the following labels: "fully support," "partially support," or "not support."
   - If all subclaims are supported by the document, select "fully support."
   - If only some of the subclaims are supported, select "partially support."
   - If none of the subclaims are supported, select "not support."

Important:
Provide a brief explanation for your chosen level of support. The final answer should begin with "Answer: ".

Statement: {statement}
Documents: """

COMPLETENESS_PROMPT = """Instruction:
1. You will be given a fact and model response.
2. Evaluate how thoroughly the fact is addressed by the model response.
3. Assign one of the following labels:
   - Fully addressed: The fact is completely addressed in the model response. The details of the fact are fully supported by the model response.
   - Partially addressed: The fact is addressed to some extent, but important details are missing or insufficiently supported. Some details of the fact are not supported by the model response.
   - Not addressed: The fact is not addressed at all in the model response.

Important: The final answer should begin with 'Label:', and must not include any other text.

Fact: {fact}
Statement: {statement}"""

RELEVANCE_PROMPT = """Instructions:
1. You will be given a question and a statement.
2. Evaluate how the statement is related to the question.
3. Assign one of the following labels to each subclaim:
   - Fully relevant: The statement directly addresses the question.
   - Partially relevant: The statement is somewhat related to the question.
   - Not relevant: The statement is unrelated to the question.

Important:
Provide a brief explanation for your chosen level of relevance. The final label should begin with 'Label:'.

Question: """

CITATION = re.compile(r"\[([\d,\s]+)\]")
LABEL = re.compile(r"\b(fully|partially|not)\b")
METRICS = ("recall", "precision", "completeness", "relevance", "groundedness", "informativeness")
MINI_SIZE = 200

# --------------------------------------------------------------------------------------
# mavis_gt_mini
# --------------------------------------------------------------------------------------


def mini_indices(domains, qids, size=MINI_SIZE):
    """Row indices of a fixed subset with the same domain mix as the full test set.

    Each domain gets a share of `size` proportional to its size (largest remainder), filled with the
    rows whose sha256("<row index>:<qid>") is smallest, so the subset is the same on every machine.
    """
    by_domain = defaultdict(list)
    for i, domain in enumerate(domains):
        by_domain[domain].append(i)
    exact = {d: size * len(rows) / len(domains) for d, rows in by_domain.items()}
    quota = {d: int(v) for d, v in exact.items()}
    for d in sorted(exact, key=lambda d: (quota[d] - exact[d], d))[: size - sum(quota.values())]:
        quota[d] += 1

    def key(i):
        return hashlib.sha256(f"{i}:{qids[i]}".encode()).hexdigest()

    return sorted(i for d, rows in by_domain.items() for i in sorted(rows, key=key)[: quota[d]])


def mavis_mini_process_docs(dataset):
    return dataset.select(mini_indices(dataset["domain"], dataset["qid"]))


# --------------------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------------------


def _documents(doc):
    """[(doc_id, "text" | "image", text or PIL image)] in the order shown to the model."""
    images = iter(doc["document_images"])
    docs = []
    for doc_id, kind, text in zip(doc["documents_id"], doc["documents_type"], doc["documents_text"]):
        docs.append((str(doc_id), kind, next(images) if kind == "image" else text))
    return docs


def mavis_doc_to_visual(doc):
    return [doc["image"].convert("RGB")] + [content.convert("RGB") for _, kind, content in _documents(doc) if kind == "image"]


def mavis_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Plain-text prompt for models without interleaved inputs; images come first via doc_to_visual."""
    parts = [f"{VANILLA_INSTRUCTION}\n\nQuestion: {doc['Q']}\nDocument:"]
    image_number = 1  # image 1 is the input image
    for i, (_, kind, content) in enumerate(_documents(doc)):
        if kind == "image":
            image_number += 1
            parts.append(DOC_PREFIX.format(i + 1) + f"<image {image_number}>")
        else:
            parts.append(DOC_PREFIX.format(i + 1) + content)
    parts.append("\n\nAnswer:")
    return "".join(parts)


def mavis_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Interleaved prompt laid out as in the paper: instruction, image, question, numbered documents."""
    content = [
        {"type": "text", "text": f"{VANILLA_INSTRUCTION}\n\nQuestion: "},
        {"type": "image", "url": doc["image"].convert("RGB")},
        {"type": "text", "text": f"{doc['Q']}\nDocument:"},
    ]
    for i, (_, kind, item) in enumerate(_documents(doc)):
        if kind == "image":
            content += [{"type": "text", "text": DOC_PREFIX.format(i + 1)}, {"type": "image", "url": item.convert("RGB")}]
        else:
            content.append({"type": "text", "text": DOC_PREFIX.format(i + 1) + item})
    content.append({"type": "text", "text": "\n\nAnswer:"})
    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------------------

_JUDGE = None


def _judge():
    global _JUDGE
    if _JUDGE is None:
        from lmms_eval.llm_judge import ServerConfig, get_server

        # The paper's judge sampling: temperature 0.7, top-p 1.0, at most 1,024 tokens.
        config = ServerConfig(model_name=os.getenv("MAVIS_JUDGE_MODEL", "gpt-4.1"), temperature=0.7, top_p=1.0, max_tokens=1024)
        _JUDGE = get_server(server_name=os.getenv("MAVIS_JUDGE_API_TYPE", "openai"), config=config)
    return _JUDGE


def _ask(content):
    from lmms_eval.llm_judge import Request

    try:
        return _judge().evaluate(Request(messages=[{"role": "user", "content": content}])).content or ""
    except Exception as e:  # a failed judge call scores 0, like an unparsable label
        eval_logger.error(f"MAVIS judge call failed: {e}")
        return ""


def _ask_all(contents):
    workers = int(os.getenv("MAVIS_JUDGE_WORKERS", "16"))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(_ask, contents))


def _image_part(image):
    image = image.convert("RGB")
    image.thumbnail((512, 512))  # the paper downscales every image to <=512 px
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    # The reference judge sends every image with detail "low".
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}", "detail": "low"}}


def _label_score(response, marker):
    """First label after the last `marker`: fully -> 1, partially -> 0.5, not (or none) -> 0."""
    match = LABEL.search(response.lower().split(marker)[-1])
    return {"fully": 1.0, "partially": 0.5}.get(match.group(1), 0.0) if match else 0.0


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def _sent_tokenize(text):
    import nltk

    try:
        return nltk.tokenize.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        return nltk.tokenize.sent_tokenize(text)


def answer_sentences(answer, documents_id):
    """[(sentence without citation markers, [cited document ids])] for every sentence of the answer.

    Citation markers that sentence splitting moves to the start of the next sentence
    ("... roof. [1] Hire ...") are attached back to the sentence they close.
    """
    sentences = []
    for sentence in _sent_tokenize(answer):
        leading = re.match(r"^(\s*\[[\d,\s]+\])+", sentence)
        if leading and sentences:
            sentences[-1] += leading.group(0)
            sentence = sentence[leading.end() :]
        if sentence.strip():
            sentences.append(sentence)

    pairs = []
    for sentence in sentences:
        numbers = [int(n) for group in CITATION.findall(sentence) for n in group.split(",") if n.strip().isdigit()]
        cited = list(dict.fromkeys(str(documents_id[n - 1]) for n in numbers if 0 < n <= len(documents_id)))
        text = re.sub(r"\s+([.,;:!?])", r"\1", CITATION.sub("", sentence)).strip()
        pairs.append((text, cited))
    return pairs


def _support_content(statement, cited, id2doc):
    content = [{"type": "text", "text": GROUNDEDNESS_PROMPT.format(statement=statement)}]
    for k, doc_id in enumerate(cited):
        if k > 0:
            content.append({"type": "text", "text": "\n\n"})
        kind, item = id2doc[doc_id]
        content.append(_image_part(item) if kind == "image" else {"type": "text", "text": item})
    return content


def _f1(a, b):
    if a is None or b is None:
        return None
    return 2 * a * b / (a + b) if a + b > 0 else 0.0


def score_answer(doc, answer):
    """Per-answer recall, precision, completeness, relevance and the two F1 scores (0-1)."""
    answer = answer.split("</thinking>")[-1].strip()
    documents = _documents(doc)
    id2doc = {doc_id: (kind, item) for doc_id, kind, item in documents}
    sentences = answer_sentences(answer, [doc_id for doc_id, _, _ in documents]) if answer else []
    if not sentences:  # an empty answer covers nothing and cites nothing
        return dict.fromkeys(METRICS, 0.0)

    # One batch of judge requests per answer: (kind, sentence index, prompt content).
    requests = []
    for i, (sentence, cited) in enumerate(sentences):
        if cited:
            requests.append(("recall", i, _support_content(sentence, cited, id2doc)))
            if len(cited) > 1:  # with one citation, precision equals recall
                for doc_id in cited:
                    requests.append(("precision", i, _support_content(sentence, [doc_id], id2doc)))
        requests.append(("relevance", i, [{"type": "text", "text": RELEVANCE_PROMPT}, _image_part(doc["image"]), {"type": "text", "text": f"{doc['Q']}\nStatement: {sentence}"}]))
    for fact in doc["verifiable_facts"]:
        requests.append(("completeness", None, [{"type": "text", "text": COMPLETENESS_PROMPT.format(fact=fact, statement=answer)}]))

    responses = _ask_all([content for _, _, content in requests])

    support = {}
    relevance, completeness = [], []
    for (kind, i, _), response in zip(requests, responses):
        if kind in ("recall", "precision"):
            support.setdefault(i, {"recall": [], "precision": []})[kind].append(_label_score(response, "answer:"))
        elif kind == "relevance":
            relevance.append(_label_score(response, "label:"))
        else:
            completeness.append(_label_score(response, "label:"))

    # Recall averages over all sentences (uncited sentences score 0); precision averages over cited ones.
    recall = sum(v["recall"][0] for v in support.values()) / len(sentences)
    precision = float(np.mean([np.mean(v["precision"] or v["recall"]) for v in support.values()])) if support else 0.0
    scores = {
        "recall": recall,
        "precision": precision,
        "completeness": float(np.mean(completeness)) if completeness else None,
        "relevance": float(np.mean(relevance)),
    }
    scores["groundedness"] = _f1(scores["recall"], scores["precision"])
    scores["informativeness"] = _f1(scores["completeness"], scores["relevance"])
    return scores


def mavis_process_results(doc, results):
    scores = score_answer(doc, results[0] if results else "")
    return {f"mavis_{name}": scores[name] for name in METRICS}


def mavis_aggregate(results):
    """Mean over answers, on a 0-100 scale."""
    scored = [r for r in results if r is not None]
    return 100 * float(np.mean(scored)) if scored else 0.0
