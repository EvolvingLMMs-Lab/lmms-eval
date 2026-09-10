"""MET-Bench prompts and scoring, shared by the evaluation integrations.

Adapted from https://github.com/vanyacohen/MET-Bench (MIT License).
"""

import io
import math
import re
from typing import Any

from PIL import Image


def _image(value: Any) -> Image.Image:
    """Accept decoded dataset images or their Arrow bytes/path representation."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    source = io.BytesIO(value["bytes"]) if value.get("bytes") is not None else value["path"]
    with Image.open(source) as image:
        return image.convert("RGB")


def _sequence_content(doc: dict[str, Any], modality: str) -> list[dict[str, Any]]:
    domain = doc["metbench_domain"]
    if domain == "chess":
        instructions = "You are a helpful assistant that tracks chess moves in a game and produces the final FEN.\n"
        question = "What is the final FEN? Think step by step then output the final FEN as FINAL ANSWER: [FEN HERE]."
        caption = "The move is from the green square to the red square."
    else:
        instructions = "The shell game is a classic game where a ball is hidden under one of three shells. You are a helpful assistant that tracks the position of the ball during the swaps and determines the final position of the ball. We label the shells 1, 2, 3. The ball starts under one of the numbered shells which we call the initial state, and each move is shell swap of shell x and y written 'x swap y'.\n"
        question = "What is the final position of the ball? Think step by step then output the final ball location as FINAL ANSWER: [1, 2, or 3]."
        caption = "The shells with the numbers highlighted green are being swapped."
    text = f"{instructions}\n{question}\nThe initial state is: {doc['initial_state']}\nHere are the moves played:\n"
    content = []
    for index, action in enumerate(doc["actions"]):
        if modality == "image":
            content.append({"type": "text", "text": text})
            content.append({"type": "image", "url": _image(doc["image_actions"][index])})
            text = caption + "\n"
        else:
            text += f"{action}\n"
    content.append({"type": "text", "text": text + f"\n{question}"})
    return content


def _minecraft_content(doc: dict[str, Any], modality: str) -> list[dict[str, Any]]:
    action = doc["action"]
    count = len(doc["candidate_states"])
    if modality == "text":
        instructions = "You are evaluating a Minecraft gameplay trajectory. You will see a JSON game-state snapshot and an action that was performed. Your task is to identify which of the candidate game states is the correct next state after the action was taken."
        question = f'Which game state (1-{count}) is the correct next state after the action "{action}"? Think step by step about how the game state should change given the action, then output your answer as FINAL ANSWER: [1-{count}].'
        text = f"{instructions}\n\nAction performed: {action}\n\nInput state:\n{doc['initial_state']}\n\nCandidate next states:"
        for index, state in enumerate(doc["candidate_states"], start=1):
            text += f"\n\nChoice {index}:\n{state}"
        return [{"type": "text", "text": text + f"\n\n{question}"}]
    instructions = "You are evaluating a Minecraft gameplay trajectory. You will see a first-person screenshot from the game and an action that was performed. Your task is to identify which of the candidate images shows the correct next frame after the action was taken."
    question = f'Which image (1-{count}) shows the correct next frame after the action "{action}"? Think step by step about how the scene should change given the action, then output your answer as FINAL ANSWER: [1-{count}].'
    content = [
        {"type": "text", "text": f"{instructions}\n\nAction performed: {action}\n\nInput frame:"},
        {"type": "image", "url": _image(doc["image_initial_state"])},
        {"type": "text", "text": "\nCandidate next frames:"},
    ]
    for index, image in enumerate(doc["image_candidate_states"], start=1):
        content.extend([{"type": "text", "text": f"\nChoice {index}:"}, {"type": "image", "url": _image(image)}])
    content.append({"type": "text", "text": f"\n{question}"})
    return content


def _parse_choice(response: str, maximum: int) -> int | None:
    if not response:
        return None
    text = str(response).strip()
    markers = list(re.finditer(r"final\s+answer\s*:\s*", text, re.IGNORECASE))
    if markers:
        text = text[markers[-1].end() :].strip()
        text = text.splitlines()[0] if text else ""
    text = text.replace(r"\boxed", "")
    text = re.sub(r"[`*{}\[\]]", "", text).strip()
    match = re.fullmatch(r"(?:(?:choice|candidate|shell)\s*#?\s*)?([1-" + str(maximum) + r"])[.!]?", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _chess_accuracy(target: str, response: str) -> float:
    # Task discovery does not need the Chess scoring dependency.
    import chess

    truth = chess.Board(fen=str(target).strip().split("\n")[0].replace("`", "").replace("*", ""))
    for line in reversed(str(response).splitlines()):
        if "final answer" not in line.lower():
            continue
        for token in ("\\boxed{", "\\boxed"):
            line = line.replace(token, " ")
        for char in "`*[](){}'\",;:.":
            line = line.replace(char, " ")
        tokens = line.split()
        candidates = []
        for index in range(len(tokens) - 5):
            try:
                candidates.append(chess.Board(fen=" ".join(tokens[index : index + 6])))
            except ValueError:
                continue
        if len(candidates) == 1:
            predicted = candidates[0]
            return sum(truth.piece_at(square) == predicted.piece_at(square) for square in range(64)) / 64
    return 0.0


def build_messages(doc: dict[str, Any], modality: str) -> list[dict[str, Any]]:
    """Build one user turn, preserving the order of text and image inputs."""
    if modality not in {"text", "image"}:
        raise ValueError(f"Unsupported modality: {modality}")
    builder = _minecraft_content if doc["metbench_domain"] == "minecraft" else _sequence_content
    return [{"role": "user", "content": builder(doc, modality)}]


def score(doc: dict[str, Any], response: str) -> float:
    """Return board-square accuracy for Chess or answer accuracy otherwise."""
    if doc["metbench_domain"] == "chess":
        return _chess_accuracy(doc["target"], response)
    maximum = 4 if doc["metbench_domain"] == "minecraft" else 3
    return float(_parse_choice(response, maximum) == int(doc["target"]))


def clustered_ratio_interval(counts: list[tuple[float, int]]) -> tuple[float | None, float | None]:
    """Return a 95% normal interval for a ratio, clustered by example.

    Each pair contains correct and total states for one example. For Chess's
    64-square score this is the mean board score plus or minus 1.96 standard
    errors, using the sample variance across boards. Unequal denominators use
    the cluster delta method. Fewer than two contributing examples has no CI.
    """
    counts = [(correct, total) for correct, total in counts if total > 0]
    n = len(counts)
    if n < 2:
        return None, None
    total = sum(total for _, total in counts)
    accuracy = sum(correct for correct, _ in counts) / total
    residual_ss = sum((correct - accuracy * size) ** 2 for correct, size in counts)
    stderr = math.sqrt(n * residual_ss / (n - 1)) / total
    margin = 1.959963984540054 * stderr
    return max(0.0, accuracy - margin), min(1.0, accuracy + margin)
