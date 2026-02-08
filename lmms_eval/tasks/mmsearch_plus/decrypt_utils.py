"""Decryption utilities for MMSearch-Plus dataset."""

import base64
import hashlib
from typing import Any, Dict


def derive_key(password: str, length: int) -> bytes:
    """
    Derive encryption key from password using SHA-256.

    Args:
        password: Password/canary string
        length: Desired key length

    Returns:
        Derived key of specified length
    """
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt_text(ciphertext_b64: str, password: str) -> str:
    """
    Decrypt base64-encoded ciphertext using XOR cipher with derived key.

    Args:
        ciphertext_b64: Base64-encoded encrypted string
        password: Password/canary string

    Returns:
        Decrypted string
    """
    if not ciphertext_b64:
        return ciphertext_b64

    try:
        encrypted = base64.b64decode(ciphertext_b64)
        key = derive_key(password, len(encrypted))
        decrypted = bytes([a ^ b for a, b in zip(encrypted, key)])
        return decrypted.decode("utf-8")
    except Exception as e:
        print(f"[Warning] Decryption failed: {e}")
        return ciphertext_b64  # Return original if decryption fails


def decrypt_sample(sample: Dict[str, Any], canary: str) -> Dict[str, Any]:
    """
    Decrypt text fields in a single sample using the provided canary password.

    Args:
        sample: Dataset sample with encrypted fields
        canary: Canary string (e.g., 'MMSearch-Plus')

    Returns:
        Decrypted sample
    """
    decrypted_sample = sample.copy()

    # Decrypt text fields
    text_fields = ["question", "video_url", "arxiv_id"]

    for field in text_fields:
        if field in sample and sample[field]:
            decrypted_sample[field] = decrypt_text(sample[field], canary)

    # Handle answer field (list of strings)
    if "answer" in sample and sample["answer"]:
        decrypted_answers = []
        for answer in sample["answer"]:
            if answer:
                decrypted_answers.append(decrypt_text(answer, canary))
            else:
                decrypted_answers.append(answer)
        decrypted_sample["answer"] = decrypted_answers

    # Images are NOT encrypted in the current version
    # category, difficulty, and subtask are also NOT encrypted

    return decrypted_sample
