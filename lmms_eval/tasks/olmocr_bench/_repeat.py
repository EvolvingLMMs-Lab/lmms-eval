# Adapted from allenai/olmocr at f7cfe4c22098b154c76b6ec950d1c0a464eecf8d.
# Copyright Allen Institute for Artificial Intelligence.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Official trailing n-gram repetition detector."""

import re


class RepeatDetector:
    def __init__(self, max_ngram_size: int = 10):
        self.max_ngram_size = max_ngram_size
        self.data = ""

    def add_letters(self, new_str: str):
        self.data += new_str

    def ngram_repeats(self) -> list[int]:
        result = [0] * self.max_ngram_size

        if not self.data:
            return result

        # Normalize all whitespace to single spaces
        text = re.sub(r"\s+", " ", self.data)

        # For each n-gram size
        for size in range(1, self.max_ngram_size + 1):
            if len(text) < size:
                continue

            # Get the last n-gram
            target = text[-size:]

            # Count backwards from the end to find repeats
            count = 0
            pos = len(text) - size  # Start position for previous n-gram

            while pos >= 0:
                if text[pos : pos + size] == target:
                    count += 1
                    pos -= size  # Move back by the size of the n-gram
                else:
                    break

            result[size - 1] = count

        return result
