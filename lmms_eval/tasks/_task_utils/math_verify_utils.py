# Copyright 2025 Xiaomi Corporation.


import importlib

from wrapt_timeout_decorator import timeout


def patch_target_module(
    to_patch: str,
    replace_with,
):
    to_patch = to_patch.split(".")
    assert len(to_patch) > 1, "must have an object to patch"

    to_patch, obj_name_to_patch = to_patch[:-1], to_patch[-1]
    to_patch = ".".join(to_patch)
    source = importlib.import_module(to_patch)
    setattr(source, obj_name_to_patch, replace_with)


def timeout_adapter(func=None, **kwargs):
    timeout_val = kwargs.pop("timeout_seconds", None)
    return timeout(dec_timeout=timeout_val, use_signals=False, **kwargs)


# replace the signal-based timeout with a non-signal-based timeout to allow multithreading
patch_target_module("math_verify.utils.timeout", timeout_adapter)
patch_target_module("math_verify.parser.timeout", timeout_adapter)
patch_target_module("math_verify.grader.timeout", timeout_adapter)


import os

from latex2sympy2_extended.latex2sympy2 import NormalizationConfig
from math_verify import *


def monkeypatch_math_verify_logger():
    """
    replace the loggers in math_verify with a self-returning object, so that it does not print any logs
    """
    import math_verify

    class SelfReturningObject:
        def __getattr__(self, name):
            return self

        def __call__(self, *args, **kwargs):
            return self

        def __getitem__(self, key):
            return self

    self_returning_object = SelfReturningObject()

    def bfs_search(module, lst):
        lst.append(module)
        for name, obj in module.__dict__.items():
            if isinstance(obj, type(math_verify)):
                if obj not in lst:
                    bfs_search(obj, lst)

    all_modules = []
    bfs_search(math_verify, all_modules)
    all_modules = [module for module in all_modules if module.__name__.startswith("math_verify")]
    for module in all_modules:
        if hasattr(module, "logger"):
            module.logger = self_returning_object


class MathVerifyFn:
    def __init__(self, correct_score=1.0, incorrect_score=0.0, timeout_seconds=10, strict=True, silent=True):
        self.correct_score = correct_score
        self.incorrect_score = incorrect_score
        self.timeout_seconds = timeout_seconds
        self.strict = strict
        if silent:
            monkeypatch_math_verify_logger()

    def __call__(self, solution_str: str, ground_truth) -> float:
        # return self.compute_score(solution_str, ground_truth)
        return self.compute_score_with_ext(solution_str, ground_truth)

    def preprocess_answer(self, annotated_answer: str) -> str:
        if annotated_answer:
            if annotated_answer.startswith("$") and annotated_answer.endswith("$"):
                annotated_answer = f"\\boxed{{{annotated_answer.strip('$')}}}"
            elif "\\boxed" not in annotated_answer:
                annotated_answer = f"\\boxed{{{annotated_answer}}}"
        return annotated_answer

    def parse_LatexExpr(self, input_str: str):
        config = NormalizationConfig(
            basic_latex=True,
            units=True,
            malformed_operators=True,
            nits=True,
            boxed="last",
            equations=False,
        )
        return parse(
            input_str,
            extraction_mode="first_match",
            extraction_config=[
                LatexExtractionConfig(boxed_match_priority=0, normalization_config=config),
            ],
            parsing_timeout=self.timeout_seconds,
        )

    def parse_String(self, input_str: str):
        return parse(
            input_str,
            extraction_mode="first_match",
            extraction_config=[
                StringExtractionConfig(),
            ],
            parsing_timeout=self.timeout_seconds,
        )

    def judge_with_ext(self, solution_str: str, ground_truth) -> float:
        prediction_str = solution_str
        answer_str = self.preprocess_answer(ground_truth)
        answer_parsed = self.parse_LatexExpr(answer_str)

        def _judger(x):
            if len(x) == 0:
                return False
            if verify(answer_parsed, x, timeout_seconds=self.timeout_seconds, strict=self.strict):
                return True
            return False

        def ext_to_str(x):
            for item in x:
                if isinstance(item, str):
                    return item
            for item in x:
                return str(item)
            return ""

        ext_pred = self.parse_LatexExpr(prediction_str)
        ext_str = ext_to_str(ext_pred)
        # print(solution_str[:20], ground_truth, ext_pred, ext_str, _judger(ext_pred))
        if _judger(ext_pred):
            return True, ext_str
        return False, ext_str

    def compute_score_with_ext(self, solution_str: str, ground_truth) -> float:
        try:
            is_correct, ext_pred = self.judge_with_ext(solution_str, ground_truth)
            if is_correct:
                return self.correct_score, ext_pred
            else:
                return self.incorrect_score, ext_pred
        except Exception as e:
            print(e)
            return self.incorrect_score, ""


if __name__ == "__main__":
    math_verify_fn = MathVerifyFn()
    print(math_verify_fn("\\boxed{D}", "D"))
