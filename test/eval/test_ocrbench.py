from lmms_eval.tasks.ocrbench.utils import ocrbench_process_results


def _score(answer: str, prediction: str, dataset: str = "IIIT5K") -> int:
    result = ocrbench_process_results(
        {
            "answer": [answer],
            "dataset": dataset,
            "question_type": "Regular Text Recognition",
        },
        [prediction],
    )
    return result["ocrbench_accuracy"]["score"]


def test_ocrbench_treats_fullwidth_ascii_as_equivalent():
    assert _score("2590", "２５９０") == 1
    assert _score("２５９０", "2590") == 1
    assert _score("both sides", "ｂｏｔｈ　ｓｉｄｅｓ") == 1


def test_ocrbench_does_not_fold_other_compatibility_characters():
    assert _score("1", "①") == 0
    assert _score("2", "²") == 0
    assert _score("IV", "Ⅳ") == 0
    assert _score("kg", "㎏") == 0


def test_ocrbench_keeps_hme_width_sensitive():
    assert _score("x+1", "ｘ＋１", dataset="HME100k") == 0
