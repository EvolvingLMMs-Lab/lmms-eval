import pytest

from lmms_eval.tasks.humanref import utils

PERSON_A = [10.0, 10.0, 110.0, 210.0]
PERSON_B = [300.0, 20.0, 400.0, 220.0]
PERSON_C = [600.0, 30.0, 700.0, 230.0]


def doc(domain="attribute", answer_boxes=(PERSON_A, PERSON_B), candidate_boxes=(PERSON_A, PERSON_B, PERSON_C)):
    return {
        "id": "0",
        "image_name": "x.jpg",
        "referring": "the two people on the left",
        "domain": domain,
        "sub_domain": f"{domain}_sub",
        "answer_boxes": [list(b) for b in answer_boxes],
        "candidate_boxes": [list(b) for b in candidate_boxes],
    }


def test_doc_to_text_matches_official_qwen_prompt():
    assert utils.humanref_doc_to_text({"referring": "the man in red"}) == "Locate the man in red, output its bbox coordinates using JSON format."


@pytest.mark.parametrize(
    "response, expected",
    [
        ('```json\n[{"bbox_2d": [1, 2, 3, 4], "label": "person"}, {"bbox_2d": [5, 6, 7, 8]}]\n```', [[1, 2, 3, 4], [5, 6, 7, 8]]),
        ('{"bbox": [0.1, 0.2, 0.3, 0.4]}', [[0.1, 0.2, 0.3, 0.4]]),
        ("[[1, 2, 3, 4], [5, 6, 7, 8]]", [[1, 2, 3, 4], [5, 6, 7, 8]]),
        ("The people are at (1, 2, 3, 4) and (5, 6, 7, 8).", [[1, 2, 3, 4], [5, 6, 7, 8]]),
        ("<think>maybe [9, 9, 9, 9]</think>```json\n[]\n```", []),
        ("[]", []),
        ("There is no such person.", []),
        ('[{"bbox_2d": [5, 5, 1, 1]}]', []),
    ],
)
def test_parse_boxes(response, expected):
    assert utils.parse_boxes(response) == [[float(v) for v in box] for box in expected]


def test_official_metric_perfect_partial_and_over_detection():
    perfect = utils.box_metrics([PERSON_A, PERSON_B], [PERSON_A, PERSON_B], total_persons=3)
    assert perfect["recall"] == perfect["precision"] == perfect["density_f1"] == 1.0

    partial = utils.box_metrics([PERSON_A, PERSON_B], [PERSON_A], total_persons=3)
    assert partial["recall_50"] == 0.5 and partial["precision_50"] == 1.0
    assert partial["density_f1_50"] == pytest.approx(2 / 3)

    over = utils.box_metrics([PERSON_A, PERSON_B], [PERSON_A, PERSON_B, PERSON_C], total_persons=3)
    assert over["recall_50"] == 1.0 and over["precision_50"] == pytest.approx(2 / 3)
    assert over["density_f1_50"] == pytest.approx(0.8 * 2 / 3)

    assert utils.box_metrics([PERSON_A], [], total_persons=3)["density_f1"] == 0.0


def test_greedy_matching_uses_each_prediction_once():
    assert utils.recall_precision([PERSON_A, PERSON_A], [PERSON_A], 0.5) == (0.5, 1.0)


def test_score_prediction_picks_best_coordinate_reading():
    width, height = 1000, 500
    normalized = [[b[0] / width, b[1] / height, b[2] / width, b[3] / height] for b in (PERSON_A, PERSON_B)]
    assert utils.score_prediction(doc(), normalized, width, height)["density_f1"] == pytest.approx(1.0)
    grid_1000 = [[b[0], b[1] * 1000 / height, b[2], b[3] * 1000 / height] for b in (PERSON_A, PERSON_B)]
    assert utils.score_prediction(doc(), grid_1000, width, height)["density_f1"] == pytest.approx(1.0)


def test_rejection_domain_and_aggregations():
    rejection = doc("rejection", answer_boxes=())
    records = [
        utils.score_prediction(doc("attribute"), [PERSON_A, PERSON_B], 1000, 500),
        utils.score_prediction(doc("celebrity"), [], 1000, 500),
        utils.score_prediction(rejection, [], 1000, 500),
        utils.score_prediction(rejection, [PERSON_C], 1000, 500),
    ]
    assert utils.humanref_density_f1(records) == pytest.approx(0.5)
    assert utils.humanref_recall_50(records) == pytest.approx(0.5)
    assert utils.humanref_rejection_acc(records) == pytest.approx(0.5)
    assert utils.humanref_attribute_density_f1(records) == pytest.approx(1.0)
    assert utils.humanref_celebrity_density_f1(records) == 0.0
