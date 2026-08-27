import numpy as np

from lmms_eval.tasks._task_utils.point_format import parse_point2d


def test_parse_point2d_scales_integer_grid_coordinates():
    points = parse_point2d(
        '```json\n[{"point_2d": [500, 250], "label": "target"}]\n```',
        width=200,
        height=100,
    )

    np.testing.assert_array_equal(points, np.array([[100, 25]]))


def test_parse_point2d_scales_normalized_float_coordinates():
    points = parse_point2d("[0.5, 0.25]", width=200, height=100)

    np.testing.assert_array_equal(points, np.array([[100, 25]]))


def test_parse_point2d_recovers_multiple_pairs():
    points = parse_point2d(
        '[{"point_2d": [100, 200]}, {"point_2d": [900, 800]}]',
        width=100,
        height=50,
    )

    np.testing.assert_array_equal(points, np.array([[10, 10], [90, 40]]))
