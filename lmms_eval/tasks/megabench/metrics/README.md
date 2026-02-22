# MEGA-Bench metrics

Each task's metrics are specified in `metrics.json` and follow the schema outlined below:

```json
{
    "field_score_function": {
        for field_name in field_names:
        field_name: scoring_function
    },
    "aggregation": {
        "function": aggregation_function,
        "field_weights": {
            for field_name in field_names:
            field_name: field_weight
        }
    },
    "response_parse_function": response_parse_function
}
```

## Scoring Functions

### String Comparisons
These metrics are applied when both the response and the correct field are strings.

- `exact_str_match`: Checks if the field exactly matches the reference response.
- `simple_str_match`: Performs a case-insensitive comparison, ignoring spaces and hyphens, to determine if the response matches the correct field.
- `exact_str_match_case_insensitive`: A case-insensitive version of `exact_str_match`.
- `normalized_similarity_demarau_levenshtein`: Computes the normalized Damerau-Levenshtein similarity between the strings.
- `near_str_match`: Normalizes accented characters to their ASCII equivalents, then performs a case-insensitive fuzzy string match. If the similarity score is below a certain threshold (currently 0.9), the score is set to 0.
- `program_judge`: A custom suite of test cases specifically for the `code_programming_test` task.

### Set Comparisons
These metrics are used when inputs are iterables or strings that represent sets. Inputs are converted into sets and treated as empty if parsing fails. These metrics are useful when the order doesn't matter.

`set_equality`: Checks if the sets are identical.
`jaccard_index`: Calculates the Jaccard index of the two sets.
`set_precision`: Measures the ratio of the predicted set elements that appear in the correct set.
`chess_move_list_jaccard_index`: Computes the Jaccard index without requiring the response to specify whether a move results in check or checkmate.

### List Comparisons
These metrics apply when inputs are iterables or strings that represent lists. Inputs are converted into lists and treated as empty if parsing fails.

`longest_common_list_prefix_ratio`: Calculates the ratio of the length of the longest common prefix (list) to the length of the correct solution.

#### Bounding boxes
Bounding boxes are a specialized type of list metric. Each solution consists of a list of 4-tuples, where each tuple is of the form (x1, y1, x2, y2), representing the top-left and bottom-right corners of the bounding box, respectively. Since images are dynamically resized before being sent to the LMM, coordinates are normalized to the range [0, 1].

`nbbox_iou_tuple`: Matches each predicted bounding box with the one that has the highest Intersection over Union (IoU) score, which is then used as the score for that bounding box. The mean score across all predicted bounding boxes is calculated.

### Dictionary Comparisons
These metrics apply when inputs are dictionaries or strings that encode dictionaries. Inputs are converted into dictionaries and treated as empty if parsing fails.

Generally, these metrics follow a two-step approach:
1. Calculate a metric for values with matching keys, resulting in a mapping of key-score pairs. If a key is missing in the response, its score is set to 0.
2. Aggregate the scores.

This approach is straightforward when the keys in the response and the correct answer match. If they don't, various strategies can be employed.

- `agg_recall`: Computes the mean score for keys that appear in the correct answer.
- `agg_jaccard`: Computes the mean score across all keys, using the size of the union of keys from both the response and the correct answer as the denominator.
 
Derivative metrics that follow this format include:
- `dict_exact_str_match_agg_recall`
- `dict_set_equality_agg_jaccard`
- `dict_jaccard_agg_jaccard`
- `dict_nbbox_iou_tuple_agg_jaccard`

## Aggregation Functions
The following functions are used to aggregate the field scores:

- `mean`: Calculates a weighted mean, with weights specified in `aggregation.field_weights`.
- `min`: Returns the minimum field score.

## Response Parsing Functions
These functions are used to parse the model's response:

`json`: Parses the response as a JSON object.
`odd_one_out`: A custom parser for the `logical_reasoning_find_odd_out_one` task.
`logical_2d_views_3d_shapes`: A custom parser for the `logical_reasoning_2D_views_of_3D_shapes` task.
