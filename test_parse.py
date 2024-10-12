from lmms_eval.filters.extraction import MultiChoiceRegexFilter

def parse_multi_choice_answer(answer):
    # Example responses and documents
    model_responses = [
        ["The answer is (B)", "I believe it is (A)", "(C) seems correct"],  # Model response set 1
        ["Answer is: B!", "Answer: B", "Answer: B"]  # Model response set 2
    ]

    documents = [
        {"choices": ["A. Apple", "B. Banana", "C. Cherry"]},  # Multiple choice options for question 1
        {"choices": ["A. Alpha", "B. Beta", "C. Gamma"]}      # Multiple choice options for question 2
    ]

    # Instantiate the filter
    multi_choice_filter = MultiChoiceRegexFilter(
        regex_pattern=r"\(([A-D])\)",
        group_select=0,
        ignore_case=False,
        ignore_punctuation=True
    )

    filtered_responses = multi_choice_filter.apply(model_responses, documents)

    # Print the filtered answers
    for i, filtered in enumerate(filtered_responses):
        print(f"Question {i+1} filtered responses: {filtered}")

parse_multi_choice_answer("a")