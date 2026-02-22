DESCRIPTIVE_RESP_INST = {
    1: """{}what is its title?
    * Your final answer should be the most relevant title of the plot that is explicitly written.
    * If the plot does not have an explicit title or contains only a letter, answer 'Not Applicable'.
    """,
    2: """{}what is the label of the x-axis?
    * Your final answer should be the label of the x-axis that is explicitly written, including the case when x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the plot, answer the label of the x-axis at the bottom.
    * If the plot does not have an explicit x-axis label, answer 'Not Applicable'.
    """,
    3: """{}what is the label of the y-axis?
    * Your final answer should be the label of the y-axis that is explicitly written, including the case when y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the plot, answer the label of the y-axis at the left.
    * If the plot does not have an explicit y-axis label, answer 'Not Applicable'.""",
    4: """{}what is the leftmost labeled tick on the x-axis?
    * Your final answer should be the tick value on the x-axis that is explicitly written, including the case when x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the plot, answer based on the axis at the bottom. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.""",
    5: """{}what is the rightmost labeled tick on the x-axis?
    * Your final answer should be the tick value on the x-axis that is explicitly written, including the case when x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the plot, answer based on the axis at the bottom. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.""",
    6: """{}what is the spatially lowest labeled tick on the y-axis?
    * Your final answer should be the tick value on the y-axis that is explicitly written, including the case when y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the plot, based on the axis at the left. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.""",
    7: """{}what is the spatially highest labeled tick on the y-axis?
    * Your final answer should be the tick value on the y-axis that is explicitly written, including the case when y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the plot, based on the axis at the left. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.""",
    8: """{}what is difference between consecutive numerical tick values on the x-axis?
    * Your final answer should be the difference between consecutive numerical tick values of the x-axis, including the case when x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the plot, answer based on the axis at the bottom. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.
    * If the plot does not have an explicit x-axis tick value, or if the tick values are not numerical, or if the difference is not constant between all consecutive tick values, answer "Not Applicable".""",
    9: """{}what is difference between consecutive numerical tick values on the y-axis?
    * Your final answer should be the difference between consecutive numerical tick values of the y-axis, including the case when y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the plot, answer based on the axis at the left. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.
    * If the plot does not have an explicit y-axis tick value, or if the tick values are not numerical, or if the difference is not constant between all consecutive tick values, answer "Not Applicable".""",
    10: """{}how many lines are there?
    * Your final answer should be the number of lines in the plot. Ignore grid lines, tick marks, and any vertical or horizontal auxiliary lines.
    * If the plot does not contain any lines or is not considered a line plot, answer "Not Applicable".""",
    11: """{}do any lines intersect?
    * Your final answer should be "Yes" if any lines intersect, and "No" otherwise. Ignore grid lines, tick marks, and any vertical or horizontal auxiliary lines.
    * If the plot does not contain any lines or is not considered a line plot, answer "Not Applicable".""",
    12: """{}how many discrete labels are there in the legend?
    * Your final answer should account for only labels relevant to the plot in the legend, even if the legend is located outside the plot. 
    * If the plot does not have a legend or no legend is not considered relevant to this plot, answer "Not Applicable".""",
    13: """{}what are the names of the labels in the legend?
    * You should write down the labels from top to bottom, then from left to right and separate the labels with commas. Your final answer should account for only labels relevant to the plot in the legend, even if the legend is located outside the plot.
    * If the plot does not have a legend or no legend is not considered relevant to this plot, answer "Not Applicable".""",
    14: """{}what is the difference between the maximum and minimum values of the tick labels on the continuous legend (i.e., colorbar)?
    * You should remove the percentage sign (if any) in your answer.
    * If the plot does not have an explicit colorbar-based continuous legend or the legend is not considered relevant to this subplot, answer "Not Applicable".""",
    15: """{}what is the maximum value of the tick labels on the continuous legend (i.e., colorbar)?
    * You should remove the percentage sign (if any) in your answer. 
    * If the plot does not have an explicit colorbar-based continuous legend or the legend is not considered relevant to this subplot, answer "Not Applicable".""",
    16: """{}what is the general trend of data from left to right?
    * Your final answer should be within a few words, such as "increases", "increases then stabilizes".""",
    17: """{}What is the total number of explicitly labeled ticks across all axes?
    * Your final answer should be the total number of explicitly labeled ticks across all axes, including the case when any axis is shared across multiple subplots.""",
    18: """What is the layout of the subplots?
    * Your final answer should follow "n by m" format, where n is the number of rows and m is the number of columns.
    * If the plot does not contain subplots, answer "1 by 1".""",
    19: """What is the number of subplots?
    * Your final answer should be the total number of subplots in the plot.
    * If the plot does not contain subplots, answer "1".""",
}

DESCRIPTIVE_GRADING_PREFIX = """
You will be given <|NUM_TRIPLETS|> pairs of ground truth answers and model responses under an overarching question. You need to go through each of the pairs, extract the final answer from the model response, compare it with the ground truth answer, and then assign a binary score. Avoid providing explanations in your response. If there is no provided model response, please leave the extracted answer empty and give a score of 0. Your response must follow json formats with keys [<|JSON_KEYS|>] where the value for any `extract_answer` is your extracted answer and `score` is an interge in [0, 1] based on the following rules:\n

Overarching Question: <|OVERARCHING_QUESTION|>
"""

DESCRIPTIVE_GRADING_QMAP = {
    1: "What is the title of the plot?",
    2: "What is the label of the x-axis?",
    3: "What is the label of the y-axis?",
    4: "What is the leftmost labeled tick on the x-axis?",
    5: "What is the rightmost labeled tick on the x-axis?",
    6: "What is the spatially lowest labeled tick on the y-axis?",
    7: "What is the spatially highest labeled tick on the y-axis?",
    8: "What is difference between consecutive numerical tick values on the x-axis?",
    9: "What is difference between consecutive numerical tick values on the y-axis?",
    10: "How many lines are there?",
    11: "Do any lines intersect?",
    12: "How many discrete labels are there in the legend?",
    13: "What are the names of the labels in the legend? (from top to bottom, then left to right)",
    14: "What is the difference between the maximum and minimum values of the tick labels on the continuous legend (i.e., colorbar)?",
    15: "What is the maximum value of the tick labels on the continuous legend (i.e., colorbar)?",
    16: "What is the general trend of data from left to right?",
    17: "What is the total number of explicitly labeled ticks across all axes?",
    18: "What is the layout of the subplots?",
    19: "What is the number of subplots?",
}

DESCRIPTIVE_GRADING_ICL = {
    "title": """
Rubric: 
    * Give a score of 1 if and only if the extracted answer and the ground truth answer are referring to the same term. It's acceptable to have different grammar or form (e.g., α and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). It's acceptable to omit letter prefixes (e.g., (a) Increment over time and Increment over time).
    * Give a score of 0 if any term in the extracted answer is different from the ground truth answer.
    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.

    ### Example Start ###
    T1:
    Response 1: The title of the plot is "The number of students in each grade".
    Ground Truth 1: The variance of students in each grade

    T2:
    Response 2: There is no title.
    Ground Truth 2: Not Applicable

    T3:
    Response 3: A_v^t
    Ground Truth 3: A^t_v

    {
        "extract_answer_T1": "The number of students in each grade",
        "score_T1": 0
        "extract_answer_T2: "Not Applicable",
        "score_T2": 1
        "extract_answer_T3": "A_v^t",
        "score_T3": 1
    }
    ### Example End ###        
""",
    "ocr": """
Rubric: 
    * Give a score of 1 if and only if the extracted answer and the ground truth answer are referring to the same term. It's acceptable to have equivalent grammar or form (e.g., α and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). If the ground truth is a number, the extracted answer should be the number with the exact same value.
    * Give a score of 0 if any term in the extracted answer is different from the ground truth answer, or if the extracted number is different in value from the ground truth number.
    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.

    ### Example Start ###
    T1:
    Response 1: The answer is 1.0
    Ground Truth 1: 1.00

    T2:
    Response 2: By manually inspecting the plot, the final answer should be 0.
    Ground Truth 2: Not Applicable

    T3:
    Response 3: A_v^t
    Ground Truth 3: A^t_v

    {
        "extract_answer_T1": 1.0,
        "score_T1": 1
        "extract_answer_T2: 0,
        "score_T2": 0
        "extract_answer_T3": "A_v^t",
        "score_T3": 1
    }
    ### Example End ###        
""",
    "quant": """
Rubric:
    * Give a score of 1 if and only if the extracted answer and the ground truth answer are numbers with the exact same value.
    * Give a score of 0 if the extracted answer is different in value from the ground truth answer.
    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.

    ### Example Start ###
    T1:
    Response 1: 5
    Ground Truth 1: 6

    T2:
    Response 2: 0
    Ground Truth 2: Not Applicable

    T3:
    Response 3: 4
    Ground Truth 3: 4

    {
        "extract_answer_T1": 5,
        "score_T1": 0
        "extract_answer_T2: 0,
        "score_T2": 0
        "extract_answer_T3": 4,
        "score_T3": 1
    }
    ### Example End ###   
""",
    "bool": """
Rubric:
    * Give a score of 1 if and only if the extracted answer and the ground truth answer are the same.
    * Give a score of 0 if the extracted answer and the ground truth answer are different.
    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.

    ### Example Start ###
    T1:
    Response 1: No, there are no intersections.
    Ground Truth 1: no

    T2:
    Response 2: No, all the lines are parallel.
    Ground Truth 2: Yes

    T3:
    Response 3: There are no lines in the plot.
    Ground Truth 3: Not Applicable

    {
        "extract_answer_T1": "No",
        "score_T1": 1
        "extract_answer_T2: "No",
        "score_T2": 0
        "extract_answer_T3": "Not Applicable",
        "score_T3": 1
    }
    ### Example End ###   
""",
    "enum": """
Rubric:
    * Give a score of 1 if and only if the extracted answer and the ground truth answer are referring to the same term. It's acceptable to have equivalent grammar or form (e.g., α and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). The order of the terms must be the same.
    * Give a score of 0 if any term in the extracted answer is different from the ground truth answer, or if the order of the terms is different.
    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.

    ### Example Start ###
    T1:
    Response 1: Here are the names of the labels: A, B, C
    Ground Truth 1: B, A, C

    T2:
    Response 2: The labels are T56, B33.
    Ground Truth 2: T56,B33,A12

    T3:
    Response 3: \alpha, \beta, \gamma^t_v
    Ground Truth 3: α, β, γ_v^t

    {
        "extract_answer_T1": "A, B, C",
        "score_T1": 0
        "extract_answer_T2: "T56, B33",
        "score_T2": 0
        "extract_answer_T3": "\alpha, \beta, \gamma^t_v",
        "score_T3": 1
    }
    ### Example End ###
""",
    "trend": """
Rubric:
    * Give a score of 1 if and only if the extracted answer and the ground truth answer share the same general trend.
    * Give a score of 0 if the extracted answer and the ground truth answer are different in trend expression.

    ### Example Start ###
    T1:
    Response 1: there is an increase in the data from left to right
    Ground Truth 1: Decreases

    T2:
    Response 2: the curves move up and stay constant
    Ground Truth 2: Increases then stabilizes

    T3:
    Response 3: Decreases
    Ground Truth 3: Decreases then increases

    {
        "extract_answer_T1": "Increases",
        "score_T1": 0
        "extract_answer_T2: "Move up and stay constant",
        "score_T2": 1
        "extract_answer_T3": "Decreases",
        "score_T3": 0
    }
    ### Example End ###
""",
    "layout": """
Rubric:
    * Give a score of 1 if and only if the extracted answer and the ground truth answer are the same in terms of the number of rows and columns (e.g., n by m).
    * Give a score of 0 if the extracted answer is different from the ground truth answer.

    ### Example Start ###
    T1:
    Response 1: 2 by 3
    Ground Truth 1: 3 by 2

    T2:
    Response 2: the layout is 1 by 1
    Ground Truth 2: 1 by 1

    T3:
    Response 3: there are two rows and three columns
    Ground Truth 3: 2 by 3

    {
        "extract_answer_T1": "2 by 3",
        "score_T1": 0
        "extract_answer_T2: "1 by 1",
        "score_T2": 1
        "extract_answer_T3": "2 by 3",
        "score_T3": 1
    }
    ### Example End ###
""",
}

REASONING_GRADING_PREFIX = """
You will be given a question, an ground truth answer and a model response. You need to extract the final answer from the model response, compare it with the ground truth answer, and then assign a binary score. Avoid providing explanations in your response. If there is no provided model response, please leave the extracted answer empty and give a score of 0. 

Your response must follow json formats with keys [extract_answer, score] where the value of the score is an integer in [0, 1]. You must follow the scoring rules:\n"""

REASONING_GRADING_INST = {
    1: """
    ### Rules ###
    * Give a score of 1 if and only if the final answer and the ground truth answer are referring to the same term. It's acceptable to have different grammar or form (e.g., α and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). It's also acceptable to have different orders of the terms when question asks for multiple terms.
    * Give a score of 0 if any term (e.g., ACC+ and ACC; P-101 and P=101) is different between the final answer and the ground truth.

    ### Example 1 Starts ###
    * Question: What is the name of the curve that intersects y=\lambda exactly three times?
    * Ground Truth: P56962
    * Response: There is only one curve that intersects y=\lambda exactly three times. The name of the curve is written as P55762.
    
    {
        "extracted_answer": "P55762",
        "score": 0
    }
    ### Example 1 Ends ###


    ### Example 2 Starts ###
    * Question: What is the letter of the subplot where all bars are above 35?
    * Ground Truth: (b)
    * Response: The letter of the subplot where all bars are above 35 is b.

    {
        "extracted_answer": "b",
        "score": 1
    }
    ### Example 2 Ends ###

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>

    """,
    2: """
    ### Rules ###
    * If there are predefined options in the question:
        * Give a score of 1 if the final answer matches the ground truth answer exactly.
        * Give a score of 0 if the final answer does not match the ground truth answer.
    * If there are no predefined options in the question:
        * Give a score of 1 if the final answer shares the same semantic meaning with the ground truth answer (e.g., "increasing then decreasing" and "moving up then down"; "converge" and "move closer together").
        * Give a score of 0 if the final answer shares different semantic meanings from the ground truth answer (e.g., "increasing then decreasing" and "remain constant"; "converge" and "diverge").

    ### Example 1 Starts ###
    * Question: What is the trend of the red curve between t=10 and t=25?
    * Ground Truth: increasing then decreasing
    * Response: The red curve is increasing between t=10 and t=25.

    {
        "extracted_answer": "increasing",
        "score": 0
    }
    ### Example 1 Ends ###

    ### Example 2 Starts ###
    * Question: What is the interval where the blue curve achieves the maximum value among [0, 50], [50, 100], [100, 150], and [150, 200]?
    * Ground Truth: [50, 100]
    * Response: The interval where the blue curve achieves the maximum value is [50, 100].

    {
        "extracted_answer": "[50, 100]",
        "score": 1
    }
    ### Example 2 Ends ###

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>

    """,
    3: """
    ### Rules ###
    * Give a score of 1 if and only if the two numbers are exactly equal in values. It's acceptable to have different notations (e.g., 0.01 and 10^-2; 1500 and 1.5e3).
    * Give a score of 0 if the two numbers are different in values.

    ### Example 1 Starts ###
    * Question: What is the value of the red curve at t=10?
    * Ground Truth: 0.01
    * Response: The value of the red curve at t=10 is 0.012.

    {
        "extracted_answer": "0.012",
        "score": 0
    }
    ### Example 1 Ends ###

    ### Example 2 Starts ###
    * Question: What is the value of the blue curve at t=50?
    * Ground Truth: 1500
    * Response: The value of the blue curve at t=50 is 1.5e3.

    {
        "extracted_answer": "1.5e3",
        "score": 1
    }
    ### Example 2 Ends ###

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>

    """,
    4: """
    ### Rules ###
    * Give a score of 1 if and only if the two numbers are exactly equal in values. It's acceptable to have different notations (e.g., 0.01 and 10^-2; 1500 and 1.5e3).
    * Give a score of 0 if the two numbers are different in values.

    ### Example 1 Starts ###
    * Question: What is the value of the red curve at t=10?
    * Ground Truth: 0.01
    * Response: The value of the red curve at t=10 is 0.012.

    {
        "extracted_answer": "0.012",
        "score": 0
    }
    ### Example 1 Ends ###

    ### Example 2 Starts ###
    * Question: What is the value of the blue curve at t=50?
    * Ground Truth: 1500
    * Response: The value of the blue curve at t=50 is 1.5e3.

    {
        "extracted_answer": "1.5e3",
        "score": 1
    }
    ### Example 2 Ends ###

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>

    """,
}

REASONING_RESP_INST = {
    1: """{}
    * Your final answer must be grounded to some text that is explicitly written and relevant to the question in the chart.
    * If you need to answer multiple terms, separate them with commas.
    * Unless specified in the question (such as answering with a letter), you are required to answer the full names of subplots and/or labels by default.
    """,
    2: """{}
    * If there are options in the question, your final answer must conform to one of the options.
    * If there are additional instructions in the question, follow them accordingly.
    * If there are neither options nor additional instructions, you are allowed to respond with a short phrase only.
    """,
    3: """{}
    * Your final answer must be grounded to a number that is exlicitly written and relevant to the question in the chart, even if it's an approximate value.
    * You are allowed to extract numbers within some text when needed.
    """,
    4: """{}
    {}
    """,
}
