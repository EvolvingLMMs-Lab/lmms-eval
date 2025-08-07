REFINEMENT_PROMPT = """
You are a mathematical expert with extensive knowledge across various mathematical fields. Your task is to meticulously evaluate and, if necessary, correct a given mathematical question and its proposed solution. Follow these steps:
1. Carefully read the provided question and solution.
2. Conduct a step-by-step review of the solution, addressing the following for each step:
  - Verify the mathematical correctness and logical flow.
  - Identify any errors including calculation errors, misunderstanding of the problem, or reasoning error. 
  - If an error is found, immediately stop the review process and proceed to step 3.
  - If no error is found, continue to the next step.
3. If an error is found:
  - Provide a brief explanation of the error.
  - Correct the solution starting from the erroneous step.
  - Complete the rest of the solution correctly.
4. If no errors are found in the entire solution, provide a brief confirmation of its correctness.

Output your analysis in the following format:

Review and Correction (if applicable):
Step 1: [Brief assessment of step 1]
Step 2: [Brief assessment of step 2]
...
Step X: [Brief assessment of step X]
Error found in step X: [Brief explanation of the error]
Corrected solution from step X:
Step X: [Corrected step]
Step X+1: [Next correct step]
...
[Final step]
...
Overall Assessment:
Correctness: [Correct / Incorrect]
Explanation: [Concise explanation of the assessment]
Final Answer: [Correct final answer]

Question: {question}
Solution: {initial_solution}
""".strip()

EVAL_PROMPT_CORRECT = """
Given the model's response, output 1 if 'Correctness' is 'correct', otherwise output 0. Respond with only the number.

Model's Response: {response}

Output:
""".strip()

EVAL_PROMPT_INCORRECT = """
You are an expert evaluator assessing the quality of feedback provided on an initial solution to a problem. Your task is to determine if the feedback is effective in guiding the initial solution towards a correct answer. You will be provided with three components:

1.  **Initial Solution:** The initial attempt at solving the problem.
2.  **Feedback:**  Specific feedback provided in response to the initial solution.
3.  **Reference Feedback:** A verified, high-quality feedback to the initial solution.

Your evaluation should consider the following aspects:

*   **Error Detection:** Does the feedback correctly identify the errors or shortcomings in the initial solution?
*   **Error Correction:** Does the feedback effectively address the problems in the initial solution?
*   **Effectiveness and Correctness of the Feedback:** Does the feedback guide the initial solution towards the correct answer efficiently? Does it reach the same answer and logic as the reference feedback in terms of its core principles?

Output your assessment in the following format:

Error Detection: [0/1]
Error Correction: [0/1]
Effectiveness and Correctness of the Feedback: [0/1]

No additional feedback or comment is required.

Initial Solution: {initial_solution}
Feedback: {feedback}
Reference Feedback: {reference_feedback}

Output:
""".strip()

PARSING_PROMPT = """
Given the model's response, parse "{target}" from the response. Respond with only the number.

If the model's response does not contain "{target}", output 0.

Model's Response: {model_response}
""".strip()