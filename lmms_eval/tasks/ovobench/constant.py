#The code below is from https://github.com/JoeLeelyf/OVO-Bench/blob/main/constant.py
BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REAL_TIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]

# Prompt template for backward-tracing and real-time visual perception task
BR_PROMPT_TEMPLATE = """
Question: {}
Options:
{}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
Do not include any additional text or explanation in your response.
"""

# Prompt template for REC task
# REC_PROMPT_TEMPLATE = """ 
# You're provided with multiple images which are frames extracted from a video, in which the man/woman are performing an action repetitively.

# Now, answer the following question: Have the person in the video {} {} times?

# Answer only with “Yes” or “No”.
# Do not include any additional text or explanation in your response.
# """
REC_PROMPT_TEMPLATE = """
You're watching a video in which people may perform a certain type of action repetively. 
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one. 
Now, answer the following question: {}
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
"""

# Prompt template for SSR task
# SSR_PROMPT_TEMPLATE = """
# You're provided with multiple images which are frames extracted from a tutorial video, in which the whole process may contain multiple different steps.

# Now, answer the following question: Have the person in the video complete {}?

# Answer only with “Yes” or “No”.
# Do not include any additional text or explanation in your response.
# """
SSR_PROMPT_TEMPLATE = """
You're watching a tutorial video which contain a sequential of steps. 
The following is one step from the whole procedures: 
{}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.
"""

# Prompt template for CRR task
CRR_PROMPT_TEMPLATE = """
You're responsible of answering questions based on the video content. 
The following question are relevant to the latest frames, i.e. the end of the video.
{}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.
"""