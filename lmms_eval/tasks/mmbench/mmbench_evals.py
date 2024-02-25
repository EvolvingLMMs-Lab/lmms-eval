import pandas as pd


class MMBench_Evaluator:
    def __init__(self, sys_prompt="There are several options:"):
        self.sys_prompt = sys_prompt

    def create_options_prompt(self, row_data, option_candidate):
        available_keys = set(row_data.keys()) & set(option_candidate)
        options = {cand: row_data[cand] for cand in available_keys if row_data[cand]}
        sorted_options = dict(sorted(options.items()))
        options_prompt = f"{self.sys_prompt}\n"
        for key, item in sorted_options.items():
            if pd.notna(item) and item != "nan":
                options_prompt += f"{key}. {item}\n"
        return options_prompt.rstrip("\n"), sorted_options
