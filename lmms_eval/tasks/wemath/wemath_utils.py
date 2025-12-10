import pandas as pd


# Function to process steps data and merge results
def process_steps_data(df, steps):
    steps_data = {f"{steps}steps_{i}": df[df["key"] == f"{steps}steps_{i}"] for i in range(1, steps + 1)}
    steps_data[f"{steps}steps_multi"] = df[df["key"] == f"{steps}steps_multi"]
    for key, data in steps_data.items():
        data.columns = [col + f'_{key.split("_")[-1]}' for col in data.columns]
    merged_data = steps_data[f"{steps}steps_1"]
    for i in range(2, steps + 1):
        merged_data = pd.merge(merged_data, steps_data[f"{steps}steps_{i}"], left_on=f"ID_1", right_on=f"ID_{i}", how="left")
    merged_data = pd.merge(merged_data, steps_data[f"{steps}steps_multi"], left_on=f"ID_1", right_on="ID_multi", how="left")
    return merged_data


# Function to calculate evaluation metrics
def calculate_metrics(merged_2steps, merged_3steps):
    metrics = {}
    metrics["steps2_filtered_rows_1_loose"] = merged_2steps[((merged_2steps["joker_1"] == False) & (merged_2steps["joker_2"] == False)) & (merged_2steps["joker_multi"] == True)]
    metrics["steps2_filtered_rows_1_strict"] = merged_2steps[((merged_2steps["joker_1"] == False) | (merged_2steps["joker_2"] == False)) & (merged_2steps["joker_multi"] == True)]
    metrics["steps2_filtered_rows_2"] = merged_2steps[((merged_2steps["joker_1"] == True) & (merged_2steps["joker_2"] == True)) & (merged_2steps["joker_multi"] == False)]
    metrics["steps2_filtered_rows_3"] = merged_2steps[((merged_2steps["joker_1"] == False) | (merged_2steps["joker_2"] == False)) & (merged_2steps["joker_multi"] == False)]
    metrics["steps2_filtered_rows_4_loose"] = merged_2steps[((merged_2steps["joker_1"] == True) | (merged_2steps["joker_2"] == True)) & (merged_2steps["joker_multi"] == True)]
    metrics["steps2_filtered_rows_4_strict"] = merged_2steps[((merged_2steps["joker_1"] == True) & (merged_2steps["joker_2"] == True)) & (merged_2steps["joker_multi"] == True)]
    metrics["steps3_filtered_rows_1_loose"] = merged_3steps[((merged_3steps["joker_1"] == False) & (merged_3steps["joker_2"] == False) & (merged_3steps["joker_3"] == False)) & (merged_3steps["joker_multi"] == True)]
    metrics["steps3_filtered_rows_1_strict"] = merged_3steps[((merged_3steps["joker_1"] == False) | (merged_3steps["joker_2"] == False) | (merged_3steps["joker_3"] == False)) & (merged_3steps["joker_multi"] == True)]
    metrics["steps3_filtered_rows_2"] = merged_3steps[((merged_3steps["joker_1"] == True) & (merged_3steps["joker_2"] == True) & (merged_3steps["joker_3"] == True)) & (merged_3steps["joker_multi"] == False)]
    metrics["steps3_filtered_rows_3"] = merged_3steps[((merged_3steps["joker_1"] == False) | (merged_3steps["joker_2"] == False) | (merged_3steps["joker_3"] == False)) & (merged_3steps["joker_multi"] == False)]
    metrics["steps3_filtered_rows_4_loose"] = merged_3steps[((merged_3steps["joker_1"] == True) | (merged_3steps["joker_2"] == True) | (merged_3steps["joker_3"] == True)) & (merged_3steps["joker_multi"] == True)]
    metrics["steps3_filtered_rows_4_strict"] = merged_3steps[((merged_3steps["joker_1"] == True) & (merged_3steps["joker_2"] == True) & (merged_3steps["joker_3"] == True)) & (merged_3steps["joker_multi"] == True)]
    # metrics.to_csv("/Users/mac/Desktop/测试结果/error_anal/csv/gpt4o-0626.csv", index = False)
    return metrics


# Function to compute evaluation rates and final scores
def compute_final_scores(metrics, total_count):
    total_counts = {
        "InadequateGeneralization": len(metrics["steps2_filtered_rows_2"]) + len(metrics["steps3_filtered_rows_2"]),
        "InsufficientKnowledge": len(metrics["steps2_filtered_rows_3"]) + len(metrics["steps3_filtered_rows_3"]),
        "CompleteMastery_loose": len(metrics["steps2_filtered_rows_4_loose"]) + len(metrics["steps3_filtered_rows_4_loose"]),
        "CompleteMastery_strict": len(metrics["steps2_filtered_rows_4_strict"]) + len(metrics["steps3_filtered_rows_4_strict"]),
        "RoteMemorization_loose": len(metrics["steps2_filtered_rows_1_loose"]) + len(metrics["steps3_filtered_rows_1_loose"]),
        "RoteMemorization_strict": len(metrics["steps2_filtered_rows_1_strict"]) + len(metrics["steps3_filtered_rows_1_strict"]),
    }
    rates = {
        "InadequateGeneralization_rate": "{:.2%}".format(total_counts["InadequateGeneralization"] / total_count),
        "InsufficientKnowledge_rate": "{:.2%}".format(total_counts["InsufficientKnowledge"] / total_count),
        "CompleteMastery_loose_rate": "{:.2%}".format(total_counts["CompleteMastery_loose"] / total_count),
        "CompleteMastery_strict_rate": "{:.2%}".format(total_counts["CompleteMastery_strict"] / total_count),
        "RoteMemorization_loose_rate": "{:.2%}".format(total_counts["RoteMemorization_loose"] / (total_counts["CompleteMastery_loose"] + total_counts["RoteMemorization_loose"])),
        "RoteMemorization_strict_rate": "{:.2%}".format(total_counts["RoteMemorization_strict"] / (total_counts["CompleteMastery_strict"] + total_counts["RoteMemorization_strict"])),
    }
    return total_counts, rates


# Function to update main results DataFrame
def update_main_results_df(total_counts, rates):

    final_score_loose = "{:.2%}".format((525 - 0.5 * total_counts["InadequateGeneralization"] - total_counts["RoteMemorization_loose"] - total_counts["InsufficientKnowledge"]) / 525)
    final_score_strict = "{:.2%}".format((525 - 0.5 * total_counts["InadequateGeneralization"] - total_counts["RoteMemorization_strict"] - total_counts["InsufficientKnowledge"]) / 525)

    new_row = {
        "Score (Strict)": final_score_strict,
        "InsufficientKnowledge (Strict)": f"{rates['InsufficientKnowledge_rate']} ({total_counts['InsufficientKnowledge']})",
        "InadequateGeneralization (Strict)": f"{rates['InadequateGeneralization_rate']} ({total_counts['InadequateGeneralization']})",
        "CompleteMastery (Strict)": f"{rates['CompleteMastery_strict_rate']} ({total_counts['CompleteMastery_strict']})",
        "RoteMemorization (Strict)": f"{rates['RoteMemorization_strict_rate']} ({total_counts['RoteMemorization_strict']})",
        "Score (Loose)": final_score_loose,
        "InsufficientKnowledge (Loose)": f"{rates['InsufficientKnowledge_rate']} ({total_counts['InsufficientKnowledge']})",
        "InadequateGeneralization (Loose)": f"{rates['InadequateGeneralization_rate']} ({total_counts['InadequateGeneralization']})",
        "CompleteMastery (Loose)": f"{rates['CompleteMastery_loose_rate']} ({total_counts['CompleteMastery_loose']})",
        "RoteMemorization (Loose)": f"{rates['RoteMemorization_loose_rate']} ({total_counts['RoteMemorization_loose']})",
    }
    return new_row
