import json

# Paths to your JSON files
file1 = "perception_gpt4o.json"
file2 = "comprehension_gpt4o.json"
file3 = "application_gpt4o.json"
file4 = "w2r_gpt4o.json"
file5 = "r2w_gpt4o.json"

# Load the IDs from each file
with open(file1, "r") as f1, open(file2, "r") as f2, open(file3, "r") as f3, open(file4, "r") as f4, open(file5, "r") as f5:
    data1 = json.load(f1)
    data2 = json.load(f2)
    data3 = json.load(f3)
    data4 = json.load(f4)
    data5 = json.load(f5)

# Convert IDs lists to sets for set operations
ids1 = set(data1["correct_ids"])
ids2 = set(data2["correct_ids"])
ids3 = set(data3["correct_ids"])
ids4 = set(data4["ids"])
ids5 = set(data5["ids"])

# Calculate required groups of IDs
# 1. IDs in file 1, 2, and 3 together
ids_in_files_1_2_3 = ids1 & ids2 & ids3

# 2. IDs in file 4, 1, and 2 together
ids_in_files_4_1_2 = ids4 & ids1 & ids2

# 3. IDs in file 5 and in file 1 and file 2
ids_in_file5_not_in_files_1_2 = ids5 & ids1 & ids2

# Print results
print("Summary of ID comparison:")
print(f"1. Total IDs in file 1, file 2, and file 3 together: {len(ids_in_files_1_2_3)}")
print(f"   IDs: {sorted(ids_in_files_1_2_3)}")
print(f"2. Total IDs in file 4, file 1, and file 2 together: {len(ids_in_files_4_1_2)}")
print(f"   IDs: {sorted(ids_in_files_4_1_2)}")
print(f"3. Total IDs in file 5 and in file 1 and file 2 together: {len(ids_in_file5_not_in_files_1_2)}")
print(f"   IDs: {sorted(ids_in_file5_not_in_files_1_2)}")
print(len(ids5))
