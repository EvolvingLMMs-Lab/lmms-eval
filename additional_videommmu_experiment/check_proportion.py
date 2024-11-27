import json

# Paths to your two JSON files
# file1 = 'eightframe_llavavid.json'  # Replace with your actual first JSON file path
# file2 = 'application_llavavid.json'  # Replace with your actual second JSON file path
file1 = 'application_gpt4o.json'  # Replace with your actual first JSON file path
file2 = 'image_gpt4o.json' 
# Load the IDs from each file
with open(file1, 'r') as f1, open(file2, 'r') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# Convert IDs lists to sets for set operations
ids1 = set(data1["correct_ids"])
ids2 = set(data2["correct_ids"])

# Calculate differences and intersection
ids_in_file1_not_in_file2 = ids1 - ids2
ids_in_file2_not_in_file1 = ids2 - ids1
ids_in_both_files = ids1 & ids2

# Print results
print("Summary of ID comparison:")
print(f"1. Total IDs in application but not in image: {len(ids_in_file1_not_in_file2)}")
print(f"   IDs: {sorted(ids_in_file1_not_in_file2)}")
print(f"2. Total IDs in image but not in application: {len(ids_in_file2_not_in_file1)}")
print(f"   IDs: {sorted(ids_in_file2_not_in_file1)}")
print(f"3. Total IDs in both files: {len(ids_in_both_files)}")
print(f"   IDs: {sorted(ids_in_both_files)}")
print(f"4. Total correct in Application {len(ids1)}")
print(f"5. Total correct in Image {len(ids2)}")
