from datasets import Dataset, load_dataset

# Load the deduplicated VideoSearch dataset
videosearch_dataset = load_dataset("lmms-lab/VideoSearch", "deduplicated_combined_milestone", split="test")

# ID to be removed
id_to_remove = "validation_Biology_18"

# Filter out the row with the missing ID
filtered_rows = [row for row in videosearch_dataset if row["id"] != id_to_remove]

# Create a new dataset from the filtered rows
filtered_dataset = Dataset.from_list(filtered_rows)

# Save the filtered dataset locally or push it to Hugging Face hub
filtered_dataset.push_to_hub("lmms-lab/VideoSearch", "final_combined_milestone", split="test")

# Check and print the number of rows before and after filtering
print(f"Original dataset size: {len(videosearch_dataset)}")
print(f"Filtered dataset size: {len(filtered_dataset)}")
