import os
from datasets import load_dataset

# Load the VideoSearch dataset
videosearch_dataset = load_dataset('lmms-lab/VideoSearch', 'final_combined_milestone', split='test')

# Path to the videos directory (replace with your actual path)
videos_directory = '/mnt/sfs-common/krhu/.cache/huggingface/Combined_milestone/videos/'

# Get all IDs from the dataset
videosearch_ids = set(videosearch_dataset['id'])

# List to store IDs of files that are not in the dataset
extra_files = []

# Loop through all .mp4 files in the videos directory
for file in os.listdir(videos_directory):
    if file.endswith('.mp4'):
        # Extract the ID from the file name (remove the .mp4 extension)
        file_id = file.replace('.mp4', '')
        
        # Check if the file ID exists in the VideoSearch dataset
        if file_id not in videosearch_ids:
            extra_files.append(file_id)

# Print the IDs of .mp4 files that are not in the dataset
if extra_files:
    print(f"MP4 files not included in the VideoSearch dataset: {extra_files}")
else:
    print("All MP4 files have corresponding entries in the VideoSearch dataset.")

# Optionally, print the total number of extra files
print(f"Total extra MP4 files: {len(extra_files)}")
