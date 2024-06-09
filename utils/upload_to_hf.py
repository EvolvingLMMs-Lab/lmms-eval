import zipfile
import os

def chunk_zip(input_zip_path, output_folder, max_size_mb):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the zip file
    with zipfile.ZipFile(input_zip_path, 'r') as zfile:
        # Get a list of all archived file names from the zip
        all_files = zfile.namelist()
        total_size = 0
        count = 1

        # Create a new zip file with three-digit formatting
        chunk_zip = zipfile.ZipFile(os.path.join(output_folder, f'videos_chunked_{count:02d}.zip'), 'w')

        for file in all_files:
            # Get file size in MB
            file_size = zfile.getinfo(file).file_size / 1024 / 1024

            # Check if adding the file to current chunk will exceed the max size
            if total_size + file_size > max_size_mb and total_size > 0:
                # Close current chunk and start a new one
                chunk_zip.close()
                count += 1
                chunk_zip = zipfile.ZipFile(os.path.join(output_folder, f'videos_chunked_{count:02d}.zip'), 'w')
                total_size = 0

            # Write the file to the chunk
            
            chunk_zip.writestr(zfile.getinfo(file), zfile.read(file))
            total_size += file_size

        chunk_zip.close()

input_zip_path = 'data/vatex_test.zip'
output_folder = 'data/hf_upload'
max_size_mb = 25*1024 # for 25GB
chunk_zip(input_zip_path, output_folder, max_size_mb)