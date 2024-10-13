## Usage

### Upload Results

```sh
python upload_results.py -f <log_folder> -m <model_name> [-F]
```

`[-F]` means the script will automatically upload the results without human checking. Otherwise, the script will print the results and ask for confirmation before uploading.

Example:

```sh
python upload_results.py -f logs/0706_0959_model_outputs_gpt4v_model_args_c974bc -m gpt-4o -F
```
