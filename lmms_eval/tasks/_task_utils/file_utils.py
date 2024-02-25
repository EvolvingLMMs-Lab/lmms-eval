import os


def generate_submission_file(file_name, args):
    path = os.path.join(args.output_path, "submissions")
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, file_name)
    return os.path.abspath(path)
