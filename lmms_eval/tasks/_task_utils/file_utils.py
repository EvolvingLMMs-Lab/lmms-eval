import os


def generate_submission_file(file_name, args, subpath="submissions"):
    if args is None or args.output_path is None:
        # If no output path is specified, use current directory
        path = subpath
    else:
        path = os.path.join(args.output_path, subpath)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, file_name)
    return os.path.abspath(path)
