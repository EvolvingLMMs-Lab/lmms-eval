"""Launch the evaluation server."""

import argparse
import sys

from lmms_eval.entrypoints.http_server import launch_server
from lmms_eval.entrypoints.server_args import ServerArgs


def prepare_server_args(argv: list[str]) -> ServerArgs:
    """Parse command line arguments and return ServerArgs."""
    parser = argparse.ArgumentParser(description="Launch LMMS-Eval HTTP Server")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args(argv)

    return ServerArgs(host=args.host, port=args.port)


def run_server(server_args: ServerArgs) -> None:
    """Run the evaluation server."""
    launch_server(server_args)


if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    except KeyboardInterrupt:
        print("\nServer shutdown by user")
