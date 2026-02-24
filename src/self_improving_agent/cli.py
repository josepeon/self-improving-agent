"""CLI entry point for the Self-Improving Agent."""

import argparse
import sys


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Self-Improving Agent — OH AI Infrastructure",
        prog="sia",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Future commands
    subparsers.add_parser("evaluate", help="Run evaluation harness")
    subparsers.add_parser("train", help="Run fine-tuning pipeline")
    subparsers.add_parser("serve", help="Start local model server")
    subparsers.add_parser("status", help="Show system status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch (to be implemented)
    print(f"Command '{args.command}' not yet implemented.")
    sys.exit(1)


if __name__ == "__main__":
    main()
