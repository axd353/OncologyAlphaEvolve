from __future__ import annotations

import argparse

from .runner import run_from_config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run tabular foundation model experiments with mixture, same-ancestry, "
            "and priority-function-curated context selection."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON config file.",
    )
    args = parser.parse_args()
    run_from_config_path(args.config)


if __name__ == "__main__":
    main()