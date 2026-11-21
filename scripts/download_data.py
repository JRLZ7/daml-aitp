"""Download this course's private competition files; run from any directory."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("competition", help="Slug after /competitions/ in the Kaggle URL")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
    )
    parser.add_argument("--login", action="store_true", help="Prompt privately for your API token")
    args = parser.parse_args()
    if "/" in args.competition or not args.competition.strip():
        parser.error("Provide only the competition slug, not its full URL.")

    import kagglehub

    if args.login:
        kagglehub.login()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = kagglehub.competition_download(args.competition, output_dir=str(output))
    print(f"Downloaded competition files to: {path}")
    print("If images are still archived, extract them under this directory as described in data/README.md.")


if __name__ == "__main__":
    main()
