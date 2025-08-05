import argparse
import os

from .vqa import abstract_vqa
from .vqa import digest_vqa


def _start_question_cli(vqa: abstract_vqa.AbstractVqa):
    current_time: float | None = None

    print("Enter ':TIME' to set the time or ask a question.")
    print(
        "Example input: ':300' to set time to 300 seconds or 'What is the student doing?' to ask a question."
    )
    while True:
        print()
        try:
            user_input = input("$ ")
        except EOFError:  # Ctrl+D
            break

        if user_input.startswith(":"):
            # Set time.
            try:
                time_str = float(user_input[1:])
                current_time = time_str
                print(f"Time set to {current_time} seconds.")
            except ValueError:
                print("Invalid time format.")
                continue
        elif user_input:
            # Question.
            if current_time is None:
                print("Please set a time first using ':TIME'.")
                continue
            question = user_input
            # Our llm implementation already echoes the answer to stderr.
            _ = vqa.ask(current_time, question)


def main():
    parser = argparse.ArgumentParser(description="Video Quastion & Answer")
    parser.add_argument(
        "--video-path", type=str, help="Path to the video file.", required=True
    )
    args = parser.parse_args()

    vqa = digest_vqa.DigestVqa(video_path=args.video_path, maintain_context=True)

    os.system("clear")
    print()
    print(f"Video loaded: {os.path.basename(args.video_path)}")
    print()

    _start_question_cli(vqa)


if __name__ == "__main__":
    main()
