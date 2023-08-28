import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproduction."
    )
    parser.add_argument("--llm_config_file", type=str, default=None, help="LLM configs")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to export responses and evaluation results",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to input dataset.",
    )
    parser.add_argument(
        "--email_file",
        type=str,
        default="freq_emails.jsonl"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size to inference."
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default=None,
        help="The name of infer method"
    )
    parser.add_argument(
        "--defense_template_index",
        type=int,
        default=0,
        help="The defense template to use (different tone)."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume from previous stored file. If the file does not exist test from scracth.",
    )

    args = parser.parse_args()

    return args
