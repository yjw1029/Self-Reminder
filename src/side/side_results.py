# calculate std and mean of w/ ChatGPT and w/o ChatGPT

import argparse
import glob
import json
from collections import defaultdict
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_pattern",
        type=str,
        default=None,
        help="The files for calculating results."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Given the results of a repeated experiment, calculate the mean and standard deviation
    args = parse_args()

    files = glob.glob(args.file_pattern)

    rslts = defaultdict(list)
    for file in files:
        with open(file, 'r') as f:
            result = json.load(f)

        for key, value in result.items():
            if isinstance(value, float):
                rslts[key].append(result[key])

    df = pd.DataFrame.from_dict(rslts)

    for col in df.columns:
        print(col)
        print(len(df[col]), df[col].std(), df[col].mean())