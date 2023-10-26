# The script is used for construct jailbreak prompts for APO training
# The similar prompts are filtered to avoid train-test contamination and improve efficiency.

import requests
import json
import pandas as pd
import numpy as np
import argparse

from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help=""
    )
    parser.add_argument(
        "--prompt_itw_path",
        type=str,
        default=None,
        help=""
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to export evaluation results.",
    )
    parser.add_argument(
        "--filter_threshold",
        type=float,
        default=0.7,
        help="The threshold to filter similar prompts."
    )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    # load sentence transformer
    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # request prompts from jailbreakchat and the prompts we selected from [paper](https://arxiv.org/abs/2308.03825)
    url = 'https://www.jailbreakchat.com/api/getprompts'
    response = requests.get(url)
    all_prompts = {i["name"]: i["text"] for i in json.loads(response.text)}
    df = pd.read_csv(args.prompt_itw_path, index_col=0)

    for i in df.index:
        all_prompts[df.loc[i, "name"]] = df.loc[i, "prompt"]

    # load testing prompts we collected before
    exist_prompts = pd.read_csv(args.prompt_path, index_col=0)

    exist_prompt_names = list(exist_prompts["name"].values)
    all_prompt_names = list(all_prompts.keys())

    new_prompt_names = list(set(all_prompt_names) - set(exist_prompt_names))
    new_prompts = [all_prompts[i] for i in new_prompt_names]
    exist_prompts = list(exist_prompts["prompt"].values)

    # filter prompt similar to testing prompts
    new_embeddings = st_model.encode(new_prompts)
    exist_embeddings = st_model.encode(exist_prompts)

    sim = np.dot(new_embeddings, exist_embeddings.T)
    max_sim = np.max(sim, axis=-1)

    sel_indices = np.argwhere(max_sim < args.filter_threshold).flatten()

    sel_prompt_names = [new_prompt_names[i] for i in sel_indices]
    sel_prompts = [new_prompts[i] for i in sel_indices]

    # filter similar train prompts
    sel_prompts_embeddings = new_embeddings[sel_indices]
    sel_sim = np.dot(sel_prompts_embeddings, sel_prompts_embeddings.T)
    sel_sim = np.tril(sel_sim, k=-1)
    max_sel_sim = np.max(sel_sim, axis=-1)
    sel_indices = np.argwhere(max_sel_sim < args.filter_threshold).flatten()

    sel_prompt_names = [sel_prompt_names[i] for i in sel_indices]
    sel_prompts = [sel_prompts[i] for i in sel_indices]

    data = {"name": sel_prompt_names, "prompt": sel_prompts}

    df = pd.DataFrame.from_dict(data)
    df.to_csv(args.output_path)