import os
import random
import openai
from pathlib import Path
import json
from datasets import load_dataset, get_dataset_config_info
from openai.error import RateLimitError, InvalidRequestError
import time

from utils import set_openai, create_prompt, sent_prompt

set_openai()

# defining the system message
system_message_template = "<|im_start|>system\n{}\n<|im_end|>"
system_message = system_message_template.format("You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Knowledge cutoff: 2021-09. Current date: 2023-01-14.")


task_names = ["wnli", "cola", "mrpc", "sst2", "stsb", "mnli", "qqp", "qnli"]


task_prompts = {
    "wnli": """Read the following sentence pair and determine if the second sentence can be inferred from the first sentence or if it cannot be inferred.

Sentence 1: "{}"
Sentence 2: "{}"

A) Entailment (the second sentence can be inferred from the first sentence)
B) Not Entailment (the second sentence cannot be inferred from the first sentence)
Directly and concisely answer A or B.""",
    "cola": """For the following sentence, determine if it is grammatically acceptable or not:

Sentence: "{}"

A) Acceptable (the sentence is grammatically correct)
B) Unacceptable (the sentence contains grammatical errors)
Directly and concisely answer A or B.""",
    "mrpc": """Determine if the following two sentences are semantically equivalent or not:

Sentence 1: "{}"
Sentence 2: "{}"

A) Semantically Equivalent (the two sentences have the same meaning)
B) Not Semantically Equivalent (the two sentences do not have the same meaning)
Directly and concisely answer A or B.""",
    "sst2": """For the following sentence, determine if the sentiment expressed is positive or negative:

Sentence: "{}"

A) Positive (the sentiment expressed is positive)
B) Negative (the sentiment expressed is negative)
Directly and concisely answer A or B.""",
    "stsb": """Rate the semantic similarity between the following two sentences on a scale of 0 (completely dissimilar) to 5 (completely similar):

Sentence 1: "{}"
Sentence 2: "{}"

Semantic Similarity Score:
A) 0
B) 1
C) 2
D) 3
E) 4
F) 5
Directly and concisely answer A, B, C, D, E or F.""",
"mnli": """Given the statement below, determine if the provided hypothesis can be categorized as entailment, neutral, or contradiction based on the given statement.

Statement: "{}"
Hypothesis: "{}"

A) Entailment
B) Neutral
C) Contradiction
Directly and concisely answer A, B or C.""",
"qnli": """Read the following question and sentence and determine whether the sentence is the answer to the question.

Question: "{}"
Sentence: "{}"

A) Entailment (the sentence is the answer to the question)
B) Not Entailment (the sentence is not the answer to the question)
Directly and concisely answer A or B.""",
"qqp": """Determine if the following two questions are semantically equivalent or not:

Question 1: "{}"
Question 2: "{}"

A) Semantically Equivalent (the two questions have the same meaning)
B) Not Semantically Equivalent (the two questions do not have the same meaning)
Directly and concisely answer A or B.""",
"rte": """Read the following sentence pair and determine if the second sentence can be inferred from the first sentence or if it cannot be inferred.

Sentence 1: "{}"
Sentence 2: "{}"

A) Entailment (the second sentence can be inferred from the first sentence)
B) Not Entailment (the second sentence cannot be inferred from the first sentence)
Directly and concisely answer A or B."""
}

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


for task_name in task_to_keys:
    if task_name != "mnli":
        print(task_name, get_dataset_config_info("glue", task_name).splits["validation"].num_examples)
    else:
        print(task_name, get_dataset_config_info("glue", task_name).splits["validation_matched"].num_examples)



def create_glue_message(task_name, example):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    texts = (
            (example[sentence1_key],)
            if sentence2_key is None
            else (example[sentence1_key], example[sentence2_key])
        )
    task_prompt = task_prompts[task_name].format(*texts)
    return [{"sender": "user", "text": task_prompt}], example["label"]

def gen(task_name, already_exists_prompts):
    raw_datasets = load_dataset("glue", task_name)
    if task_name != "mnli":
        validation_dataset = raw_datasets["validation"]
    else:
        validation_dataset = raw_datasets["validation_matched"]
    print(task_name, len(validation_dataset))
    if len(validation_dataset) > 2000:
        indexes = list(range(len(validation_dataset)))
        rng = random.Random(2023)
        sample_indexes = rng.sample(indexes, 2000)
        validation_dataset = validation_dataset.select(sample_indexes)

    prompts = []
    labels = []
    indexes = []

    for i, example in enumerate(validation_dataset):
        message, label = create_glue_message(task_name, example)
        prompt= create_prompt(system_message, message)

        if prompt not in already_exists_prompts:
            prompts.append(prompt)
            labels.append(label)
            indexes.append(i)

        if len(prompts) == 20:
            yield prompts, labels, indexes
            prompts, labels, indexes = [], [], []
    if len(prompts) != 0:
        yield prompts, labels, indexes
        prompts, labels, indexes = [], [], []

for t in range(5):
    file_dir = Path(f"../glue_{t}")
    file_dir.mkdir(exist_ok=True)
    for task_name in task_names:
        file_name = file_dir / f"raw_{task_name}.jsonl"

        if file_name.exists():
            already_exists_prompts = []
            with open(file_name, "r") as f:
                for line in f:
                    rslt = json.loads(line.strip())
                    already_exists_prompts.append(rslt["prompt"])
            fout = open(file_name, "a")
        else:
            already_exists_prompts = []
            fout = open(file_name, "w")
        
        g = gen(task_name, already_exists_prompts)

        for prompts, labels, indexes in g:
            rslts = sent_prompt(prompts)
            
            for prompt, label, rslt, index in zip(prompts, labels, rslts, indexes):
                json_rslt = json.dumps({
                    "index": index, "prompt": prompt, "label": label, "rslt": rslt})
                fout.write(json_rslt + "\n")

