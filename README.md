# Self-Reminder

## Contents
- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Results](#results)
- [License](./LICENSE)


## Overview
ChatGPT has demonstrated itself as a powerful AI tool and has garnered hundreds of millions of users. 
However, the recent emergence of Jailbreak Attacks poses a significant threat to the responsible and secure use of ChatGPT, as the carefully crafted Jailbreak prompts may circumvent ChatGPT's ethics safeguards and trigger harmful responses.
In this work, we explores the severe yet underexplored problems brought by Jailbreaks and corresponding defense techniques. 
We introduce a Jailbreak dataset with various types of Jailbreak prompts and malicious instructions.
We further draw inspiration from the psychological concept of self-reminder
and propose a simple yet effective defense technique called System-Mode Self-Reminder.

## Repo Contents
- [src](./src): source code to reproduce all results in the manuscript.
- [figures](./figures): figures used in `README.md`.


## System Requirements

### Hardware Requirements

To run this package, the following hardware specifications are recommended:

A standard computer with a stable and reliable internet connection is required to access the OpenAI API.

The package has been tested on a machine with the following specifications:

* Memory: 216GB
* Processor: AMD EPYC 7V13 64-Core Processor

### Software Requirement

The package has been tested and verified to work on 

* Linux: Ubuntu 20.04.6.

It is recommended to use this operating system for optimal compatibility.

Before installing the required Python dependencies and running the source code, ensure that you have the following software installed:

* Python: Version 3.7 or above is recommended. 
* pip or conda: Install the corresponding package manager for Python, depending on your preference. These package managers are used to install and manage the required Python packages.

Since this package requires access to the OpenAI API, you will need to register an account and obtain your `OPENAI_API_KEYS`. Please follow the instructions provided in the OpenAI documentation for registration and obtaining the API keys: [OpenAI Documentation](https://platform.openai.com/docs/introduction) or [Azure OpenAI Document](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/).
The code has been test with both OpenAI Services and Azure OpenAI Services on Linux.
Setup the your model configs of GPT models in src/config. Here is an example of `src/config/gpt4_azure.yaml`.
```bash
api_key: "[YOUR API KEY]"
api_type: "azure"
api_base: "[YOUR API Base]"
api_version: "[YOUR API VERSION]"
engine: "[YOUR API ENGINE]"
chat: True
llm_name: "gpt4_wosys"
``` 
Here is another example of `src/config/gpt4_openai.yaml`.
```bash
api_key: "[YOUR API KEY]"
api_type: "openai"
model: "gpt-4-0613"
chat: True
llm_name: "gpt4_wosys"
``` 

We also conduct experiments with LLAMA-2-Chat. Please apply for LLAMA-2 access on the [official meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and the [huggingface repo](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) to get the access token. Here is an example of `src/config/llama2_13b.yaml`.
```bash
load_8bit: False
model_name: meta-llama/Llama-2-13b-chat-hf
llm_name: "llama2-chat"
auth_token: "[YOUR AUTH TOKEN]"
``` 

## Installation Guide

### Requirements
We use docker to manage the experimental enviroments. Pull the following docker image to your local devices.
```bash
docker pull yjw1029/torch:2.0-llama-update2
```

We can also use `pip` or `conda` to install all following requirements
```bash
jsonlines
pandas
seaborn
scipy
tqdm
scikit-learn
nltk
openai==0.27.4
torch==2.0.1
datasets>=2.8.0
evaluate==0.3.0
accelerate>=0.15.0
transformers==4.30.2
deepspeed>=0.9.5
git+https://github.com/huggingface/peft
sentence_transformers
absl-py
rouge_score
sacrebleu

# (Optional) Command line tool for downloading data, intermediate results 
gdown
```

### Download Dataset
The dataset is released in the [Self-Reminder-Data](https://github.com/yjw1029/Self-Reminder-Data) repo. Clone this repo to access the dataset.
```bash
git clone git@github.com:yjw1029/Self-Reminder-Data.git
```

## Demo
### Self-Reminder Evaluation
Collect responses from ChatGPT/GPT4/LLAMA-2-Chat/Vicuna. Set different `--llm_config_file` to select different models. You can increase batch size according to your quota.
```bash
cd src

# w/o defense
python pia_attack.py --data_path your/data/path \
--output_path your/output/file --resume \
--seed {seed} --llm_config_file config/{model}.yaml --batch_size 1

# Set different indexes:
# 0: w/ self-reminder defense (3 difference tones)
# 1: w/ warn
# 2: w/ praise
# 3: w/ prefix
# 4: w/ suffix
# 5: w/ adaptive attack 1
# 6: w/ adaptive attack 2
python pia_defense.py --data_path your/data/path \
--output_path your/output/file --resume \
--seed {seed} --llm_config_file config/{model}.yaml --batch_size 1 \
--defense_template_index {template_index}
```

### Automatic Self-Reminder Generation
Collect responses of ChatGPT with the generated reminder prompts. We hard code the generated prompts in `evaluate_prompts.py`.
```bash
cd src/systematic

# Diferent indexes -- 0-4: Systematic Prompt 1-5
python evaluate_prompts.py --data_path your/data/path \
--output_path your/output/file \
--seed {seed} --llm_config_file ../config/{model}.yaml --batch_size 1 \
--resume --defense_template_index {template_index}
```

### Automatic Self-Reminder Optimization
Optimize the reminder prompt starting with different prompts as intial points.
```bash
cd src/systematic

# Diferent indexes -- 0: Self Reminder 1-5: Systematic Prompt 1-5
python apo.py --data_path your/data/path --resume \
--output_path your/output/file \
--optim_llm_config_file ../config/{optim_model}.yaml \
--eval_llm_config_file ../config/{eval_model}.yaml \
--cls_llm_config_file ../config/{cls_model}.yaml \
--batch_size 20 --defense_template_index {defense_template_index} \
--target_asr 0.05 --seed 0 --max_num_iters 10 \
--num_errors 5 --num_feedbacks 5 --steps_per_gradient 1
```

Collect responses of ChatGPT with the optmized reminder prompts. Currently we hard code the optimized prompt as the 5-th prompt in `evaluate_prompts.py`.
```bash
cd src/systematic

# Diferent indexes -- 5: Optimized prompt
python evaluate_prompts.py --data_path your/data/path \
--output_path your/output/file \
--seed {seed} --llm_config_file ../config/{model}.yaml --batch_size 1 \
--resume --defense_template_index 5
```


### Get Evaluation Predictions
Get predictions with watermark detection and ChatGPT classification
```bash
cd src
python classification.py --cls_prompt_path your/cls/prompt/path \
--result_path your/result/path --output_path your/output/path \
--llm_config_file config/{model}.yaml --batch_size 1 --resume
```

Human evaluate the samples with contradictory predictions.
Compute mean and standard deviation

### Side Effect Evaluation
Collect responses from ChatGPT.
```bash
cd src/side

# Diferent indexes -- 0: w/o defense 1: w Remind

# glue
python glue.py --data_path your/data/path --output_path your/output/path \
--seed {seed} --llm_config_file config/{model}.yaml --batch_size 1 \
--defense_template_index {template_index}
# WMT
python wmt.py --data_path your/data/path --output_path your/output/path \
--seed {seed} --llm_config_file config/{model}.yaml --batch_size 1 \
--defense_template_index {template_index}
# sQuaD
python squad.py --data_path your/data/path --output_path your/output/path \
--seed {seed} --llm_config_file config/{model}.yaml --batch_size 1 \
--defense_template_index {template_index}
# CNN / Daily Mail
python cnn_dm.py --data_path your/data/path --output_path your/output/path \
--seed {seed} --llm_config_file config/{model}.yaml --batch_size 1 \
--defense_template_index {template_index}
# XSum
python xsum.py --data_path your/data/path --output_path your/output/path \
--seed {seed} --llm_config_file config/{model}.yaml --batch_size 1 \
--defense_template_index {template_index}

```
Calculate performance
```bash
# glue
python glue_results.py --result_path your/result/path --output_path your/output/path
# WMT
python wmt_results.py --result_path your/result/path --output_path your/output/path
# sQuaD
python squad_results.py --result_path your/result/path --output_path your/output/path
# CNN / Daily Mail
python cnn_dm.py --result_path your/result/path --output_path your/output/path
# XSum
python xsum.py --result_path your/result/path --output_path your/output/path
```

* Compute mean and standard deviation
```bash
python python side_results.py --file_pattern your/results/file/pattern
```


### Defense against Privacy Attacks
* Collect response 
```bash
# Diferent indexes -- 0: w/o defense 1: w Remind
cd src/privacy

# DP
python infer_dp.py --data_path your/data/path \
--output_path your/output/path \
--seed {seed} --llm_config_file ../config/{model}.yaml --batch_size 1 \
--resume --defense_template_index {template_index} \
--method_name {method_name} --email_file [freq/infreq]_emails.jsonl

# JP
python infer_jp.py --data_path your/data/path \
--output_path your/output/path \
--seed {seed} --llm_config_file ../config/{model}.yaml --batch_size 1 \
--resume --defense_template_index {template_index} \
--method_name {method_name} --email_file [freq/infreq]_emails.jsonl

# MJP
python infer_mjp.py --data_path your/data/path \
--output_path your/output/path \
--seed {seed} --llm_config_file ../config/{model}.yaml --batch_size 1 \
--resume --defense_template_index {template_index} \
--method_name {method_name} --email_file [freq/infreq]_emails.jsonl
```

* Compute metrics
```bash
cd src/privacy

# Run frequency_type (2) * Attack type (3) * prompt type (2) times 
# hit@5
python stat_hit5.py --data_pattern file/pattern/of/repeat/exps
# parse score
python stat.py --data_pattern file/pattern/of/repeat/exps
```


## Results


Download all intermediate [results](https://drive.google.com/file/d/1FSjHBYCZjQumdibre_o2aR81iS6Vtdp7/view?usp=sharing
usp=drive_link) in our experiments.
```bash
gdown 1FSjHBYCZjQumdibre_o2aR81iS6Vtdp7
unzip results.zip
```

Compute metrics with the above scripts to get the results reported in out paper.