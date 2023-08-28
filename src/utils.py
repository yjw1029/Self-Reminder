import openai
import os
import time
from openai.error import RateLimitError, InvalidRequestError


def set_openai():
    openai.api_key = os.environ["OPENAI_API_KEYS"]
    openai.api_type = "azure"
    openai.api_base = "https://llm4rec.openai.azure.com/"
    openai.api_version = "2022-12-01"
    openai.openai_engine = "gpt-35-turbo"


def set_openai_tmp():
    openai.api_key = os.environ["OPENAI_API_KEYS2"]


def create_prompt(system_message, messages):
    prompt = system_message
    message_template = "\n<|im_start|>{}\n{}\n<|im_end|>"
    for message in messages:
        prompt += message_template.format(message["sender"], message["text"])
    prompt += "\n<|im_start|>assistant\n"
    return prompt


def create_prompt_chat(system_message, messages):
    if system_message:
        prompt = [{"role": "system", "content": system_message}]
    else:
        prompt = []
    for message in messages:
        prompt.append({"role": message["sender"], "content": message["text"]})
    return prompt


def sent_prompt(
    prompts,
    engine="gpt-35-turbo",
    temperature=0,
    max_tokens=800,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["<|im_end|>"],
):
    success = False
    while not success:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
            )
            success = True
        except RateLimitError as e:
            print(e)
            time.sleep(1)
        except InvalidRequestError as e:
            print(e)
            success = True
            response = {"choices": []}
            # print(prompts)
        except Exception as e:
            print(e)

    rslts = [i["text"] for i in response["choices"]]
    return rslts


def sent_prompt_chat(
    prompts,
    model="gpt-3.5-turbo-0301",
    temperature=0,
    max_tokens=800,
    frequency_penalty=0,
    presence_penalty=0,
):
    success = False
    while not success:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            success = True
        except RateLimitError as e:
            print(e)
            time.sleep(1)
        except InvalidRequestError as e:
            print(e)
            success = True
            response = {"choices": []}
            # print(prompts)
        except Exception as e:
            print(e)

    rslts = [i["message"] for i in response["choices"]]
    return rslts
