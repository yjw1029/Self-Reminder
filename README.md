# Self-Reminder
Python implementation of "Defending ChatGPT against Jailbreak Attack via1
Self-Reminder"

## Introduction
ChatGPT has demonstrated itself as a powerful AI tool and has garnered hundreds of millions of users. 
However, the recent emergence of Jailbreak Attacks poses a significant threat to the responsible and secure use of ChatGPT, as the carefully crafted Jailbreak prompts may circumvent ChatGPT's ethics safeguards and trigger harmful responses.
In this work, we explores the severe yet underexplored problems brought by Jailbreaks and corresponding defense techniques. 
We introduce a Jailbreak dataset with various types of Jailbreak prompts and malicious instructions.
We further draw inspiration from the psychological concept of self-reminder
and propose a simple yet effective defense technique called System-Mode Self-Reminder.


## Enviroments
Requirements
```
jsonlines
pandas
openai==0.27.0
datasets==2.10.1
evaluate==0.4.0
```

## Get Start
1. Jailbreak attack 

```bash
cd src

# w/o self-reminder defense
python jailbreak_attack.py

# w/ self-reminder defense (3 difference tones)
python jailbreak_w_defenfe.py

# w/ prefix defense
python jailbreak_w_prefix.py

# w/ suffix defense
python jailbreak_w_suffix.py

# adpative attack w/ defense
python adaptive.py
```

2. Evaluation on GLUE
```bash
cd src

# w/o defense
python glue_wo_defense.py

# w/ self-reminder defense
python glue_w_defense.py
```
