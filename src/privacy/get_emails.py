import re
import jsonlines
from pathlib import Path
import random

email_path = Path("../../enron_email/maildir")


# Get name to email address dict
name_email_dict = {}

# parse 'To' emails addresses
for path in email_path.iterdir():
    for sub_folder in ["_sent_mail", "inbox"]:
        if not (path / sub_folder).exists():
            continue
        
        for file in (path / sub_folder).iterdir():
            if not file.is_file():
                continue
            try:
                with open(file, 'r') as f:
                    text = f.read()
            except:
                continue
            to_pattern = r"To:\s*([^X]+)"
            x_to_pattern = r"X-To:\s*([^\n]+)"

            # Find matches using the regular expressions
            to_match = re.search(to_pattern, text)
            x_to_match = re.search(x_to_pattern, text)

            if to_match:
                to_text = to_match.group(1)
                emails = re.findall(r"[\w.-]+@[\w.-]+", to_text)
                names = re.findall(r"[\w\s]+", x_to_match.group(1))
                
                if len(emails) != len(names):
                    continue
                for email, name in zip(emails, names):
                    name_email_dict[name] = email

# parse 'From' email addresses
for path in email_path.iterdir():
    for sub_folder in ["_sent_mail", "inbox"]:
        if not (path / sub_folder).exists():
            continue

        for file in (path / sub_folder).iterdir():
            if not file.is_file():
                continue
            try:
                with open(file, 'r') as f:
                    text = f.read()
            except:
                continue
            from_pattern = r"From:\s*([^X]+)"
            x_from_pattern = r"X-From:\s*([^\n]+)"

            # Find matches using the regular expressions
            from_match = re.search(from_pattern, text)
            x_from_match = re.search(x_from_pattern, text)

            if from_match:
                from_text = from_match.group(1)
                emails = re.findall(r"[\w.-]+@[\w.-]+", from_text)
                names = re.findall(r"[\w\s]+", x_from_match.group(1))
                
                if len(emails) != len(names):
                    continue
                for email, name in zip(emails, names):
                    name_email_dict[name] = email


# split the emails into frequent email groups (ends with enron.com) and infrequent email groups
frequent_emails = []
infrequent_emails = []
for name, email in name_email_dict.items():
    if name.lower() == name or len(name.split()) < 2:
        continue
    name = name.strip()
    email = email.strip()
    if email.endswith("enron.com"):
        frequent_emails.append({"name": name, "email": email})
    else:
        infrequent_emails.append({"name": name, "email": email})


# sample 100 emails for both group for testing
rng = random.Random(x=2023)

sam_freq_emails = rng.sample(frequent_emails, k=100)
sam_infreq_emails = rng.sample(infrequent_emails, k=100)


with jsonlines.open("../../data/freq_emails.jsonl", 'w') as writer:
    writer.write_all(sam_freq_emails)


with jsonlines.open("../../data/infreq_emails.jsonl", 'w') as writer:
    writer.write_all(sam_infreq_emails)