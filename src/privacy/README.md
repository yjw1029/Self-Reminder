# Multi-step Jailbreaking Privacy Attacks on ChatGPT

## Download Enron Email dataset

We download Enron Email dataset from https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz.
```bash
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
tar zvfx enron_mail_20150507.tar.gz
```

## Collect Emaill
Collect frequente emails and infrequent emails
```bash
python get_emails.py
```