import torch
from transformers import AutoTokenizer, AutoModel

# 1) Load tokenizer + pretrained BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

model.eval()  # inference mode (important)

# 2) Example input (replace with log lines later)
texts = [
    "type=EXECVE user=root cmd=/bin/sh",
    "type=SYSCALL success=no exit=13",
]

# 3) Tokenize (text -> token IDs)
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

# 4) Run BERT
with torch.no_grad():
    outputs = model(**inputs)

# 5) Get embeddings
# last_hidden_state shape: [batch, tokens, hidden_dim]
last_hidden = outputs.last_hidden_state

# Use [CLS] token as sentence/log embedding
cls_embeddings = last_hidden[:, 0, :]  # shape: [batch, 768]

print("Embedding shape:", cls_embeddings.shape)
print("First embedding (first 5 values):")
print(cls_embeddings[0][:5])
