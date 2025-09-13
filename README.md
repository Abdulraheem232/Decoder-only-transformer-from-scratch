

---

````markdown
# Decoder-Only Transformer From Scratch (PyTorch Lightning)

A from-scratch implementation of a **decoder-only Transformer** (GPT-style) in **PyTorch Lightning**.  
This repo is designed for learning how causal self-attention, positional encodings, and token embeddings come together to form a next-token prediction model.

---

## 🚀 Features

- Custom implementation of:
  - Token embeddings + positional encoding  
  - Causal self-attention with masking  
  - Output projection head → vocabulary logits  
- Trains using **PyTorch Lightning** (no need for manual loops)  
- Supports Hugging Face `datasets` + `transformers` for easy preprocessing  
- Educational: minimal but extensible (add multi-head attention, feedforward layers, etc.)

---

## 📚 Motivation

Why build this?

- To understand Transformer **decoder blocks** step-by-step  
- To train/test on **small datasets** like [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)  
- To have a clean Lightning implementation that’s easy to extend into GPT-like models  

---

## 🔧 Requirements

- Python 3.8+  
- [PyTorch](https://pytorch.org/)  
- [PyTorch Lightning](https://www.pytorchlightning.ai/)  
- [Transformers](https://huggingface.co/transformers/)  
- [Datasets](https://huggingface.co/docs/datasets/)  

Install everything with:

```bash
pip install torch pytorch-lightning transformers datasets
````

---

## ⚙️ Setup & Usage

### 1. Clone the repo

```bash
git clone https://github.com/Abdulraheem232/Decoder-only-transformer-from-scratch.git
cd Decoder-only-transformer-from-scratch
```

### 2. Prepare data

For example, using **TinyStories**:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("roneneldan/TinyStories")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

max_len = 128
def preprocess(example):
    ids = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_len,
    )["input_ids"]
    labels = ids[1:] + [tokenizer.pad_token_id]
    return {"input_tokens": ids, "labels": labels}

ds = ds.map(preprocess, remove_columns=ds["train"].column_names)
```

Wrap with a DataLoader:

```python
import torch
from torch.utils.data import DataLoader

def collate(batch):
    inputs = torch.tensor([b["input_tokens"] for b in batch])
    labels = torch.tensor([b["labels"] for b in batch])
    return inputs, labels

train_loader = DataLoader(ds["train"], batch_size=32, shuffle=True, collate_fn=collate)
val_loader   = DataLoader(ds["validation"], batch_size=32, collate_fn=collate)
```

### 3. Initialize model

```python
from model import DecoderOnlyTransformer  # your class

model = DecoderOnlyTransformer(
    vocab_size=len(tokenizer),
    max_len=max_len,
    d_model=256
)
```

### 4. Train with Lightning

```python
from pytorch_lightning import Trainer

trainer = Trainer(
    max_epochs=3,
    accelerator="auto",  # uses GPU if available
    devices=1
)

trainer.fit(model, train_loader, val_loader)
```

---

## 🚧 Example Training Step (inside the model)

Your Lightning module implements `training_step` like:

```python
def training_step(self, batch, batch_idx):
    input_tokens, labels = batch
    logits = self.forward(input_tokens)  # [B, T, vocab_size]
    loss = self.loss(
        logits.view(-1, self.vocab_size),
        labels.view(-1)
    )
    self.log("train_loss", loss)
    return loss
```

---

## ✅ Roadmap

* [ ] Add **multi-head self-attention**
* [ ] Add **feedforward layers + layer norms**
* [ ] Proper handling of pad tokens in loss/mask
* [ ] Implement text generation utilities
* [ ] Experiment with larger datasets

---

## 🔗 References

* Vaswani et al. (2017) — *Attention Is All You Need*
* OpenAI GPT papers
* Hugging Face Datasets & Transformers
* [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

---

## 📬 Contributing

Contributions are welcome! If you add improvements (multi-head attention, generation, etc.), please include docs/examples so others can learn too.

---

## 📝 License

MIT (or update if you prefer another license).

```

---

```
