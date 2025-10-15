"""LSTM (Long Short-Term Memory)
Різновид рекурентної мережі (RNN), яка «пам’ятає» контекст послідовності слів довше завдяки коміркам пам’яті. 
Добре підходить для задач, де порядок слів важливий (текст, мова).

CRF (Conditional Random Field)
Ймовірнісна надбудова над токенами: замість того, щоб передбачати тег кожного слова незалежно, 
CRF враховує сусідні теги, щоб послідовність була узгодженою (наприклад, “B-ANIMAL” не може безпосередньо переходити в “I-PERSON”). 
В NER це часто дає чистіші, цілісніші підсвічені сутності.

LSTM + CRF разом
Типовий класичний стек для NER:
токени → ембедінги → BiLSTM (контекст) → CRF (узгоджені послідовності тегів).
Плюс: простіше, легше тренувати на CPU, мало залежностей. Мінус: гірше контекстні зв’язки на рівні фраз/довгих залежностей, 
ніж сучасні трансформери.

BERT (base)
Трансформер, який читає фразу двонапрямно (ліворуч+праворуч) і дає контекстні ембедінги для кожного токена. 
«base» — це стандартний розмір (≈110М параметрів). Для NER зазвичай беруть bert-base-cased/-uncased або україно-/мульти-мовні варіанти, 
і додають зверху тонкий класифікатор токенів.
Плюс: зазвичай точніший за LSTM+CRF, краще розуміє контекст. Мінус: важчий, повільніший."""

# -*- coding: utf-8 -*-
# PyTorch версія NER моделі (DistilBERT)

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from pathlib import Path
import numpy as np

# ==== 1) Дані ====
texts = [
    "The dog is barking loudly.",
    "A small dog runs across the street.",
    "The cat is sleeping on the sofa.",
    "A black cat jumps on the table.",
    "The horse is running fast.",
    "A white horse drinks water.",
    "A squirrel climbs a tree.",
    "The squirrel is eating a nut.",
    "A spider makes a web.",
    "The spider hangs from the wall.",
    "An elephant is big.",
    "The elephant walks in the field.",
    "A cow is eating grass.",
    "The farmer has a black cow.",
    "A chicken is near the barn.",
    "The chicken lays an egg.",
    "A butterfly sits on a flower.",
    "The butterfly is very colorful.",
    "A sheep grazes on the hill.",
    "The sheep follows the flock.",
]

tags = [
    ["O","B-ANIMAL","O","O","O"],
    ["O","O","B-ANIMAL","O","O","O","O","O"],
    ["O","B-ANIMAL","O","O","O","O","O","O"],
    ["O","O","B-ANIMAL","O","O","O","O","O"],
    ["O","B-ANIMAL","O","O","O"],
    ["O","O","B-ANIMAL","O","O"],
    ["O","B-ANIMAL","O","O","O"],
    ["O","O","B-ANIMAL","O","O","O","O"],
    ["O","B-ANIMAL","O","O","O"],
    ["O","O","B-ANIMAL","O","O","O","O","O"],
    ["O","B-ANIMAL","O","O"],
    ["O","O","B-ANIMAL","O","O","O","O"],
    ["O","B-ANIMAL","O","O","O"],
    ["O","O","O","O","B-ANIMAL","O","O"],
    ["O","B-ANIMAL","O","O","О","O"],
    ["O","O","B-ANIMAL","O","O","O"],
    ["O","B-ANIMAL","O","O","O","O"],
    ["O","O","B-ANIMAL","O","O","O"],
    ["O","B-ANIMAL","O","O","O","O","O"],
    ["O","O","B-ANIMAL","O","O","O","O"],
]
# Нормалізація — заміна кириличних "О" на латинські "O"
tags = [[t.replace("О", "O") for t in seq] for seq in tags]
label2id = {"O": 0, "B-ANIMAL": 1}
id2label = {v: k for k, v in label2id.items()}

# ==== 2) Токенізація ====
MODEL_NAME = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize_and_align_labels(texts, tags, max_length=64):
    encodings = tokenizer(
        [t.split() for t in texts],
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    all_labels = []
    for i in range(len(texts)):
        word_ids = encodings.word_ids(batch_index=i)
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[tags[i][wid]])
        all_labels.append(label_ids)

    labels = torch.tensor(all_labels)
    return encodings["input_ids"], encodings["attention_mask"], labels

input_ids, attention_mask, labels = tokenize_and_align_labels(texts, tags)

# ==== 3) Dataset ====
dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ==== 4) Модель ====
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
).to(device)

# ==== 5) Оптимізатор ====
optimizer = AdamW(model.parameters(), lr=3e-5)

# ==== 6) Навчання ====
model.train()
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")

# ==== 7) Збереження ====
out_dir = Path("task_2/artifacts/ner_model_torch")
out_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out_dir.as_posix())
tokenizer.save_pretrained(out_dir.as_posix())

print("✅ NER модель збережено у:", out_dir.resolve())