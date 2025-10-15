# -*- coding: utf-8 -*-
# NER_INFER — інференс для натренованої NER-моделі (PyTorch)

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ==== 1. Завантажуємо модель і токенайзер ====
MODEL_PATH = "task_2/artifacts/ner_model_torch"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

# ==== 2. Функція для прогнозу ====
def predict_entities(text):
    """
    Приймає один рядок тексту.
    Розбиває його на токени, проганяє через модель і повертає знайдені сутності.
    """
    tokens = tokenizer(
        text.split(),
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits  # [batch_size, seq_len, num_labels]
        predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    # перетворюємо id → назви тегів
    id2label = model.config.id2label
    labels = [id2label.get(int(p), "O") for p in predictions]

    # декодуємо токени для зручності
    words = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].cpu().numpy())

    # вивід
    print("\n🟩 Аналіз речення:")
    print("Текст:", text)
    print("-" * 60)
    for w, label in zip(words, labels):
        if label != "O" and not w.startswith("##"):
            print(f"{w:<15} → {label}")
    print("-" * 60, "\n")


# ==== 3. Тести ====
if __name__ == "__main__":
    examples = [
        "The dog is barking loudly.",
        "A black cat jumps on the table.",
        "A butterfly sits on a flower.",
        "The elephant walks in the field.",
        "A small squirrel climbs a tree.",
        "The sheep follows the flock."
    ]

    for ex in examples:
        predict_entities(ex)