# -*- coding: utf-8 -*-
# PIPELINE: поєднання NER (DistilBERT) + CNN (Keras) для перевірки тверджень

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from transformers import AutoTokenizer, TFAutoModelForTokenClassification

# ==== 1. Шляхи до моделей ====
ROOT = os.path.dirname(os.path.abspath(__file__))
CNN_MODEL_PATH = os.path.join(ROOT, "artifacts/img_model/animal_cnn_model.h5")
NER_MODEL_PATH = os.path.join(ROOT, "artifacts/ner_model_torch")

# ==== 2. Завантаження моделей ====
print("🔹 Завантаження моделей...")
cnn_model = load_model(CNN_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
ner_model = TFAutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH, from_pt=True)

# ==== 3. Відповідність назв ====
# Італійські папки → англійські класи
ITALIAN_TO_ENGLISH = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}

# Класи в тому ж порядку, що й у навчанні CNN
CLASSES = list(ITALIAN_TO_ENGLISH.values())

# ==== 4. NER-функція ====
def extract_animal_from_text(text: str) -> str:
    """Витягує назву тварини з тексту"""
    tokens = tokenizer(
        text.split(),
        is_split_into_words=True,
        return_tensors="tf",
        truncation=True,
        padding=True
    )
    outputs = ner_model(tokens)
    logits = outputs.logits
    predictions = tf.math.argmax(logits, axis=-1).numpy()[0]

    id2label = ner_model.config.id2label
    labels = [id2label.get(int(p), "O") for p in predictions]
    words = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].numpy())

    for w, l in zip(words, labels):
        if l == "B-ANIMAL" and not w.startswith("##"):
            return w.lower()
    return None

# ==== 5. CNN-функція ====
def predict_animal_from_image(img_path: str) -> str:
    """Розпізнає тварину на зображенні"""
    img = image.load_img(img_path, target_size=(75, 75))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = cnn_model.predict(img_array, verbose=0)
    pred_class = np.argmax(preds, axis=1)[0]
    return CLASSES[pred_class]

# ==== 6. Перевірка твердження ====
def verify_statement(text: str, img_path: str) -> bool:
    """Перевіряє, чи збігається тварина з тексту та зображення"""
    animal_text = extract_animal_from_text(text)
    animal_image = predict_animal_from_image(img_path)

    print(f"\n📘 Текстова тварина: {animal_text}")
    print(f"🖼️  Тварина на зображенні: {animal_image}")

    if animal_text is None:
        print("⚠️ У тексті не знайдено жодної тварини.")
        return False

    match = animal_text in animal_image or animal_image in animal_text
    print(f"✅ Збіг: {match}\n")
    return match

# ==== 7. Автоматичний вибір італійських шляхів ====
def get_image_path(english_class: str) -> str:
    """Повертає шлях до випадкового зображення потрібного класу"""
    for italian, english in ITALIAN_TO_ENGLISH.items():
        if english == english_class:
            folder = os.path.join(ROOT, "dataset/raw-img", italian)
            images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not images:
                raise FileNotFoundError(f"No images found in {folder}")
            return os.path.join(folder, images[0])  # перше фото
    raise ValueError(f"Class {english_class} not found in mapping")

# ==== 8. Тест ====
if __name__ == "__main__":
    samples = [
        ("There is a cow in the picture.", get_image_path("cow")),
        ("I see a small cat on the sofa.", get_image_path("cat")),
        ("This image shows a dog running.", get_image_path("dog")),
        ("There is a horse near the tree.", get_image_path("horse")),
    ]

    for text, img in samples:
        verify_statement(text, img)