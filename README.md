# 🧠 Winstars AI Test Project  
**Author:** Oleksandr Novokhatskyi  
**Environment:** macOS + Python 3.9 + TensorFlow 2.15 + PyTorch (for NER)  

---

## 📘 Опис проєкту
Репозиторій містить два повноцінних машинних завдання:

1. **Task 1 – MNIST Image Classification**
2. **Task 2 – Named Entity Recognition + Image Classification Pipeline**

Основна мета — продемонструвати розуміння архітектури ML-проєктів, структуру коду, роботу з моделями глибокого навчання, а також вміння інтегрувати NLP і Computer Vision у спільний пайплайн.

---

## 🧩 Task 1 – MNIST Image Classification

### 🎯 Мета:
Навчити три різні моделі класифікувати рукописні цифри з датасету **MNIST**:
1. **Random Forest**
2. **Feed-Forward Neural Network (FFNN)**
3. **Convolutional Neural Network (CNN)**

### ⚙️ Реалізація:
- Усі три моделі реалізовані у вигляді **окремих класів** із загальним інтерфейсом `MnistClassifierInterface`.
- Структура проєкту дозволяє легко додавати нові моделі.
- Навчання та оцінка моделей проводяться у файлі `demo.ipynb` з візуалізацією результатів.

### 🧠 Основні моменти:
- Використано `Keras Sequential API`.
- Мережа CNN побудована з Conv2D, MaxPooling2D, Flatten, Dense.
- Порівняння моделей за точністю (`accuracy`) та часом навчання.
- Створена таблиця результатів у `pandas.DataFrame`.

### 📊 Результати:
| Model | Accuracy | Training Time |
|--------|-----------|----------------|
| Random Forest | ~96% | 25s |
| FFNN | ~98% | 8s |
| CNN | ~99% | 18s |

---

## 🧠 Task 2 – NER + Image Classification Pipeline

### 🎯 Мета:
Побудувати інтегрований пайплайн, який:
1. Зчитує текст (наприклад: *“There is a cat in the picture.”*)  
2. Визначає у тексті назву тварини (**NER**-модель).
3. Аналізує зображення за допомогою **CNN** і визначає, чи справді на зображенні ця тварина.
4. Повертає `True` або `False`.

---

### 🏗️ Структура проєкту
task_2/
│
├── dataset/
│   └── raw-img/              # Animal Image Dataset (10 класів)
│
├── vision/
│   ├── cnn_model_2.py        # CNN модель для класифікації зображень
│   ├── img_train.py          # Навчання моделі
│   └── img_infer.py          # Інференс моделі
│
├── ner/
│   ├── ner_train.py          # Навчання NER (HuggingFace Transformers)
│   └── ner_infer.py          # Інференс текстових запитів
│
├── artifacts/
│   ├── img_model/            # Збережені ваги CNN (.h5)
│   └── ner_model_torch/      # Ваги NER (.safetensors)
│
├── pipeline.py               # Фінальний інтеграційний скрипт
└── animals_eda.ipynb         # Аналіз датасету

### ⚙️ Етапи розробки

#### 1️⃣ Image Classification
- Використано **CNN** із шарами:
Conv2D → MaxPooling2D → Conv2D → MaxPooling2D → Flatten → Dense → Dropout → Dense

- Картинки масштабовано з `300x300` до `75x75`.
- Навчання виконувалось через `ImageDataGenerator` з `validation_split=0.2`.

#### 2️⃣ NER (Named Entity Recognition)
- Базована на **HuggingFace Transformers** (`BERT-base-cased`).
- Модель навчається на кастомному наборі коротких англомовних фраз із тегами типу `B-ANIMAL`.
- Збереження ваг відбувалося у форматі `.safetensors`.

#### 3️⃣ Pipeline Integration
- Текст обробляється через `ner_infer.py`.
- Зображення — через `img_infer.py`.
- Основна логіка в `pipeline.py`:
1. Витягує назву тварини з тексту.
2. Класифікує тварину на зображенні.
3. Порівнює результат і повертає `True` або `False`.

---

## ⚠️ Особливості та складнощі Task 2
Цей етап був технічно найскладнішим через:
- **несумісність TensorFlow, Keras, PyTorch, NumPy** між різними версіями;
- необхідність підібрати стабільну конфігурацію середовища;
- конфлікти залежностей при роботі з `transformers` та `tensorflow-macos`;
- перевантаження пам’яті при роботі з великими моделями.

✅ Остаточна стабільна конфігурація:
Python 3.9.24
TensorFlow 2.15.0
Keras 3.11.3
PyTorch 2.3.0
NumPy 1.26.4
opencv-python 4.10.0.84
transformers 4.41.2

---

## 🚀 Запуск проєкту

### 1️⃣ Клонування репозиторію
```bash
git clone https://github.com/novokhatskyi/winstars_tests.git
cd winstars_tests