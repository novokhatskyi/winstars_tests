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