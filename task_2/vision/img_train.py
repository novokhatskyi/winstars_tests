# task_2/vision/img_train.py

from keras.src.legacy.preprocessing.image import ImageDataGenerator  # ✅ новий імпорт
from cnn_model_2 import CNNModel
import tensorflow as tf
import os

# --- Системні налаштування для macOS ---
tf.config.set_visible_devices([], 'GPU')  # вимикає GPU, щоб уникнути mutex-помилки
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_METAL_ENABLE"] = "0"       # вимикає Metal backend на Mac
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def train():
    # 1. Підготовка генераторів зображень
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        'task_2/dataset/raw-img',
        target_size=(75, 75),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        'task_2/dataset/raw-img',
        target_size=(75, 75),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # 2. Ініціалізація моделі
    cnn_model = CNNModel(input_shape=(75, 75, 3), num_classes=10)
    cnn_model.build()

    # 3. Навчання
    history = cnn_model.model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        verbose=1
)

    # 4. Збереження навченої моделі
    cnn_model.model.save('task_2/artifacts/img_model/animal_cnn_model.h5')
    print("✅ Навчання завершено. Модель збережено у artifacts/img_model/")

# ---- Ключовий блок ----
if __name__ == "__main__":
    train()

"""Що таке ImageDataGenerator наскільки я зрозцмів. 
1. Це конвеєр подачі картинок у мережу.
2. Він на ходу читає файли з диска, масштабує до потрібного розміру, нормалізує, формує батчі(порції).
"""