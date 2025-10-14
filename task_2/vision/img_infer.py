import tensorflow as tf
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

# === 1. Load trained model ===
MODEL_PATH = 'task_2/artifacts/img_model/animal_cnn_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model successfully loaded:", MODEL_PATH)

# === 2. Class labels (translated to English) ===
class_labels = [
    'dog',        # cane
    'horse',      # cavallo
    'elephant',   # elefante
    'butterfly',  # farfalla
    'chicken',    # gallina
    'cat',        # gatto
    'cow',        # mucca
    'sheep',      # pecora
    'spider',     # ragno
    'squirrel'    # scoiattolo
]

# === 3. Image preprocessing ===
def prepare_image(image_path, target_size=(75, 75)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"File {image_path} not found or cannot be opened.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_norm = img_resized.astype("float32") / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)
    return img_rgb, img_batch

# === 4. Prediction ===
def predict_image(image_path):
    img_rgb, img_batch = prepare_image(image_path)
    preds = model.predict(img_batch)
    class_index = np.argmax(preds)
    confidence = np.max(preds)
    label = class_labels[class_index]

    plt.imshow(img_rgb)
    plt.title(f"Prediction: {label} ({confidence:.2f})")
    plt.axis("off")
    plt.show()

    print(f"🔍 Predicted: {label} ({confidence:.2f})")

# === 5. Pick random image automatically ===
def get_random_image():
    base_dir = "task_2/dataset/raw-img"
    categories = os.listdir(base_dir)
    random_class = random.choice(categories)
    class_path = os.path.join(base_dir, random_class)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        raise FileNotFoundError(f"No images found in {class_path}")
    random_image = random.choice(images)
    return os.path.join(class_path, random_image)

# === 6. Run prediction ===
if __name__ == "__main__":
    TEST_IMAGE = get_random_image()
    print(f"🖼 Selected image: {TEST_IMAGE}")
    predict_image(TEST_IMAGE)