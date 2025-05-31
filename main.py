import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

model = MobileNetV2(weights='imagenet')

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    x = np.array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    return decoded

# Основное окно
root = tk.Tk()
root.title("Распознавание кошек и собак")
root.geometry("600x600")

# Метка для отображения изображения
img_label = tk.Label(root)
img_label.pack(pady=10)

# Метка для результатаgit init
result_label = tk.Label(root, text="Выберите изображение", font=("Arial", 14))
result_label.pack(pady=10)

# Функция выбора файла и предсказания
def load_and_predict():
    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path:
        # Загрузка и отображение изображения
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(img)
        img_label.config(image=tk_img)
        img_label.image = tk_img

        # Предсказание
        preds = predict_image(file_path)
        # Выводим топ-3 предсказания
        top_preds = "\n".join([f"{label}: {prob*100:.2f}%" for (_, label, prob) in preds])
        result_label.config(text=top_preds)

# Кнопка для выбора изображения
btn = tk.Button(root, text="Выбрать изображение", command=load_and_predict)
btn.pack(pady=20)

root.mainloop()