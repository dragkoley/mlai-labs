from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras.src import Functional
from keras.src.applications.resnet import preprocess_input, decode_predictions
from keras.src.utils import load_img, img_to_array


def predict_and_show(model: Functional, img_path: str | Path) -> None:
    img = load_img(path=img_path, target_size=(224, 224))
    x = img_to_array(img=img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    results = decode_predictions(preds=preds, top=5)[0]

    original_img = load_img(path=img_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(original_img)
    plt.axis("off")
    plt.title(f"Predicted: {results[0][1]} ({(results[0][2] * 100):.1f}%)", fontsize=20)
    plt.show()

    print("Top-5:")
    for rank, (_,label, score) in enumerate(results):
        print(f"{rank}. Class: {label}, Score: {score:.2f}")
