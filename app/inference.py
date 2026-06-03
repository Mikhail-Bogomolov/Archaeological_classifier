"""Единая точка инференса: археологический пайплайн или legacy MNIST."""

from __future__ import annotations

import io
import os
from typing import Any

from PIL import Image

CLASSIFIER_MODE = os.getenv("CLASSIFIER_MODE", "archaeology").lower()


def run_inference(contents: bytes, object_name: str | None = None) -> dict[str, Any]:
    if CLASSIFIER_MODE == "mnist":
        return _run_mnist(contents, object_name)
    from app.ml.pipeline import get_pipeline

    pipeline = get_pipeline()
    result = pipeline.predict(contents, object_name)
    d = pipeline.to_api_dict(result)
    if result.preprocess_meta.get("clahe_applied"):
        d.setdefault("features", []).append("Предобработка: CLAHE")
    return d


def _run_mnist(contents: bytes, object_name: str | None) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    import numpy as np

    class CNN_MNIST(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            x = self.conv_layers(x)
            return self.fc_layers(x)

    device = torch.device("cpu")
    model = CNN_MNIST().to(device)
    model.load_state_dict(
        torch.load("models/mnist_model.pth", map_location=device, weights_only=False)
    )
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    image = Image.open(io.BytesIO(contents))
    image_gray = image.convert("L")
    arr = np.array(image_gray)
    if np.mean(arr) > 127:
        arr = 255 - arr
        image_gray = Image.fromarray(arr)

    w, h = image_gray.size
    scale = 20.0 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image_resized = image_gray.resize((new_w, new_h), Image.LANCZOS)
    result = Image.new("L", (28, 28), 0)
    result.paste(image_resized, ((28 - new_w) // 2, (28 - new_h) // 2))

    tensor = transform(result).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    digit = int(predicted.item())
    display_name = object_name if object_name else "Новый объект"
    conf_pct = int(confidence.item() * 100)

    return {
        "name": display_name,
        "description": f"Распознанная цифра: {digit}",
        "category": "Тест (MNIST)",
        "confidence": conf_pct,
        "features": [f"Цифра: {digit}", f"Уверенность: {conf_pct}%"],
    }
