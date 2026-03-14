import os
import io
import json
import base64

import torch
import torchvision.transforms as transforms
from PIL import Image
from google.cloud import storage


GCS_BUCKET = os.environ.get("GCS_BUCKET", "sls-dev0099")
MODEL_PATH = os.environ.get("MODEL_PATH", "mobilenetV2.pt")
LABELS_PATH = os.environ.get("LABELS_PATH", "imagenet_labels.json")

MODEL_LOCAL_PATH = "/tmp/mobilenetV2.pt"

model = None
labels = None


def load_artifacts():
    global model, labels

    if model is not None and labels is not None:
        return

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    model_blob = bucket.blob(MODEL_PATH)
    model_blob.download_to_filename(MODEL_LOCAL_PATH)
    model = torch.jit.load(MODEL_LOCAL_PATH, map_location=torch.device("cpu"))
    model.eval()

    labels_blob = bucket.blob(LABELS_PATH)
    labels = json.loads(labels_blob.download_as_string())


def transform_image(image_bytes: bytes):
    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transformations(image).unsqueeze(0)


def get_prediction(image_bytes: bytes) -> int:
    load_artifacts()
    tensor = transform_image(image_bytes)
    with torch.no_grad():
        return model(tensor).argmax().item()


def build_response(status_code: int, payload: dict):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": True,
        },
        "body": json.dumps(payload),
    }


def extract_image_bytes(event):
    # GCP / Flask-style request
    if hasattr(event, "files") and event.files:
        files_dict = event.files.to_dict()
        if files_dict:
            first_file = next(iter(files_dict.values()))
            return first_file.read()

    # Lambda/local dict style: {"body": "...base64...", "isBase64Encoded": true}
    if isinstance(event, dict):
        body = event.get("body")
        if body is None:
            raise ValueError("No body found in event")

        if event.get("isBase64Encoded", False):
            return base64.b64decode(body)

        if isinstance(body, str):
            # allow plain base64 string even if flag is absent
            try:
                return base64.b64decode(body)
            except Exception:
                raise ValueError("Body was present but not valid base64")

    raise ValueError("Unsupported event format")


def handler(event, context=None):
    try:
        image_bytes = extract_image_bytes(event)
        prediction = get_prediction(image_bytes)
        label = labels[str(prediction)]
        return build_response(200, {"predicted": prediction, "label": label})
    except Exception as e:
        return build_response(500, {"error": repr(e)})
