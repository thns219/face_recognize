import cv2
import numpy as np

def load_recognizer(model_path="models/face_recognition.onnx"):
    recognizer = cv2.FaceRecognizerSF.create(model_path, "")
    return recognizer


def get_embedding(recognizer, image, face):
    try:
        aligned = recognizer.alignCrop(image, face)
        feature = recognizer.feature(aligned)

        # normalize (quan trọng → giảm nhầm)
        feature = feature / np.linalg.norm(feature)

        return feature
    except:
        return None