import cv2

def load_detector(model_path="models/face_detection.onnx"):
    detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (320, 320)
    )
    return detector


def detect_faces(detector, image):
    if image is None:
        return None

    h, w = image.shape[:2]
    detector.setInputSize((w, h))

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    return faces