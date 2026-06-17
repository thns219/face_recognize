import os
import cv2
import time
import pickle
import numpy as np

from src.detect import load_detector, detect_faces
from src.embed import load_recognizer, get_embedding


# =========================
# CONFIG
# =========================

DATASET_DIR = "data/dataset"

DB_PATH = "data/embeddings.pkl"

THRESHOLD = 0.5


# =========================
# COSINE SIMILARITY
# =========================

def cosine_similarity(a, b):

    a = a.flatten()
    b = b.flatten()

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# =========================
# EVALUATOR
# =========================

class Evaluator:

    def __init__(self):

        print("[INFO] Loading detector...")
        self.detector = load_detector()

        print("[INFO] Loading recognizer...")
        self.recognizer = load_recognizer()

        print("[INFO] Loading embeddings database...")

        with open(DB_PATH, "rb") as f:
            self.database = pickle.load(f)

    # =========================
    # PREDICT
    # =========================

    def predict(self, image):

        start_time = time.time()

        # Detect face
        faces = detect_faces(
            self.detector,
            image
        )

        # Không detect được mặt
        if faces is None or len(faces) == 0:

            return {
                "detected": False,
                "name": "Unknown",
                "score": 0,
                "time": time.time() - start_time
            }

        # Lấy embedding
        emb = get_embedding(
            self.recognizer,
            image,
            faces[0]
        )

        if emb is None:

            return {
                "detected": False,
                "name": "Unknown",
                "score": 0,
                "time": time.time() - start_time
            }

        best_name = "Unknown"

        best_score = -1

        # So sánh database
        for person_name, embeddings in self.database.items():

            for db_emb in embeddings:

                score = cosine_similarity(
                    emb,
                    db_emb
                )

                if score > best_score:

                    best_score = score

                    best_name = person_name

        # Threshold
        if best_score < THRESHOLD:

            best_name = "Unknown"

        process_time = time.time() - start_time

        return {
            "detected": True,
            "name": best_name,
            "score": best_score,
            "time": process_time
        }

    # =========================
    # EVALUATE
    # =========================

    def evaluate(self):

        if not os.path.exists(DATASET_DIR):

            print(
                f"[ERROR] Dataset not found: {DATASET_DIR}"
            )

            return

        total_images = 0

        correct = 0

        detected_faces = 0

        false_positive = 0

        false_negative = 0

        processing_times = []

        print("\n========== START EVALUATION ==========\n")

        # Duyệt từng người
        for person_name in os.listdir(DATASET_DIR):

            person_dir = os.path.join(
                DATASET_DIR,
                person_name
            )

            if not os.path.isdir(person_dir):
                continue

            # Duyệt từng ảnh
            for image_name in os.listdir(person_dir):

                image_path = os.path.join(
                    person_dir,
                    image_name
                )

                image = cv2.imread(image_path)

                if image is None:
                    continue

                total_images += 1

                result = self.predict(image)

                pred_name = result["name"]

                score = result["score"]

                process_time = result["time"]

                processing_times.append(
                    process_time
                )

                # Detect được mặt
                if result["detected"]:

                    detected_faces += 1

                # Đúng
                if pred_name == person_name:

                    correct += 1

                # False Negative
                elif pred_name == "Unknown":

                    false_negative += 1

                # False Positive
                else:

                    false_positive += 1

                print(
                    f"[{total_images}] "
                    f"GT={person_name} | "
                    f"PRED={pred_name} | "
                    f"SCORE={score:.4f} | "
                    f"TIME={process_time:.4f}s"
                )

        # =========================
        # METRICS
        # =========================

        accuracy = (
            correct / total_images
            if total_images > 0 else 0
        )

        detection_rate = (
            detected_faces / total_images
            if total_images > 0 else 0
        )

        avg_processing_time = (
            np.mean(processing_times)
            if len(processing_times) > 0 else 0
        )

        false_positive_rate = (
            false_positive / total_images
            if total_images > 0 else 0
        )

        false_negative_rate = (
            false_negative / total_images
            if total_images > 0 else 0
        )

        # =========================
        # RESULT
        # =========================

        print("\n========== RESULT ==========\n")

        print(f"Threshold: {THRESHOLD}")

        print(
            f"Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"Face Detection Rate: "
            f"{detection_rate * 100:.2f}%"
        )

        print(
            f"Average Processing Time: "
            f"{avg_processing_time:.4f} sec/image"
        )

        print(
            f"False Positive Rate: "
            f"{false_positive_rate * 100:.2f}%"
        )

        print(
            f"False Negative Rate: "
            f"{false_negative_rate * 100:.2f}%"
        )

        print("\n========== DETAIL ==========\n")

        print(f"Total Images: {total_images}")

        print(f"Correct Predictions: {correct}")

        print(f"Detected Faces: {detected_faces}")

        print(f"False Positive: {false_positive}")

        print(f"False Negative: {false_negative}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    evaluator = Evaluator()

    evaluator.evaluate()