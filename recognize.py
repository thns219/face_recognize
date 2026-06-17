import cv2
import pickle
import numpy as np
import time
from .detect import load_detector, detect_faces
from .embed import load_recognizer, get_embedding


def cosine_similarity(a, b):
    return float(np.dot(a, b.T).squeeze())


def recognize(db_path="data/embeddings.pkl", threshold=0.5):
    detector = load_detector()
    recognizer = load_recognizer()

    try:
        with open(db_path, "rb") as f:
            database = pickle.load(f)
    except:
        print("❌ Không load được embeddings.pkl")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Không mở được camera")
        return

    print("🚀 Bắt đầu nhận diện... (ESC để thoát)")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(detector, frame)

        if faces is not None:
            for face in faces:
                emb = get_embedding(recognizer, frame, face)

                if emb is None:
                    continue

                best_name = "Unknown"
                best_score = -1

                for name, embeddings in database.items():
                    for e in embeddings:
                        score = cosine_similarity(emb, e)

                        if score > best_score:
                            best_score = score
                            best_name = name

                if best_score < threshold:
                    best_name = "Unknown"

                x, y, w, h = list(map(int, face[:4]))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{best_name} ({best_score:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        try:
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) == 27:
                break
        except cv2.error:
            # GUI not available, run in headless mode
            print("⚠️ GUI not available, running in headless mode for 100 frames")
            time.sleep(0.1)
            frame_count += 1
            if frame_count > 100:  # Exit after 100 frames in headless mode
                break

    cap.release()
    cv2.destroyAllWindows()