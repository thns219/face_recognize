import os
import cv2
import pickle
from .detect import load_detector, detect_faces
from .embed import load_recognizer, get_embedding


def train(dataset_path="data/dataset", output_path="data/embeddings.pkl"):
    detector = load_detector()
    recognizer = load_recognizer()

    database = {}

    if not os.path.exists(dataset_path):
        print("❌ Dataset path không tồn tại!")
        return

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)

        if not os.path.isdir(person_path):
            continue

        print(f"🔄 Đang xử lý: {person}")
        embeddings = []

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ Không đọc được ảnh: {img_path}")
                continue

            faces = detect_faces(detector, img)

            if faces is None:
                print(f"⚠️ Không detect được mặt: {img_name}")
                continue

            emb = get_embedding(recognizer, img, faces[0])

            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) > 0:
            database[person] = embeddings
            print(f"✅ {person}: {len(embeddings)} embeddings")
        else:
            print(f"❌ {person}: không có dữ liệu hợp lệ")

    if len(database) == 0:
        print("❌ Không có dữ liệu để lưu!")
        return

    os.makedirs("data", exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(database, f)

    print("✅ Training xong → đã lưu embeddings.pkl")