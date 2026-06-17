===========📌 Face Recognition (Nhận diện khuôn mặt)============

📖 Giới thiệu

Dự án này xây dựng một hệ thống nhận diện khuôn mặt sử dụng Python.
Hệ thống có khả năng:

- Phát hiện khuôn mặt trong ảnh hoặc video

- Trích xuất đặc trưng khuôn mặt (face encoding)

- So sánh và nhận diện danh tính

Ứng dụng phù hợp cho:

- Điểm danh

- Kiểm soát ra vào

- Nhận diện người trong ảnh/video

🚀 Tính năng chính

📷 Nhận diện khuôn mặt từ ảnh

🎥 Nhận diện realtime từ webcam (nếu có OpenCV)

🧠 So sánh khuôn mặt bằng vector encoding

⚡ Tốc độ xử lý nhanh với dữ liệu nhỏ - trung bình

🔧 Dễ mở rộng và tùy chỉnh

🧠 Công nghệ sử dụng

- Python 3.x

- face_recognition (dlib - deep learning)
  
- OpenCV

- NumPy

⚙️ Cài đặt

1. Clone project

- git clone https://github.com/tlinh5488/face_recognition.git
- cd face_recognition

2. Tạo môi trường ảo (khuyến nghị)

python -m venv venv

Kích hoạt:

Windows:

- venv\Scripts\activate

Linux / Mac:

- source venv/bin/activate

3. Cài thư viện

- pip install -r requirements.txt

⚠️ Cài dlib (quan trọng)

Nếu lỗi khi cài face_recognition, cài dlib trước:

Windows:

- pip install cmake

- pip install dlib

👉 Nếu vẫn lỗi: dùng file .whl phù hợp Python tại:

https://www.lfd.uci.edu/~gohlke/pythonlibs/

Ubuntu:
- sudo apt update

- sudo apt install build-essential cmake

- pip install dlib

▶️ Cách sử dụng

1. Chuẩn bị dữ liệu

- Tạo thư mục dataset/ (nếu chưa có)

- Thêm ảnh của những người cần nhận diện vào đây

👉 Mỗi người nên có nhiều ảnh để tăng độ chính xác

Ví dụ:

dataset/

├── person1.jpg

├── person2.jpg

2. Chạy chương trình

Chỉ cần chạy file chính:

- python main.py

hoặc nếu project dùng:

- python app.py

3. Quá trình hoạt động

- Khi chạy, chương trình sẽ:

- Đọc toàn bộ ảnh trong dataset/

- Tự động encode khuôn mặt ngay trong lúc chạy

- Mở webcam hoặc đọc ảnh/video

- So sánh khuôn mặt với dữ liệu đã có

- Hiển thị kết quả nhận diện

🎥 Nhận diện realtime (nếu có)

Nếu project hỗ trợ webcam:

python main.py

⚡ Tối ưu để chạy mượt (QUAN TRỌNG)

1. Giảm kích thước frame (tăng FPS)

Trong code OpenCV:

frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

👉 Giảm 4 lần kích thước → tăng tốc ~3–5x

2. Chỉ xử lý mỗi N frame

process_this_frame = not process_this_frame

👉 Giảm load CPU

3. Dùng GPU (nếu có)

- Cài dlib với CUDA:

- Cài CUDA Toolkit

- Build dlib với GPU

👉 Tăng tốc rất mạnh khi xử lý nhiều khuôn mặt

4. Giảm số lượng ảnh dataset

- Không nên >1000 ảnh nếu chạy CPU

- Mỗi người ~5 ảnh là đủ

5. Lưu encoding ra file

Tránh encode lại mỗi lần chạy:

- pickle.dump(data, f)

6. Điều chỉnh threshold

tolerance=0.6

- 0.4 → chính xác cao (khó match)

- 0.6 → mặc định

- 0.8 → dễ match hơn (có thể sai)

🔍 Cách hoạt động

- Phát hiện khuôn mặt

- Trích xuất đặc trưng (128-d vector)

- So sánh với dữ liệu đã lưu

- Nếu khoảng cách nhỏ → nhận diện

📊 Lưu ý

- Ánh sáng ảnh hưởng mạnh đến độ chính xác

- Góc mặt (nghiêng, che mặt) làm giảm hiệu quả

- Dataset càng đa dạng → nhận diện càng tốt

❗ Hạn chế

- Không tốt với ảnh mờ hoặc thiếu sáng

- Nhận diện trẻ em kém chính xác

- Không tối ưu cho hệ thống lớn

🚀 Hướng phát triển

- API (Flask / FastAPI)

- Web app / Mobile app

- Kết nối database

- Nhận diện nhiều camera

👤 Tác giả: 
tlinh5488

📜 License

Dự án phục vụ mục đích học tập và nghiên cứu.
