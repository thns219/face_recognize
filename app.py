import streamlit as st
import cv2
import numpy as np
import pickle
import time
from src.detect import load_detector, detect_faces
from src.embed import load_recognizer, get_embedding


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Face Recognition AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>

/* =========================================================
   HEADER
========================================================= */
[data-testid="stHeader"] {
    background: transparent;
}

/* =========================================================
   MAIN LAYOUT
========================================================= */
.block-container {
    padding-top: 1rem !important;
}

/* =========================================================
   BACKGROUND
========================================================= */
.stApp {
    background: linear-gradient(
        135deg,
        #eef2ff,
        #e0f2fe
    );
}

/* =========================================================
   TITLE
========================================================= */
.header {
    font-size: 2.8rem;
    font-weight: 800;
    color: #1e3a8a;
    margin-bottom: 1rem;
}

/* =========================================================
   REMOVE WHITE CARD
========================================================= */
.card {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* =========================================================
   ANIMATION
========================================================= */
.fade-in {
    animation: fadeIn 0.3s ease-in-out;
}

@keyframes fadeIn {

    from {
        opacity: 0;
    }

    to {
        opacity: 1;
    }
}

/* =========================================================
   RESULT BOX
========================================================= */
.result-box {

    padding: 12px;

    border-radius: 12px;

    margin-bottom: 10px;

    background: rgba(255,255,255,0.5);

    backdrop-filter: blur(8px);
}

/* =========================================================
   SIDEBAR
========================================================= */
[data-testid="stSidebar"] {
    background: #0f172a;
}

/* =========================================================
   SIDEBAR TEXT
========================================================= */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* =========================================================
   SIDEBAR WIDTH
========================================================= */
[data-testid="stSidebar"] {

    min-width: 320px !important;

    max-width: 320px !important;
}

/* =========================================================
   FIX TOGGLE BUTTON
========================================================= */
button[kind="header"] {

    background: #111827 !important;

    border: none !important;

    border-radius: 10px !important;

    width: 42px !important;

    height: 42px !important;

    margin-top: 8px !important;

    margin-left: 8px !important;

    z-index: 999999 !important;
}

/* =========================================================
   TOGGLE ICON
========================================================= */
button[kind="header"] svg {
    color: white !important;
}

/* =========================================================
   TOGGLE HOVER
========================================================= */
button[kind="header"]:hover {
    background: #1f2937 !important;
}

/* =========================================================
   BUTTON
========================================================= */
.stButton > button {

    background: #111827 !important;

    color: white !important;

    border: none !important;

    border-radius: 12px !important;

    font-weight: 600 !important;

    padding: 0.6rem 1rem !important;
}

/* =========================================================
   BUTTON TEXT
========================================================= */
.stButton > button p {
    color: white !important;
}

/* =========================================================
   BUTTON SPAN
========================================================= */
.stButton button span {
    color: white !important;
}

/* =========================================================
   BUTTON HOVER
========================================================= */
.stButton > button:hover {

    background: #1f2937 !important;

    color: white !important;
}

/* =========================================================
   REMOVE STREAMLIT MENU
========================================================= */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# CACHE
# =========================================================
@st.cache_resource
def get_models():
    return load_detector(), load_recognizer()


@st.cache_data
def load_database():

    try:

        with open("data/embeddings.pkl", "rb") as f:
            return pickle.load(f)

    except:

        st.error("Không load được database")
        return None


# =========================================================
# LOGIC
# =========================================================
def cosine_similarity(a, b):

    return float(np.dot(a, b.T).squeeze())


def recognize_image(
    image,
    detector,
    recognizer,
    database,
    threshold
):

    frame = image.copy()

    faces = detect_faces(detector, frame)

    results = []

    top1 = None
    top_score = -1

    if faces is not None:

        for face in faces:

            emb = get_embedding(
                recognizer,
                frame,
                face
            )

            if emb is None:
                continue

            best_name = "Unknown"
            best_score = -1

            for name, embeddings in database.items():

                for e in embeddings:

                    score = cosine_similarity(
                        emb,
                        e
                    )

                    if score > best_score:

                        best_score = score
                        best_name = name

            if best_score < threshold:
                best_name = "Unknown"

            x, y, w, h = list(
                map(int, face[:4])
            )

            # TOP 1
            if best_score > top_score:

                top_score = best_score
                top1 = (x, y, w, h)

            color = (
                (0,255,0)
                if best_name != "Unknown"
                else (0,0,255)
            )

            # FACE BOX
            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                color,
                2
            )

            # LABEL
            cv2.putText(
                frame,
                f"{best_name} ({best_score:.2f})",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            results.append(
                (best_name, best_score)
            )

    # HIGHLIGHT TOP-1
    if top1:

        x, y, w, h = top1

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (255,0,0),
            3
        )

    return frame, results


# =========================================================
# SESSION STATE
# =========================================================
if "run_cam" not in st.session_state:
    st.session_state.run_cam = False

if "logs" not in st.session_state:
    st.session_state.logs = []

if "show_log" not in st.session_state:
    st.session_state.show_log = False

if "active_faces" not in st.session_state:
    st.session_state.active_faces = {}

if "paused_faces" not in st.session_state:
    st.session_state.paused_faces = {}


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Settings")

threshold = st.sidebar.slider(
    "Threshold",
    0.3,
    0.9,
    0.6,
    0.05
)

mode = st.sidebar.radio(
    "Mode",
    [
        "Upload Image",
        "Webcam"
    ]
)

st.sidebar.markdown("---")

# =========================================================
# TOGGLE LOG
# =========================================================
if st.sidebar.button("📜 Xem Activity Log"):

    st.session_state.show_log = (
        not st.session_state.show_log
    )

# =========================================================
# SHOW LOG
# =========================================================
if st.session_state.show_log:

    st.sidebar.markdown(
        "### Activity Log"
    )

    for log in reversed(
        st.session_state.logs[-10:]
    ):

        st.sidebar.write(
            f"{log['time']} - "
            f"{log['name']} "
            f"({log['score']})"
        )

st.sidebar.markdown("---")

st.sidebar.write("Model: FaceNet")
st.sidebar.write("Detector: RetinaFace")


# =========================================================
# MAIN
# =========================================================
def main():

    detector, recognizer = get_models()

    database = load_database()

    if database is None:
        st.stop()

    # =====================================================
    # TITLE
    # =====================================================
    st.markdown(
        '<div class="header">'
        'Face Recognition AI'
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # LAYOUT
    # =====================================================
    col1, col2 = st.columns([2,1])

    # =====================================================
    # LEFT
    # =====================================================
    with col1:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        results = []

        # =================================================
        # UPLOAD IMAGE
        # =================================================
        if mode == "Upload Image":

            file = st.file_uploader(
                "Upload Image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ]
            )

            if file:

                image = cv2.imdecode(
                    np.frombuffer(
                        file.read(),
                        np.uint8
                    ),
                    1
                )

                with st.spinner(
                    "Detecting..."
                ):

                    frame, results = recognize_image(
                        image,
                        detector,
                        recognizer,
                        database,
                        threshold
                    )

                st.markdown(
                    '<div class="fade-in">',
                    unsafe_allow_html=True
                )

                st.image(
                    cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    ),
                    width="stretch"
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

        # =================================================
        # WEBCAM
        # =================================================
        else:

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:

                if st.button("▶ Start Webcam"):

                    st.session_state.run_cam = True

            with col_btn2:

                if st.button("⏹ Stop Webcam"):

                    st.session_state.run_cam = False

            FRAME_WINDOW = st.image([])

            if st.session_state.run_cam:

                cap = cv2.VideoCapture(0)

                while st.session_state.run_cam:

                    ret, frame = cap.read()

                    if not ret:

                        st.error(
                            "Không mở được webcam"
                        )

                        break

                    frame_out, results = recognize_image(
                        frame,
                        detector,
                        recognizer,
                        database,
                        threshold
                    )
                    
                    current_time = time.time()

                    # cập nhật khuôn mặt đang xuất hiện
                    for name, score in results:

                        if name == "Unknown":
                            continue

                        if name not in st.session_state.active_faces:

                            st.session_state.active_faces[name] = {
                                "start_time": current_time,
                                "last_seen": current_time
                            }

                        else:

                            st.session_state.active_faces[name]["last_seen"] = current_time

                    # kiểm tra mất mặt >10 giây
                    for name in list(st.session_state.active_faces.keys()):

                        last_seen = st.session_state.active_faces[name]["last_seen"]

                        if current_time - last_seen > 10:

                            start_time = st.session_state.active_faces[name]["start_time"]

                            st.session_state.paused_faces[name] = {
                                "start": time.strftime(
                                    "%H:%M:%S",
                                    time.localtime(start_time)
                                ),
                                "pause": time.strftime(
                                    "%H:%M:%S",
                                    time.localtime(current_time)
                                ),
                                "duration": round(
                                    current_time - start_time,
                                    1
                                )
                            }

                            del st.session_state.active_faces[name]

                    FRAME_WINDOW.image(
                        cv2.cvtColor(
                            frame_out,
                            cv2.COLOR_BGR2RGB
                        ),
                        width="stretch"
                    )

                    time.sleep(0.01)

                cap.release()

        # =================================================
        # SAVE LOG
        # =================================================
        for name, score in results:

            if name == "Unknown":
                continue

            if (
                len(st.session_state.logs) == 0
                or
                st.session_state.logs[-1]["name"] != name
            ):

                st.session_state.logs.append({

                    "time": time.strftime("%H:%M:%S"),

                    "name": name,

                    "score": round(score, 2)
                })

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

    # =====================================================
    # RIGHT
    # =====================================================
    with col2:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader("Results")
        st.markdown("---")
        st.subheader("Realtime Tracking")
        # ===============================
# ACTIVE FACES
# ===============================
        if st.session_state.active_faces:

            st.markdown("### 🟢 Present")

            for name, data in st.session_state.active_faces.items():

                duration = (
                    time.time()
                    - data["start_time"]
                )

                st.success(
                    f"""
        {name}

        Start:
        {time.strftime('%H:%M:%S', time.localtime(data['start_time']))}

        Duration:
        {duration:.1f}s
        """
                )

        # ===============================
        # PAUSED FACES
        # ===============================
        if st.session_state.paused_faces:

            st.markdown("### ⏸ Paused")

            for name, data in st.session_state.paused_faces.items():

                st.warning(
                    f"""
        {name}

        Start:
        {data['start']}

        Pause:
        {data['pause']}

        Duration:
        {data['duration']}s
        """
                )

        if results:

            st.metric(
                "Faces detected",
                len(results)
            )

            for name, score in results:

                color = (
                    "#16a34a"
                    if name != "Unknown"
                    else "#dc2626"
                )

                st.markdown(f"""
                <div class="result-box fade-in">
                    <b style="color:{color}">
                        👤 {name}
                    </b>
                    <br>
                    Confidence: {score:.2f}
                </div>
                """, unsafe_allow_html=True)

        else:

            st.info("No faces detected")

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()