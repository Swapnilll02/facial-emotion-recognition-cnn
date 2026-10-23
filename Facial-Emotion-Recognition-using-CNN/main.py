import os
import sys
import argparse
from pathlib import Path

# Suppress TensorFlow verbose logs and oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
from tensorflow.keras.models import model_from_json


# =====================================================
# 1. PATH RESOLUTION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_JSON = BASE_DIR / "model.json"
MODEL_WEIGHTS = BASE_DIR / "model.h5"
CASCADE_ALT2_PATH = BASE_DIR / "haarcascade_frontalface_alt2.xml"
CASCADE_DEFAULT_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
IMAGES_DIR = BASE_DIR / "images"


# =====================================================
# 2. EMOTION CONSTANTS & PALETTE
# =====================================================

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Color map for each emotion (BGR format)
EMOTION_COLORS = {
    "Angry": (40, 40, 220),       # Bright Red
    "Disgust": (30, 140, 240),    # Orange
    "Fear": (180, 50, 180),       # Purple / Magenta
    "Happy": (30, 215, 80),       # Vibrant Green
    "Sad": (230, 140, 30),        # Deep Blue
    "Surprise": (255, 220, 0),    # Cyan / Sky
    "Neutral": (190, 190, 190)    # Soft Gray
}


# =====================================================
# 3. LOAD MODEL & FACE DETECTORS
# =====================================================

def load_emotion_model():
    print("[INFO] Loading CNN emotion recognition model...")

    if not MODEL_JSON.exists():
        raise FileNotFoundError(f"Model architecture file not found: {MODEL_JSON}")

    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"Model weights file not found: {MODEL_WEIGHTS}")

    with open(MODEL_JSON, "r", encoding="utf-8") as file:
        model_json = file.read()

    model = model_from_json(model_json)
    model.load_weights(str(MODEL_WEIGHTS))

    print("[SUCCESS] Emotion model loaded successfully!")
    print(f"  Input Shape:  {model.input_shape}")
    print(f"  Output Shape: {model.output_shape} (7 emotion classes)")
    return model


def load_face_detectors():
    """
    Loads both primary (alt2) and fallback (default) Haar cascades.
    """
    print("[INFO] Loading Haar Cascade face detectors...")

    detector_alt2 = None
    if CASCADE_ALT2_PATH.exists():
        d = cv2.CascadeClassifier(str(CASCADE_ALT2_PATH))
        if not d.empty():
            detector_alt2 = d

    detector_default = None
    if CASCADE_DEFAULT_PATH.exists():
        d = cv2.CascadeClassifier(str(CASCADE_DEFAULT_PATH))
        if not d.empty():
            detector_default = d

    if detector_alt2 is None and detector_default is None:
        raise RuntimeError("Could not load any Haar Cascade face detector XML files.")

    print(f"[SUCCESS] Face detectors loaded: alt2={'OK' if detector_alt2 else 'FAIL'}, default={'OK' if detector_default else 'FAIL'}")
    return detector_alt2, detector_default


def detect_faces(gray_frame, detector_alt2, detector_default):
    """
    Robust multi-stage face detection:
    1. Try alt2 detector on raw grayscale.
    2. Try alt2 detector on histogram-equalized grayscale (boosts indoor low-light / contrast).
    3. Try default frontalface detector on raw grayscale.
    4. Try default frontalface detector on histogram-equalized grayscale.
    """
    eq_gray = None

    # Stage 1: Alt2 on standard grayscale
    if detector_alt2 is not None:
        faces = detector_alt2.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50)
        )
        if len(faces) > 0:
            return faces

    # Stage 2: Alt2 on contrast-equalized grayscale
    if detector_alt2 is not None:
        eq_gray = cv2.equalizeHist(gray_frame)
        faces = detector_alt2.detectMultiScale(
            eq_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50)
        )
        if len(faces) > 0:
            return faces

    # Stage 3: Default detector
    if detector_default is not None:
        faces = detector_default.detectMultiScale(
            gray_frame,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(50, 50)
        )
        if len(faces) > 0:
            return faces

    # Stage 4: Default detector on contrast-equalized grayscale
    if detector_default is not None:
        if eq_gray is None:
            eq_gray = cv2.equalizeHist(gray_frame)
        faces = detector_default.detectMultiScale(
            eq_gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(50, 50)
        )
        if len(faces) > 0:
            return faces

    return []


# =====================================================
# 4. PREPROCESS FACE FOR MODEL PREDICTION
# =====================================================

def preprocess_face(face_crop, model):
    """
    Prepares cropped face image to match the FER2013 training pipeline:
    1. Grayscale resize to (48, 48)
    2. Scale pixels to [-1, 1] range: (pixel / 255.0 - 0.5) * 2.0
    3. Reshape into 4D batch tensor: (1, 48, 48, 1)
    """
    if len(face_crop.shape) == 3:
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    face_resized = cv2.resize(face_crop, (48, 48))

    # Float conversion & [-1, 1] normalization
    normalized = face_resized.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) * 2.0

    # Expand to (1, 48, 48, 1)
    tensor = np.expand_dims(normalized, axis=0)
    tensor = np.expand_dims(tensor, axis=-1)
    return tensor


# =====================================================
# 5. UI VISUALIZATIONS & HUD OVERLAYS
# =====================================================

def draw_corner_rect(img, x, y, w, h, color, thickness=2, corner_len=16):
    """
    Draws a sleek bounding box with highlighted corner brackets.
    """
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

    # Top-Left
    cv2.line(img, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(img, (x, y), (x, y + corner_len), color, thickness)
    # Top-Right
    cv2.line(img, (x + w, y), (x + w - corner_len, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + corner_len), color, thickness)
    # Bottom-Left
    cv2.line(img, (x, y + h), (x + corner_len, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - corner_len), color, thickness)
    # Bottom-Right
    cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)


def draw_face_prediction(frame, x, y, w, h, emotion, confidence):
    """
    Draws the bounding box and the primary emotion label badge.
    """
    color = EMOTION_COLORS.get(emotion, (0, 255, 0))
    draw_corner_rect(frame, x, y, w, h, color, thickness=3, corner_len=20)

    label = f"{emotion} {confidence:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Position label above bounding box (or inside top if near upper edge)
    badge_y1 = max(y - th - 12, 10)
    badge_y2 = badge_y1 + th + 10
    badge_x1 = max(x, 10)
    badge_x2 = min(badge_x1 + tw + 14, frame.shape[1] - 10)

    # Solid colored badge background
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
    # Text in bold white
    cv2.putText(
        frame,
        label,
        (badge_x1 + 6, badge_y2 - 6),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA
    )


def draw_probability_hud(frame, probabilities):
    """
    Renders a live horizontal bar chart showing probability breakdown for all 7 emotions.
    """
    panel_w = 210
    panel_h = 175
    panel_x = frame.shape[1] - panel_w - 12
    panel_y = 12

    # Semi-transparent dark background overlay
    sub_img = frame[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
    dark_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    cv2.addWeighted(sub_img, 0.45, dark_rect, 0.55, 0, sub_img)

    # Panel Title
    cv2.putText(
        frame,
        "EMOTION PROBABILITIES",
        (panel_x + 12, panel_y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),
        1,
        cv2.LINE_AA
    )

    bar_max_w = 95
    line_y = panel_y + 36

    for idx, emo in enumerate(EMOTIONS):
        prob = float(probabilities[idx])
        color = EMOTION_COLORS.get(emo, (180, 180, 180))

        # Emotion Name
        cv2.putText(
            frame,
            f"{emo:8s}",
            (panel_x + 10, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        # Bar background
        bar_x = panel_x + 72
        cv2.rectangle(
            frame,
            (bar_x, line_y - 8),
            (bar_x + bar_max_w, line_y - 1),
            (50, 50, 50),
            -1
        )

        # Active filled bar
        fill_w = int(prob * bar_max_w)
        if fill_w > 0:
            cv2.rectangle(
                frame,
                (bar_x, line_y - 8),
                (bar_x + fill_w, line_y - 1),
                color,
                -1
            )

        # Percentage Text
        cv2.putText(
            frame,
            f"{prob * 100:4.1f}%",
            (bar_x + bar_max_w + 6, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (240, 240, 240),
            1,
            cv2.LINE_AA
        )

        line_y += 19


# =====================================================
# 6. PROCESS IMAGE FILE
# =====================================================

def process_image(image_path, model, detector_alt2, detector_default, show_window=True, save_path=None):
    path = Path(image_path)
    if not path.exists():
        alt_path = IMAGES_DIR / path.name
        if alt_path.exists():
            path = alt_path
        else:
            print(f"[ERROR] Image not found: {image_path}")
            return None

    img = cv2.imread(str(path))
    if img is None:
        print(f"[ERROR] Could not decode image: {path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, detector_alt2, detector_default)

    print(f"\n[INFO] Image: {path.name} | Faces Detected: {len(faces)}")

    last_preds = None
    for i, (x, y, w, h) in enumerate(faces):
        roi = gray[y:y + h, x:x + w]
        tensor = preprocess_face(roi, model)
        preds = model.predict(tensor, verbose=0)[0]
        last_preds = preds

        idx = int(np.argmax(preds))
        emotion = EMOTIONS[idx]
        confidence = float(preds[idx]) * 100.0

        print(f"  Person {i + 1}: {emotion} ({confidence:.1f}%)")
        draw_face_prediction(img, x, y, w, h, emotion, confidence)

    if last_preds is not None and img.shape[1] >= 400:
        draw_probability_hud(img, last_preds)

    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"[INFO] Saved annotated result to: {save_path}")

    if show_window:
        win_title = f"Emotion Recognition - {path.name}"
        cv2.imshow(win_title, img)
        print("[INFO] Press any key in the window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


# =====================================================
# 7. REAL-TIME WEBCAM RECOGNITION
# =====================================================

def run_emotion_detection(model, detector_alt2, detector_default, camera_index=0):
    print(f"\n[INFO] Starting camera device (index {camera_index})...")

    # Try DirectShow for Windows for instant start, fall back to default
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        camera.release()
        print(f"[WARNING] Could not open camera {camera_index}.")
        print("[INFO] Falling back to sample test images in images/...")
        sample = IMAGES_DIR / "mb.jpg"
        if sample.exists():
            process_image(sample, model, detector_alt2, detector_default, show_window=True)
        return

    print("[SUCCESS] Webcam stream active!")
    print("--------------------------------------------------")
    print(" CONTROLS:")
    print("   [Q] or [ESC] : Quit")
    print("   [S]          : Save Screenshot with predictions")
    print("   [H]          : Toggle live probability HUD")
    print("--------------------------------------------------\n")

    show_hud = True
    save_counter = 1

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("[WARNING] Failed to grab frame from camera.")
                break

            # Mirror webcam horizontally for intuitive preview
            frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, detector_alt2, detector_default)

            primary_preds = None

            if len(faces) == 0:
                # User guidance banner when no face is found
                cv2.rectangle(frame, (12, 12), (360, 42), (20, 20, 20), -1)
                cv2.putText(
                    frame,
                    "Face: Searching... (Look at camera)",
                    (20, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (50, 200, 255),
                    1,
                    cv2.LINE_AA
                )
            else:
                for (x, y, w, h) in faces:
                    roi = gray[y:y + h, x:x + w]
                    try:
                        tensor = preprocess_face(roi, model)
                        preds = model.predict(tensor, verbose=0)[0]
                        if primary_preds is None:
                            primary_preds = preds

                        idx = int(np.argmax(preds))
                        emotion = EMOTIONS[idx]
                        confidence = float(preds[idx]) * 100.0

                        draw_face_prediction(frame, x, y, w, h, emotion, confidence)
                    except Exception as err:
                        print(f"[ERROR] Prediction error: {err}")

                # Status banner showing face count
                top_idx = int(np.argmax(primary_preds)) if primary_preds is not None else 0
                top_emo = EMOTIONS[top_idx]
                cv2.rectangle(frame, (12, 12), (320, 42), (20, 20, 20), -1)
                cv2.putText(
                    frame,
                    f"Face: Detected ({len(faces)}) | {top_emo}",
                    (20, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (50, 255, 100),
                    1,
                    cv2.LINE_AA
                )

            # Draw HUD bar chart if enabled and predictions available
            if show_hud and primary_preds is not None and frame.shape[1] >= 500:
                draw_probability_hud(frame, primary_preds)

            # Bottom controls banner
            cv2.rectangle(frame, (0, frame.shape[0] - 28), (frame.shape[1], frame.shape[0]), (15, 15, 15), -1)
            cv2.putText(
                frame,
                "Press 'Q' to quit | 'S' to save screenshot | 'H' toggle HUD",
                (12, frame.shape[0] - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

            cv2.imshow("Facial Emotion Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("s"):
                snap_path = f"screenshot_{save_counter}.jpg"
                cv2.imwrite(snap_path, frame)
                print(f"[INFO] Saved screenshot with predictions: {snap_path}")
                save_counter += 1
            elif key == ord("h"):
                show_hud = not show_hud

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Webcam session closed.")


# =====================================================
# 8. CLI ARGUMENT PARSER
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Facial Emotion Recognition using CNN (FER2013)"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to an image to test (e.g. images/mb.jpg)"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Webcam device index (default: 0)"
    )
    parser.add_argument(
        "--test-all", "-t",
        action="store_true",
        help="Run inference on all sample images in images/ directory"
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable interactive display window (for headless/batch usage)"
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        default=None,
        help="Save output image to specified path"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("   FACIAL EMOTION RECOGNITION USING CNN (FER2013)")
    print("=" * 60)

    model = load_emotion_model()
    detector_alt2, detector_default = load_face_detectors()

    if args.test_all:
        print("\n[INFO] Running batch test on sample images:")
        sample_files = [
            f for f in IMAGES_DIR.glob("*.*")
            if f.suffix.lower() in [".jpg", ".png", ".webp", ".jpeg"]
            and not f.name.endswith("emotion.PNG")
            and not f.name.endswith("emotions.PNG")
        ]
        for img_path in sorted(sample_files):
            process_image(
                img_path,
                model,
                detector_alt2,
                detector_default,
                show_window=False,
                save_path=BASE_DIR / f"result_{img_path.stem}.jpg"
            )
        print("\n[SUCCESS] Batch testing complete!")

    elif args.image:
        process_image(
            args.image,
            model,
            detector_alt2,
            detector_default,
            show_window=not args.no_window,
            save_path=args.save
        )

    else:
        run_emotion_detection(
            model,
            detector_alt2,
            detector_default,
            camera_index=args.camera
        )


if __name__ == "__main__":
    main()
