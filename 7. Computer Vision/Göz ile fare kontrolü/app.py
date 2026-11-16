"""Göz tespiti; kamera ve fare kontrolü ayrı seçeneklerle etkinleştirilir."""
from argparse import ArgumentParser
from pathlib import Path
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=HERE.parent / "Altın Oran" / "test.png")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--blink", action="store_true", help="Kırpma modeliyle skor hesapla")
    parser.add_argument("--mouse", action="store_true", help="Canlı kamera ile fareyi denetle")
    args = parser.parse_args()
    if args.mouse and not args.camera:
        parser.error("--mouse için --camera gerekir.")
    mouse = None
    blink = None
    if args.blink or args.mouse:
        from tensorflow import keras

        class LegacyDepthwiseConv2D(keras.layers.DepthwiseConv2D):
            @classmethod
            def from_config(cls, config):
                config = dict(config)
                if config.pop("groups", 1) != 1:
                    raise ValueError("Beklenmeyen eski DepthwiseConv2D grup değeri.")
                return super().from_config(config)

        model_path = HERE / "blink-model.h5"
        if not model_path.is_file():
            parser.error("blink-model.h5 eksik; kitap kaynak arşivinden ekleyin.")
        blink = keras.models.load_model(
            model_path, compile=False, custom_objects={"DepthwiseConv2D": LegacyDepthwiseConv2D}
        )
    if args.mouse:
        import pyautogui as mouse
        print("Fare kontrolü açık. Çıkış: Esc; acil durdurma: fareyi ekran köşesine götürün.")
    face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    cap = cv2.VideoCapture(0) if args.camera else None
    last_click = [0., 0.]
    try:
        if cap is not None and not cap.isOpened():
            parser.error("Kamera açılamadı.")
        while True:
            if cap is not None:
                success, frame = cap.read()
                if not success:
                    parser.error("Kameradan kare alınamadı.")
            else:
                frame = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    parser.error("Görüntü okunamadı.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_model.detectMultiScale(gray, 1.1, 4)
            eye_count = 0
            for x, y, w, h in sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[:1]:
                roi = gray[y:y+h, x:x+w]
                eyes = [r for r in eye_model.detectMultiScale(roi, 1.1, 4)
                        if r[1] + r[3] / 2 < h * .6]
                eyes = sorted(sorted(eyes, key=lambda r: r[2]*r[3], reverse=True)[:2],
                              key=lambda r: r[0], reverse=True)
                eye_count = len(eyes)
                for i, (ex, ey, ew, eh) in enumerate(eyes):
                    if blink is not None:
                        crop = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
                        values = cv2.resize(crop, (224, 224)).astype(np.float32)[None] / 255
                        score = float(np.asarray(blink(values, training=False)).ravel()[0])
                        print(f"Göz {i+1}: kırpma skoru {score:.3f}")
                        now = time.monotonic()
                        if mouse is not None and score >= .95 and now - last_click[i] > .6:
                            mouse.click(button=("left", "right")[i])
                            last_click[i] = now
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if mouse is not None and eye_count == 2:
                    sw, sh = mouse.size()
                    px = int((x+w/2) * sw/frame.shape[1])
                    py = int((y+h/2) * sh/frame.shape[0])
                    mouse.moveTo(max(1, min(sw-2, px)), max(1, min(sh-2, py)))
            if cap is None:
                print(f"Yüz: {len(faces)}, göz: {eye_count}; fare kontrolü kapalı.")
            cv2.imshow("Goz tespiti - Esc", frame)
            if cap is None:
                cv2.waitKey(0)
                break
            if cv2.waitKey(25) & 0xFF == 27:
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
