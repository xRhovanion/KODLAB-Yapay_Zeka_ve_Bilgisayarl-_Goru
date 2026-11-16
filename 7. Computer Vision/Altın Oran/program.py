"""Yüz işaret noktalarından geometrik oranlar; bir güzellik ölçütü değildir."""
from argparse import ArgumentParser
import os
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=HERE / "test.png")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "oranlar.png")
    args = parser.parse_args()
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        parser.error("Görüntü okunamadı.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if not len(faces):
        parser.error("Yüz bulunamadı.")
    faces = np.asarray([max(faces, key=lambda box: int(box[2]) * int(box[3]))],
                       dtype=np.int32)
    model_path = HERE.parent / "Yüz işlemleri" / "landmark.yaml"
    if not model_path.is_file():
        parser.error("Yüz işlemleri/landmark.yaml eksik.")
    facemark = cv2.face.createFacemarkLBF()
    previous = Path.cwd()
    try:
        os.chdir(model_path.parent)
        facemark.loadModel(model_path.name)
    finally:
        os.chdir(previous)
    found, shapes = facemark.fit(gray, faces)
    if not found:
        parser.error("Yüz işaret noktaları bulunamadı.")
    points = shapes[0][0]
    face_width = np.linalg.norm(points[0] - points[16])
    jaw_height = np.linalg.norm(points[27] - points[8])
    eye_span = np.linalg.norm(points[36] - points[45])
    mouth_width = np.linalg.norm(points[48] - points[54])
    phi = (1 + np.sqrt(5)) / 2
    for title, a, b in [("Yüz genişliği / burun kökü-çene", face_width, jaw_height),
                         ("Dış göz aralığı / ağız genişliği", eye_span, mouth_width)]:
        if b > 0:
            print(f"{title}: {a/b:.3f}; altın oran referansı {phi:.3f}")
    for x, y in np.rint(points).astype(int):
        cv2.circle(image, (int(x), int(y)), 2, (0, 180, 255), -1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        parser.error("Çıktı kodlanamadı.")
    encoded.tofile(args.output)
    print(f"68 işaret noktası; çıktı: {args.output}")


if __name__ == "__main__":
    main()
