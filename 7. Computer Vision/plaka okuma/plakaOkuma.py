"""Konturlarla plaka bölgesi bulma; isteğe bağlı Tesseract OCR."""
from argparse import ArgumentParser
import os
from pathlib import Path
import shutil

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def main(haar=False):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=HERE / ("test3.png" if haar else "test4.png"))
    parser.add_argument("--ocr", action="store_true")
    args = parser.parse_args()
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        parser.error("Görüntü okunamadı.")
    if not haar:
        image = cv2.resize(image, (600, 400))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if haar:
        model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
        boxes = model.detectMultiScale(gray, 1.1, 3)
    else:
        smooth = cv2.bilateralFilter(gray, 13, 15, 15)
        edges = cv2.Canny(smooth, 30, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            polygon = cv2.approxPolyDP(contour, .018 * cv2.arcLength(contour, True), True)
            x, y, w, h = cv2.boundingRect(polygon)
            if len(polygon) == 4 and 2 < w / h < 7 and w >= 40:
                boxes.append((x, y, w, h))
                break
    if not len(boxes):
        parser.error("Plaka adayı bulunamadı.")
    reader = None
    if args.ocr:
        import pytesseract as reader
        exe = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
        if exe is None:
            parser.error("Tesseract bulunamadı; PATH veya TESSERACT_CMD ayarlayın.")
        reader.pytesseract.tesseract_cmd = exe
    for x, y, w, h in boxes:
        crop = gray[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print(f"Plaka adayı: x={x}, y={y}, w={w}, h={h}")
        if reader is not None:
            print("OCR:", reader.image_to_string(crop, config="--psm 7").strip())
        cv2.imshow("Kirpilmis plaka", crop)
        break
    cv2.imshow("Plaka tespiti", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
