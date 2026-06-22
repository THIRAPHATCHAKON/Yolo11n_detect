from ultralytics import YOLO
from paddleocr import TextRecognition
from thefuzz import process
import cv2
import sys

# =========================
# CONFIG
# =========================

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "img_4.jpg"

CONF_THRESHOLD = 0.4

# =========================
# YOLO
# =========================

model = YOLO("best.pt")

# =========================
# OCR
# =========================

ocr = TextRecognition(
    model_name="th_PP-OCRv5_mobile_rec"
)

# =========================
# CLASS
# =========================

class_names = {
    0: "car",
    1: "province",
    2: "plate_number",
    3: "license_plate"
}

READ_CLASSES = {"plate_number", "province"}

# BGR
class_colors = {
    "car":           (255, 80,  80),
    "license_plate": (80,  80,  255),
    "plate_number":  (0,   200, 255),
    "province":      (0,   200, 80),
}

# =========================
# 77 Provinces
# =========================

provinces = [
    "กรุงเทพมหานคร","กระบี่","กาญจนบุรี","กาฬสินธุ์","กำแพงเพชร",
    "ขอนแก่น","จันทบุรี","ฉะเชิงเทรา","ชลบุรี","ชัยนาท",
    "ชัยภูมิ","ชุมพร","เชียงราย","เชียงใหม่","ตรัง",
    "ตราด","ตาก","นครนายก","นครปฐม","นครพนม",
    "นครราชสีมา","นครศรีธรรมราช","นครสวรรค์","นนทบุรี","นราธิวาส",
    "น่าน","บึงกาฬ","บุรีรัมย์","ปทุมธานี","ประจวบคีรีขันธ์",
    "ปราจีนบุรี","ปัตตานี","พระนครศรีอยุธยา","พะเยา","พังงา",
    "พัทลุง","พิจิตร","พิษณุโลก","เพชรบุรี","เพชรบูรณ์",
    "แพร่","ภูเก็ต","มหาสารคาม","มุกดาหาร","แม่ฮ่องสอน",
    "ยโสธร","ยะลา","ร้อยเอ็ด","ระนอง","ระยอง",
    "ราชบุรี","ลพบุรี","ลำปาง","ลำพูน","เลย",
    "ศรีสะเกษ","สกลนคร","สงขลา","สตูล","สมุทรปราการ",
    "สมุทรสงคราม","สมุทรสาคร","สระแก้ว","สระบุรี","สิงห์บุรี",
    "สุโขทัย","สุพรรณบุรี","สุราษฎร์ธานี","สุรินทร์","หนองคาย",
    "หนองบัวลำภู","อ่างทอง","อำนาจเจริญ","อุดรธานี","อุตรดิตถ์",
    "อุทัยธานี","อุบลราชธานี"
]

# =========================
# FUZZY MATCH
# =========================

def predict_province(text):
    if not text:
        return ""
    result = process.extractOne(text, provinces)
    if result is None:
        return text
    province, score = result
    if score >= 75:
        return province
    return text

# =========================
# OCR
# =========================

def read_text(img):
    try:
        if img.size == 0:
            return "", 0
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        result = ocr.predict(img)
        if len(result):
            text = result[0]["rec_text"].strip()
            score = float(result[0]["rec_score"])
            return text, score
    except Exception as e:
        print("OCR ERROR:", e)
    return "", 0

# =========================
# DRAW LABEL
# =========================

def draw_label(frame, x1, y1, text, color):
    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = 0.55
    thickness   = 1
    pad         = 4

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # กล่องพื้นหลัง label
    cv2.rectangle(
        frame,
        (x1, y1 - th - pad * 2),
        (x1 + tw + pad * 2, y1),
        color, -1
    )

    # ข้อความสีขาว
    cv2.putText(
        frame, text,
        (x1 + pad, y1 - pad),
        font, font_scale,
        (255, 255, 255), thickness,
        cv2.LINE_AA
    )

# =========================
# LOAD IMAGE
# =========================

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print(f"Cannot open image: {IMAGE_PATH}")
    exit()

# =========================
# DETECT + OCR
# =========================

results = model(frame, conf=CONF_THRESHOLD, verbose=False)

for result in results:
    for box in result.boxes:

        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = class_names.get(cls_id, str(cls_id))
        color = class_colors.get(label, (200, 200, 200))

        # วาด box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if label not in READ_CLASSES:
            draw_label(frame, x1, y1, f"{label} {conf:.2f}", color)
            continue

        crop = frame[y1:y2, x1:x2]
        text, score = read_text(crop)

        if label == "province":
            text = predict_province(text)

        print(
            f"{label:<12} | "
            f"{text:<20} | "
            f"OCR={score:.2f} | "
            f"DET={conf:.2f}"
        )

        draw_label(frame, x1, y1, f"{label} {conf:.2f}", color)

        # OCR text ใต้ box
        if text:
            draw_label(frame, x1, y2 + 2, text, (30, 30, 30))

# =========================
# SHOW + SAVE
# =========================

cv2.imshow("Result", frame)
cv2.imwrite("result.jpg", frame)
print("Saved: result.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()