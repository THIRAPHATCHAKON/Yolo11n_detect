import cv2
from ultralytics import YOLO

model = YOLO("./best.pt")
results = model.predict(source="img.jpg", conf=0.25)

for r in results:
    img = r.plot()

    img = cv2.resize(img, (800, 600))  # 👈 บังคับขนาด

    cv2.imshow("YOLO Result", img)
    cv2.waitKey(0)