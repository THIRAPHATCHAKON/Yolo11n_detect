from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

results = model("5.jpg")

annotated_img = results[0].plot()

h, w = annotated_img.shape[:2]

scale = min(1280 / w, 720 / h)

show_img = cv2.resize(
    annotated_img,
    (int(w * scale), int(h * scale))
)

cv2.imshow("YOLO Result", show_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("result.jpg", annotated_img)