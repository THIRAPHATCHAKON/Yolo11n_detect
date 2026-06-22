from ultralytics import YOLO

model = YOLO("best.pt")

model.train(
    data="dataset_1/data.yaml",
    epochs=50,
    imgsz=960,
    batch=16,
    save=True,                # บันทึกโมเดล
    project="runs_plate",     # โฟลเดอร์ผลลัพธ์
    name="exp2"               # ชื่อรอบเทรน
)