from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="dataset_1/data.yaml",  # dataset
    epochs=100,               # จำนวนรอบ
    imgsz=960,                # ขนาดภาพ
    batch=16,                 # จำนวนรูปต่อ batch
    workers=4,                # โหลดข้อมูล
    patience=20,              # early stop
    save=True,                # บันทึกโมเดล
    project="runs_plate",     # โฟลเดอร์ผลลัพธ์
    name="exp1"               # ชื่อรอบเทรน
)