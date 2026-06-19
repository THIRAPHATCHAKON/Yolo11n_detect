import os
import cv2

# ==========================
# CONFIG
# ==========================
DATASET_DIR = "img_50label"

IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
CLASSES_FILE = os.path.join(DATASET_DIR, "classes.txt")

# ==========================
# LOAD CLASSES
# ==========================
with open(CLASSES_FILE, "r", encoding="utf-8") as f:
    CLASSES = [line.strip() for line in f if line.strip()]

NUM_CLASSES = len(CLASSES)

# ==========================
# FIND IMAGES
# ==========================
image_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# ==========================
# VALIDATE DATASET
# ==========================
errors = []

image_bases = {
    os.path.splitext(f)[0]
    for f in image_files
}

label_files = [
    f for f in os.listdir(LABELS_DIR)
    if f.endswith(".txt")
]

label_bases = {
    os.path.splitext(f)[0]
    for f in label_files
}

# image without label
for base in sorted(image_bases):
    if base not in label_bases:
        errors.append(f"[NO LABEL] {base}")

# label without image
for base in sorted(label_bases):
    if base not in image_bases:
        errors.append(f"[NO IMAGE] {base}")

# validate label content
for label_file in label_files:

    path = os.path.join(LABELS_DIR, label_file)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"[FORMAT] {label_file}:{line_num}"
            )
            continue

        try:
            cls = int(parts[0])
            vals = list(map(float, parts[1:]))

        except:
            errors.append(
                f"[PARSE] {label_file}:{line_num}"
            )
            continue

        if cls < 0 or cls >= NUM_CLASSES:
            errors.append(
                f"[CLASS] {label_file}:{line_num} class={cls}"
            )

        for v in vals:
            if v < 0 or v > 1:
                errors.append(
                    f"[RANGE] {label_file}:{line_num}"
                )

# ==========================
# PRINT ERRORS
# ==========================
print("\n========== DATASET CHECK ==========\n")

if errors:
    for e in errors:
        print(e)

    print(f"\nTOTAL ERRORS: {len(errors)}")

else:
    print("NO ERRORS FOUND")

print("\n===================================\n")

# ==========================
# COLORS
# ==========================
colors = [
    (0,255,0),
    (255,0,0),
    (0,0,255),
    (255,255,0),
    (255,0,255),
    (0,255,255),
]

# ==========================
# DRAW LABELS
# ==========================
def draw_labels(img, label_path):

    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        return img

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        cls = int(parts[0])

        xc, yc, bw, bh = map(float, parts[1:])

        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)

        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)

        color = colors[cls % len(colors)]

        cv2.rectangle(
            img,
            (x1,y1),
            (x2,y2),
            color,
            2
        )

        label = CLASSES[cls]

        cv2.putText(
            img,
            label,
            (x1, max(20,y1-5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return img

# ==========================
# VIEWER
# ==========================
index = 0

while True:

    image_file = image_files[index]

    image_path = os.path.join(
        IMAGES_DIR,
        image_file
    )

    base = os.path.splitext(image_file)[0]

    label_path = os.path.join(
        LABELS_DIR,
        base + ".txt"
    )

    img = cv2.imread(image_path)

    if img is None:
        print("Cannot load:", image_path)
        index += 1
        continue

    img = draw_labels(img, label_path)

    info = f"{index+1}/{len(image_files)}  {image_file}"

    cv2.putText(
        img,
        info,
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.imshow("YOLO Dataset Viewer", img)

    key = cv2.waitKeyEx(0)

    # RIGHT
    if key in [2555904, ord('d'), ord('D')]:
        index = (index + 1) % len(image_files)

    # LEFT
    elif key in [2424832, ord('a'), ord('A')]:
        index = (index - 1) % len(image_files)

    # ESC
    elif key == 27:
        break

cv2.destroyAllWindows()