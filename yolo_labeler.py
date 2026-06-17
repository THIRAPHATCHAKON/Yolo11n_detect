import os, shutil, json, tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from PIL import Image, ImageTk

COLORS = [
    "#e85d5d","#e8a83a","#4caf50","#2196f3","#9c27b0",
    "#00bcd4","#ff5722","#607d8b","#f06292","#8bc34a"
]

BG        = "#1a1a2e"
PANEL_BG  = "#16213e"
SIDEBAR_BG= "#0f3460"
ACCENT    = "#2196f3"
TEXT      = "#e0e0e0"
TEXT_DIM  = "#888888"
BORDER    = "#2a2a4a"
BTN_BG    = "#1e2a4a"
BTN_HOV   = "#2a3a5a"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Labeler")
        self.root.configure(bg=BG)
        self.root.geometry("1400x860")

        self.images   = []
        self.index    = 0
        self.labels   = {}
        self.classes  = ["car", "province", "plate_number", "license_plate"]
        self.selected_box = -1

        self.orig_w = 1
        self.orig_h = 1
        self.scale  = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # zoom
        self.zoom      = 1.0
        self.zoom_min  = 0.1
        self.zoom_max  = 8.0
        self.pan_x     = 0
        self.pan_y     = 0
        self._pan_start= None

        self.drawing   = False
        self.start_cx  = 0
        self.start_cy  = 0
        self.temp_rect = None
        self.cur_class = tk.StringVar(value="car")

        self._build_ui()
        self._refresh_classes()

    # ─────────────── UI BUILD ───────────────
    def _build_ui(self):
        # ── Toolbar ──
        toolbar = tk.Frame(self.root, bg=PANEL_BG, pady=6)
        toolbar.pack(fill="x", side="top")

        self._btn(toolbar, "📂  เพิ่มรูปภาพ", self.open_files).pack(side="left", padx=(8,4))
        self._sep(toolbar)
        tk.Label(toolbar, text="Class:", bg=PANEL_BG, fg=TEXT_DIM, font=("Segoe UI",10)).pack(side="left", padx=(8,2))
        self.class_combo = ttk.Combobox(toolbar, textvariable=self.cur_class,
                                        values=self.classes, width=16,
                                        font=("Segoe UI",10), state="readonly")
        self.class_combo.pack(side="left", padx=2)
        self.class_combo.bind("<<ComboboxSelected>>", lambda e: self._update_color_dot())
        self.color_dot = tk.Label(toolbar, text="  ", bg=COLORS[0], width=2, relief="flat")
        self.color_dot.pack(side="left", padx=4)
        self._sep(toolbar)

        # zoom controls
        tk.Label(toolbar, text="Zoom:", bg=PANEL_BG, fg=TEXT_DIM, font=("Segoe UI",10)).pack(side="left", padx=(8,2))
        self._btn(toolbar, "－", self._zoom_out, width=3).pack(side="left", padx=1)
        self.zoom_label = tk.Label(toolbar, text="100%", bg=PANEL_BG, fg=TEXT,
                                   font=("Segoe UI",10), width=5)
        self.zoom_label.pack(side="left")
        self._btn(toolbar, "＋", self._zoom_in, width=3).pack(side="left", padx=1)
        self._btn(toolbar, "⊡  Fit", self._zoom_fit, width=6).pack(side="left", padx=(2,4))
        self._btn(toolbar, "1:1", self._zoom_reset, width=4).pack(side="left", padx=2)
        self._sep(toolbar)

        self._btn(toolbar, "↩ Undo", self.undo_last).pack(side="left", padx=4)
        self._btn(toolbar, "🗑 ล้าง", self.clear_current).pack(side="left", padx=4)
        self._btn(toolbar, "💾  Export YOLO", self.export_yolo, accent=True).pack(side="right", padx=8)

        # ── Main area ──
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        # Left sidebar — file list
        left = tk.Frame(main, bg=SIDEBAR_BG, width=180)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(left, text="รูปภาพ", bg=SIDEBAR_BG, fg=TEXT,
                 font=("Segoe UI",10,"bold"), pady=8).pack(fill="x", padx=8)
        tk.Frame(left, bg=BORDER, height=1).pack(fill="x")

        self.file_listbox = tk.Listbox(left, bg=SIDEBAR_BG, fg=TEXT,
                                       selectbackground=ACCENT, selectforeground="white",
                                       font=("Segoe UI",9), borderwidth=0, highlightthickness=0,
                                       activestyle="none")
        sb = tk.Scrollbar(left, orient="vertical", command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.file_listbox.pack(fill="both", expand=True)
        self.file_listbox.bind("<<ListboxSelect>>", self._on_file_select)

        tk.Label(left, text="← → หรือ Q/E เปลี่ยนภาพ\nScroll = zoom  |  Drag กลาง = pan",
                 bg=SIDEBAR_BG, fg=TEXT_DIM, font=("Segoe UI",8), justify="center", pady=6).pack()

        # Canvas area
        self.canvas_frame = tk.Frame(main, bg=BG)
        self.canvas_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#1a1a2e",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        # Right sidebar — box list + class manager
        right = tk.Frame(main, bg=PANEL_BG, width=200)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="Bounding Boxes", bg=PANEL_BG, fg=TEXT,
                 font=("Segoe UI",10,"bold"), pady=8).pack(fill="x", padx=8)
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x")

        self.box_listbox = tk.Listbox(right, bg=PANEL_BG, fg=TEXT,
                                      selectbackground=ACCENT, selectforeground="white",
                                      font=("Segoe UI",9), borderwidth=0, highlightthickness=0,
                                      activestyle="none", height=14)
        self.box_listbox.pack(fill="x", padx=6, pady=4)
        self.box_listbox.bind("<<ListboxSelect>>", self._on_box_select)

        self._btn(right, "✕  ลบ box ที่เลือก", self._delete_selected_box).pack(fill="x", padx=6, pady=2)

        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=6)
        tk.Label(right, text="จัดการ Class", bg=PANEL_BG, fg=TEXT_DIM,
                 font=("Segoe UI",9)).pack(padx=8, anchor="w")

        self.new_class_var = tk.StringVar()
        new_cls_frame = tk.Frame(right, bg=PANEL_BG)
        new_cls_frame.pack(fill="x", padx=6, pady=4)
        tk.Entry(new_cls_frame, textvariable=self.new_class_var,
                 bg=BTN_BG, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Segoe UI",9)).pack(side="left", fill="x", expand=True)
        self._btn(new_cls_frame, "+", self.add_class, width=3).pack(side="right", padx=(4,0))
        self.new_class_var.trace_add("write", lambda *a: None)
        self.root.bind("<Return>", lambda e: self.add_class() if self.root.focus_get() and
                       str(self.root.focus_get()).endswith("entry") else None)

        self.del_class_combo = ttk.Combobox(right, values=self.classes,
                                            font=("Segoe UI",9), state="readonly", width=18)
        self.del_class_combo.pack(padx=6, pady=2, fill="x")
        self._btn(right, "🗑  ลบ class นี้", self.del_class, danger=True).pack(fill="x", padx=6, pady=2)

        # ── Status bar ──
        status = tk.Frame(self.root, bg=PANEL_BG, pady=4)
        status.pack(fill="x", side="bottom")
        self.status_img = tk.Label(status, text="ยังไม่มีรูปภาพ", bg=PANEL_BG, fg=TEXT_DIM,
                                   font=("Segoe UI",9))
        self.status_img.pack(side="left", padx=10)
        self.status_box = tk.Label(status, text="", bg=PANEL_BG, fg=TEXT_DIM,
                                   font=("Segoe UI",9))
        self.status_box.pack(side="left", padx=10)
        tk.Label(status, text="คลิกลาก = วาด box  |  Del = ลบที่เลือก  |  กลาง+ลาก = pan  |  Scroll = zoom",
                 bg=PANEL_BG, fg=TEXT_DIM, font=("Segoe UI",8)).pack(side="right", padx=10)

        # ── Bindings ──
        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_down)
        self.canvas.bind("<B1-Motion>",        self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>",  self._on_mouse_up)
        self.canvas.bind("<ButtonPress-2>",    self._pan_start_cb)
        self.canvas.bind("<B2-Motion>",        self._pan_move_cb)
        self.canvas.bind("<ButtonPress-3>",    self._pan_start_cb)
        self.canvas.bind("<B3-Motion>",        self._pan_move_cb)
        self.canvas.bind("<MouseWheel>",       self._on_scroll)
        self.canvas.bind("<Button-4>",         self._on_scroll)
        self.canvas.bind("<Button-5>",         self._on_scroll)

        self.root.bind("<Right>",  lambda e: self.next_image())
        self.root.bind("<Left>",   lambda e: self.prev_image())
        self.root.bind("e",        lambda e: self.next_image())
        self.root.bind("q",        lambda e: self.prev_image())
        self.root.bind("<Delete>", lambda e: self._delete_selected_box())
        self.root.bind("<BackSpace>", lambda e: self._delete_selected_box())
        self.root.bind("<Control-z>", lambda e: self.undo_last())

    # ─────────────── WIDGET HELPERS ───────────────
    def _btn(self, parent, text, cmd, width=None, accent=False, danger=False):
        bg = "#1565c0" if accent else ("#b71c1c" if danger else BTN_BG)
        fg = "white" if (accent or danger) else TEXT
        kw = dict(text=text, command=cmd, bg=bg, fg=fg,
                  activebackground=BTN_HOV, activeforeground=TEXT,
                  font=("Segoe UI",9), relief="flat", bd=0, padx=8, pady=4, cursor="hand2")
        if width: kw["width"] = width
        b = tk.Button(parent, **kw)
        b.bind("<Enter>", lambda e, b=b, bg=bg: b.config(bg=self._lighten(bg)))
        b.bind("<Leave>", lambda e, b=b, bg=bg: b.config(bg=bg))
        return b

    def _lighten(self, hex_color):
        try:
            r,g,b = int(hex_color[1:3],16),int(hex_color[3:5],16),int(hex_color[5:7],16)
            r,g,b = min(r+30,255),min(g+30,255),min(b+30,255)
            return f"#{r:02x}{g:02x}{b:02x}"
        except: return hex_color

    def _sep(self, parent):
        tk.Frame(parent, bg=BORDER, width=1, height=24).pack(side="left", padx=6, pady=4)

    # ─────────────── CLASS MANAGEMENT ───────────────
    def _refresh_classes(self):
        self.class_combo["values"] = self.classes
        self.del_class_combo["values"] = self.classes
        if self.cur_class.get() not in self.classes and self.classes:
            self.cur_class.set(self.classes[0])
        self._update_color_dot()

    def _update_color_dot(self):
        c = self.cur_class.get()
        idx = self.classes.index(c) if c in self.classes else 0
        self.color_dot.config(bg=COLORS[idx % len(COLORS)])

    def add_class(self):
        name = self.new_class_var.get().strip()
        if name and name not in self.classes:
            self.classes.append(name)
            self._refresh_classes()
            self.cur_class.set(name)
            self._update_color_dot()
        self.new_class_var.set("")

    def del_class(self):
        c = self.del_class_combo.get()
        if not c: return
        if len(self.classes) <= 1:
            messagebox.showwarning("", "ต้องมีอย่างน้อย 1 class")
            return
        self.classes.remove(c)
        for k in self.labels:
            self.labels[k] = [b for b in self.labels[k] if b[4] != c]
        self._refresh_classes()
        self._redraw()
        self._refresh_box_list()

    def _get_color(self, class_name):
        idx = self.classes.index(class_name) if class_name in self.classes else 0
        return COLORS[idx % len(COLORS)]

    # ─────────────── FILE MANAGEMENT ───────────────
    def open_files(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Images","*.jpg *.jpeg *.png *.webp *.bmp")])
        if not paths: return
        existing = {im["name"] for im in self.images}
        for p in paths:
            name = os.path.basename(p)
            if name not in existing:
                self.images.append({"name": name, "path": p})
        self.index = max(0, len(self.images) - len(paths))
        self._refresh_file_list()
        self._load_image()

    def _refresh_file_list(self):
        self.file_listbox.delete(0, "end")
        for i, im in enumerate(self.images):
            has = bool(self.labels.get(im["name"]))
            prefix = "✔ " if has else "  "
            self.file_listbox.insert("end", f"{prefix}{im['name']}")
            if has:
                self.file_listbox.itemconfig(i, fg="#4caf50")
        if self.images:
            self.file_listbox.selection_clear(0, "end")
            self.file_listbox.selection_set(self.index)
            self.file_listbox.see(self.index)

    def _on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if sel:
            self.index = sel[0]
            self._load_image()

    # ─────────────── IMAGE LOADING ───────────────
    def _load_image(self):
        if not self.images: return
        path = self.images[self.index]["path"]
        img  = Image.open(path).convert("RGB")
        self.orig_w, self.orig_h = img.size
        self._pil_img = img
        self._zoom_fit()
        self._refresh_file_list()
        self._refresh_box_list()
        name = self.images[self.index]["name"]
        self.status_img.config(
            text=f"{name}  ({self.orig_w}×{self.orig_h})  —  {self.index+1}/{len(self.images)}")

    def _compute_display(self):
        cw = max(self.canvas.winfo_width(),  200)
        ch = max(self.canvas.winfo_height(), 200)
        # base scale to fit
        base = min(cw / self.orig_w, ch / self.orig_h)
        self.scale = base * self.zoom
        disp_w = int(self.orig_w * self.scale)
        disp_h = int(self.orig_h * self.scale)
        # center + pan
        self.offset_x = (cw - disp_w) // 2 + int(self.pan_x)
        self.offset_y = (ch - disp_h) // 2 + int(self.pan_y)
        return disp_w, disp_h

    def _redraw(self):
        if not hasattr(self, "_pil_img") or not self._pil_img: return
        self.canvas.delete("all")
        disp_w, disp_h = self._compute_display()
        resized = self._pil_img.resize((max(1,disp_w), max(1,disp_h)), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self._tk_img)

        key = self.images[self.index]["name"]
        for i, box in enumerate(self.labels.get(key, [])):
            x1,y1,x2,y2,label = box
            dx1 = x1*self.scale + self.offset_x
            dy1 = y1*self.scale + self.offset_y
            dx2 = x2*self.scale + self.offset_x
            dy2 = y2*self.scale + self.offset_y
            col  = self._get_color(label)
            lw   = 3 if i == self.selected_box else 2
            dash = (6,3) if i == self.selected_box else ()
            self.canvas.create_rectangle(dx1,dy1,dx2,dy2, outline=col, width=lw, dash=dash)
            # label tag
            self.canvas.create_rectangle(dx1, dy1-16, dx1+len(label)*7+8, dy1,
                                         fill=col, outline="")
            self.canvas.create_text(dx1+4, dy1-8, text=label,
                                    fill="white", font=("Segoe UI",8,"bold"), anchor="w")

        self.zoom_label.config(text=f"{int(self.zoom*100)}%")

    # ─────────────── ZOOM ───────────────
    def _zoom_fit(self):
        self.zoom  = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._redraw()

    def _zoom_reset(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        base = min(cw/self.orig_w, ch/self.orig_h)
        self.zoom  = 1.0 / base          # zoom so scale == 1.0 (1:1 pixel)
        self.pan_x = 0
        self.pan_y = 0
        self._redraw()

    def _zoom_in(self):  self._set_zoom(self.zoom * 1.25)
    def _zoom_out(self): self._set_zoom(self.zoom / 1.25)

    def _set_zoom(self, z, cx=None, cy=None):
        old   = self.zoom
        self.zoom = max(self.zoom_min, min(self.zoom_max, z))
        ratio = self.zoom / old
        if cx is not None and cy is not None:
            self.pan_x = cx + (self.pan_x - cx) * ratio
            self.pan_y = cy + (self.pan_y - cy) * ratio
        self._redraw()

    def _on_scroll(self, event):
        if not self.images: return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        base = min(cw/self.orig_w, ch/self.orig_h)
        # mouse position relative to center
        cx = event.x - cw//2
        cy = event.y - ch//2
        if event.num == 4 or event.delta > 0:
            self._set_zoom(self.zoom * 1.1, cx, cy)
        else:
            self._set_zoom(self.zoom / 1.1, cx, cy)

    # ─────────────── PAN ───────────────
    def _pan_start_cb(self, event):
        self._pan_start = (event.x, event.y, self.pan_x, self.pan_y)
        self.canvas.config(cursor="fleur")

    def _pan_move_cb(self, event):
        if not self._pan_start: return
        sx, sy, px, py = self._pan_start
        self.pan_x = px + (event.x - sx)
        self.pan_y = py + (event.y - sy)
        self._redraw()

    # ─────────────── DRAWING ───────────────
    def _canvas_to_orig(self, cx, cy):
        ox = (cx - self.offset_x) / self.scale
        oy = (cy - self.offset_y) / self.scale
        ox = max(0, min(self.orig_w, ox))
        oy = max(0, min(self.orig_h, oy))
        return ox, oy

    def _on_mouse_down(self, event):
        if not self.images: return
        self.drawing    = True
        self.start_cx   = event.x
        self.start_cy   = event.y
        self.selected_box = -1
        self.temp_rect  = None

    def _on_mouse_drag(self, event):
        if not self.drawing: return
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
        col = self._get_color(self.cur_class.get())
        self.temp_rect = self.canvas.create_rectangle(
            self.start_cx, self.start_cy, event.x, event.y,
            outline=col, width=2, dash=(5,3))

    def _on_mouse_up(self, event):
        if not self.drawing or not self.images: return
        self.drawing = False
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None

        x1o, y1o = self._canvas_to_orig(self.start_cx, self.start_cy)
        x2o, y2o = self._canvas_to_orig(event.x,       event.y)
        if abs(x2o-x1o) < 3 or abs(y2o-y1o) < 3:
            return

        key = self.images[self.index]["name"]
        self.labels.setdefault(key, []).append(
            [min(x1o,x2o), min(y1o,y2o), max(x1o,x2o), max(y1o,y2o),
             self.cur_class.get()])
        self.selected_box = len(self.labels[key]) - 1
        self._redraw()
        self._refresh_box_list()
        self._refresh_file_list()

    # ─────────────── BOX LIST ───────────────
    def _refresh_box_list(self):
        self.box_listbox.delete(0, "end")
        if not self.images: return
        key   = self.images[self.index]["name"]
        boxes = self.labels.get(key, [])
        for i, b in enumerate(boxes):
            self.box_listbox.insert("end", f"  {b[4]}")
            self.box_listbox.itemconfig(i, fg=self._get_color(b[4]))
        if 0 <= self.selected_box < len(boxes):
            self.box_listbox.selection_set(self.selected_box)
        self.status_box.config(text=f"{len(boxes)} boxes")

    def _on_box_select(self, event):
        sel = self.box_listbox.curselection()
        if sel:
            self.selected_box = sel[0]
            self._redraw()

    def _delete_selected_box(self):
        if not self.images: return
        key = self.images[self.index]["name"]
        if 0 <= self.selected_box < len(self.labels.get(key, [])):
            self.labels[key].pop(self.selected_box)
            self.selected_box = max(0, self.selected_box - 1) if self.labels[key] else -1
            self._redraw()
            self._refresh_box_list()
            self._refresh_file_list()

    # ─────────────── NAVIGATION ───────────────
    def next_image(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self._load_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self._load_image()

    def undo_last(self):
        if not self.images: return
        key = self.images[self.index]["name"]
        if self.labels.get(key):
            self.labels[key].pop()
            self.selected_box = -1
            self._redraw()
            self._refresh_box_list()
            self._refresh_file_list()

    def clear_current(self):
        if not self.images: return
        if messagebox.askyesno("ล้าง", "ล้าง box ทั้งหมดในภาพนี้?"):
            key = self.images[self.index]["name"]
            self.labels[key] = []
            self.selected_box = -1
            self._redraw()
            self._refresh_box_list()
            self._refresh_file_list()

    # ─────────────── EXPORT ───────────────
    def export_yolo(self):
        labeled = {k: v for k, v in self.labels.items() if v}
        if not labeled:
            messagebox.showwarning("", "ยังไม่ได้วาด box เลย")
            return

        out = filedialog.askdirectory(title="เลือกโฟลเดอร์ output")
        if not out: return

        img_dir = os.path.join(out, "images")
        lbl_dir = os.path.join(out, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # write classes.txt
        with open(os.path.join(out, "classes.txt"), "w") as f:
            f.write("\n".join(self.classes))

        total_boxes = 0
        for im in self.images:
            key   = im["name"]
            boxes = self.labels.get(key, [])
            if not boxes: continue
            shutil.copy2(im["path"], img_dir)
            img = Image.open(im["path"])
            iw, ih = img.size
            name = os.path.splitext(key)[0]
            with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
                for x1,y1,x2,y2,label in boxes:
                    xc = ((x1+x2)/2) / iw
                    yc = ((y1+y2)/2) / ih
                    bw = abs(x2-x1) / iw
                    bh = abs(y2-y1) / ih
                    cid = self.classes.index(label) if label in self.classes else 0
                    f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    total_boxes += 1

        messagebox.showinfo("Export สำเร็จ",
            f"✔ {len(labeled)} ภาพ  |  {total_boxes} boxes\n"
            f"📁 บันทึกที่: {out}\n\n"
            f"• images/   — ไฟล์รูปต้นฉบับ\n"
            f"• labels/   — YOLO .txt\n"
            f"• classes.txt — รายชื่อ class")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()