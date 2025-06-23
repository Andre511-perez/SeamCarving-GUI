#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seam-Carving GUI
2025-06-22  phyang
"""

import os, threading, queue
from dataclasses import dataclass, replace
from typing import List, Callable, Optional, Dict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ---------- 可选深度依赖 ----------
try:
    import torch
    import torch.nn.functional as F

    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from skimage.filters.rank import entropy as skimage_entropy
    from skimage.morphology import disk

    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False


# ============ 能量函数 ============
def energy_sobel(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def energy_laplacian(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return np.abs(lap)


def energy_entropy(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if SKIMAGE_OK:
        radius = ksize // 2
        return skimage_entropy(gray, disk(radius)).astype(np.float64)
    else:
        print("警告: 未安装 scikit-image，熵能量函数计算将非常缓慢。")
        hist = np.zeros((gray.shape[0], gray.shape[1], 256), np.uint16)
        for val in range(256):
            mask = (gray == val).astype(np.uint8)
            hist[..., val] = cv2.blur(mask, (ksize, ksize))
        prob = hist / (ksize * ksize)
        prob[prob == 0] = 1
        entropy = -(prob * np.log2(prob)).sum(axis=2)
        return entropy.astype(np.float64)


def energy_saliency(img: np.ndarray) -> np.ndarray:
    sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal_map = sal.computeSaliency(img)
    if not ok:
        return np.ones(img.shape[:2], np.float64)
    return cv2.blur((sal_map * 255).astype(np.float64), (3, 3))


# ---------- 深度能量 (U2-Net) ----------
class U2NetWrapper:
    def __init__(self, model_path: str):
        if not TORCH_OK or not os.path.exists(model_path):
            raise RuntimeError(f"PyTorch 或模型文件缺失: {model_path}")

        # 动态导入U2NET和U2NETP类
        from u2net import U2NET, U2NETP

        # 根据模型路径名自动选择加载U2NET还是U2NETP
        if 'u2netp' in os.path.basename(model_path).lower():
            print(f"正在加载轻量版模型: U2NETP from {model_path}")
            net = U2NETP(3, 1)
        else:
            print(f"正在加载完整版模型: U2NET from {model_path}")
            net = U2NET(3, 1)

        # 加载模型权重
        try:
            # Pytorch 1.6+ 推荐的加载方式
            net.load_state_dict(torch.load(model_path, map_location="cpu"))
        except Exception:
            # 兼容旧版或不同保存方式的模型
            obj = torch.load(model_path, map_location="cpu")
            if "state_dict" in obj:
                net.load_state_dict(obj["state_dict"])
            else:
                net.load_state_dict(obj)

        self.net = net
        self.net.eval()

    @torch.no_grad()
    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        h_orig, w_orig = img_bgr.shape[:2]

        # 预处理
        tensor = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0)

        # 将输入图片缩放到模型期望的320x320，极大提升速度
        tensor_resized = F.interpolate(tensor, (320, 320), mode='bilinear', align_corners=False)

        # 推理
        d0, d1, d2, d3, d4, d5, d6 = self.net(tensor_resized)

        # 后处理
        sal = torch.sigmoid(d0).squeeze().cpu().numpy()

        # 将输出的能量图缩放回原始尺寸
        sal = cv2.resize(sal, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(sal, (5, 5), 0) * 255.0


try:
    # 明确指定加载 u2netp.pth 模型
    deep_model = U2NetWrapper(model_path="models/u2netp.pth")


    def energy_deep(img: np.ndarray) -> np.ndarray:
        return deep_model(img).astype(np.float64)


    DEEP_OK = True
except Exception as e:
    import traceback

    traceback.print_exc()
    DEEP_OK = False


    def energy_deep(img: np.ndarray) -> np.ndarray:
        # Fallback
        return energy_sobel(img)

ENERGY_FUNC_MAP = {
    "Sobel": energy_sobel,
    "Laplacian": energy_laplacian,
    "Saliency": energy_saliency,
    "Entropy": energy_entropy,
}
if DEEP_OK:
    ENERGY_FUNC_MAP["Deep"] = energy_deep


# ------------ Seam 算法工具 ------------
def cumulative_map_vertical(energy: np.ndarray):
    h, w = energy.shape
    M = energy.copy()
    back = np.zeros_like(M, np.int32)
    for i in range(1, h):
        left = np.pad(M[i - 1, :-1], (1, 0), constant_values=np.inf)
        mid = M[i - 1]
        right = np.pad(M[i - 1, 1:], (0, 1), constant_values=np.inf)
        choices = np.vstack((left, mid, right))
        idx = np.argmin(choices, axis=0)
        M[i] += choices[idx, np.arange(w)]
        back[i] = idx - 1
    return M, back


def find_seam_vertical(energy: np.ndarray) -> np.ndarray:
    M, back = cumulative_map_vertical(energy)
    h, _ = energy.shape
    seam = np.zeros(h, np.int32)
    seam[-1] = np.argmin(M[-1])
    for i in range(h - 2, -1, -1):
        seam[i] = seam[i + 1] + back[i + 1, seam[i + 1]]
    return seam


def remove_seam_vertical(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    mask = np.stack([mask] * c, axis=-1)
    return img[mask].reshape((h, w - 1, c))


def insert_seam_vertical(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    out = np.zeros((h, w + 1, c), dtype=img.dtype)
    for i in range(h):
        j = seam[i]
        if j == 0:
            p = img[i, j, :]
        else:
            p = (img[i, j - 1, :].astype(np.float32) + img[i, j, :].astype(np.float32)) / 2
        out[i, :j, :] = img[i, :j, :]
        out[i, j, :] = p.astype(img.dtype)
        out[i, j + 1:, :] = img[i, j:, :]
    return out


def find_seam_horizontal(energy: np.ndarray) -> np.ndarray:
    return find_seam_vertical(energy.T)


def remove_seam_horizontal(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    out = np.zeros((h - 1, w, c), dtype=img.dtype)
    for j in range(w):
        i = seam[j]
        out[:i, j, :] = img[:i, j, :]
        out[i:, j, :] = img[i + 1:, j, :]
    return out


def insert_seam_horizontal(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    out = np.zeros((h + 1, w, c), dtype=img.dtype)
    for j in range(w):
        i = seam[j]
        if i == 0:
            p = img[i, j, :]
        else:
            p = (img[i - 1, j, :].astype(np.float32) + img[i, j, :].astype(np.float32)) / 2
        out[:i, j, :] = img[:i, j, :]
        out[i, j, :] = p.astype(img.dtype)
        out[i + 1:, j, :] = img[i:, j, :]
    return out


# ------------ SeamCarver 主类 ------------
@dataclass
class ResizeOptions:
    tgt_w: int
    tgt_h: int
    batch: bool = True
    animate: bool = False
    energy_name: str = "Sobel"


class SeamCarver:
    def __init__(self, options: ResizeOptions):
        self.opt = options
        self.energy_fn = ENERGY_FUNC_MAP.get(options.energy_name, energy_sobel)
        self.progress_callback: Optional[Callable[[int, Optional[np.ndarray]], None]] = None

    def resize(self, img: np.ndarray) -> np.ndarray:
        h0, w0 = img.shape[:2]
        dw = self.opt.tgt_w - w0
        dh = self.opt.tgt_h - h0
        out = img.copy()
        if dw < 0:
            out = self._shrink(out, -dw, vertical=True)
        elif dw > 0:
            out = self._expand(out, dw, vertical=True)
        if dh < 0:
            out = self._shrink(out, -dh, vertical=False)
        elif dh > 0:
            out = self._expand(out, dh, vertical=False)
        return out

    def _shrink(self, img, pixels, *, vertical=True):
        for i in range(pixels):
            energy = self.energy_fn(img)
            if vertical:
                seam = find_seam_vertical(energy)
            else:
                seam = find_seam_horizontal(energy)
            frame = self._draw_seam(img, seam, vertical) if self.opt.animate else None
            if vertical:
                img = remove_seam_vertical(img, seam)
            else:
                img = remove_seam_horizontal(img, seam)
            if self.progress_callback:
                self.progress_callback(1, frame)
        return img

    def _expand(self, img, pixels, *, vertical=True):
        img_for_seam_finding = img.copy()
        seams_to_insert = []
        if self.opt.batch:
            energy = self.energy_fn(img_for_seam_finding)
            tmp_energy = energy.copy() if vertical else energy.T.copy()
            for _ in range(pixels):
                seam = find_seam_vertical(tmp_energy)
                seams_to_insert.append(seam)
                # Mark seam as infinite energy to find a different one next time
                if vertical:
                    for row, col in enumerate(seam):
                        if 0 <= row < tmp_energy.shape[0] and 0 <= col < tmp_energy.shape[1]:
                            tmp_energy[row, col] = np.inf
                else:  # horizontal
                    for col, row in enumerate(seam):
                        if 0 <= row < tmp_energy.shape[0] and 0 <= col < tmp_energy.shape[1]:
                            tmp_energy[row, col] = np.inf

        if vertical and self.opt.batch:
            seams_to_insert.sort(key=lambda s: s[-1])

        for i in range(pixels):
            if not self.opt.batch:
                energy = self.energy_fn(img)
                if vertical:
                    seam = find_seam_vertical(energy)
                else:
                    seam = find_seam_horizontal(energy)
            else:
                seam = seams_to_insert[i]
            frame = self._draw_seam(img, seam, vertical) if self.opt.animate else None
            if vertical:
                img = insert_seam_vertical(img, seam)
            else:
                img = insert_seam_horizontal(img, seam)
            if self.progress_callback:
                self.progress_callback(1, frame)
        return img

    def _draw_seam(self, img, seam, vertical=True):
        vis = img.copy()
        color = (0, 0, 255)  # Red
        if vertical:
            vis[np.arange(vis.shape[0]), seam] = color
        else:
            vis[seam, np.arange(vis.shape[1])] = color
        return vis


# ------------ 动画播放窗口 ------------
class AnimatorWindow(tk.Toplevel):
    def __init__(self, parent, fps: int = 30):
        super().__init__(parent)
        self.title("Seam Animation")
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.delay = int(1000 / fps)
        self.img_cache: List[ImageTk.PhotoImage] = []
        self.frame_queue = queue.Queue()
        self.idx = 0
        self.is_playing = True
        self.canvas = tk.Canvas(self, width=300, height=300, background="black")
        self.canvas.pack()
        tk.Button(self, text="关闭", command=self.destroy).pack(pady=4)
        self.after(100, self._play)

    def add_frame(self, frame: np.ndarray):
        self.frame_queue.put(frame)

    def _play(self):
        if not self.is_playing: return
        try:
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                scale = min(300 / w, 300 / h, 1)
                pil = Image.fromarray(rgb).resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                self.img_cache.append(ImageTk.PhotoImage(pil))
        except queue.Empty:
            pass
        if self.idx < len(self.img_cache):
            current_img = self.img_cache[self.idx]
            if not hasattr(self, 'img_id'):
                w, h = current_img.width(), current_img.height()
                self.canvas.config(width=w, height=h)
                self.img_id = self.canvas.create_image(0, 0, anchor="nw", image=current_img)
            else:
                self.canvas.itemconfig(self.img_id, image=current_img)
            self.idx += 1
        self.after(self.delay, self._play)

    def destroy(self):
        self.is_playing = False
        super().destroy()


# ------------ GUI ------------
class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Seam Carving")
        self.img_orig: Optional[np.ndarray] = None
        self.img_processed: Optional[np.ndarray] = None
        self.animator: Optional[AnimatorWindow] = None
        self.q = queue.Queue()
        self._build_widgets()
        self._poll_queue()

    def _build_widgets(self):
        top = tk.Frame(self);
        top.pack(pady=4, padx=10, fill='x')
        tk.Button(top, text="选择图片", command=self.open_img).grid(row=0, column=0, padx=4)
        self.fname_var = tk.StringVar(value="未选择")
        tk.Label(top, textvariable=self.fname_var, width=40, anchor="w").grid(row=0, column=1)

        form = ttk.LabelFrame(self, text="选项");
        form.pack(pady=4, padx=10, fill='x')
        form.columnconfigure(1, weight=1);
        form.columnconfigure(3, weight=1)

        tk.Label(form, text="横向比例 (0<×≤2)").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        tk.Label(form, text="纵向比例 (0<×≤2)").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.sx_entry = tk.Entry(form, width=8);
        self.sy_entry = tk.Entry(form, width=8)
        self.sx_entry.insert(0, "1.0");
        self.sy_entry.insert(0, "1.0")
        self.sx_entry.grid(row=0, column=1, sticky='w');
        self.sy_entry.grid(row=1, column=1, sticky='w')

        tk.Label(form, text="能量函数").grid(row=0, column=2, sticky='e', padx=5, pady=2)
        energy_options = list(ENERGY_FUNC_MAP.keys()) + ["ALL"]
        self.energy_combo = ttk.Combobox(form, values=energy_options, width=10, state="readonly")
        self.energy_combo.set("Sobel");
        self.energy_combo.grid(row=0, column=3, sticky='w')
        self.energy_combo.bind("<<ComboboxSelected>>", self._on_energy_select)

        self.chk_ani_var = tk.BooleanVar(value=False)
        self.chk_ani_btn = tk.Checkbutton(form, text="动画演示", variable=self.chk_ani_var)
        self.chk_ani_btn.grid(row=1, column=2, columnspan=2, sticky='w', padx=5)

        self.chk_batch_var = tk.BooleanVar(value=True)
        tk.Checkbutton(form, text="批量seam(加速放大)", variable=self.chk_batch_var).grid(row=2, column=0, columnspan=2,
                                                                                          sticky='w', padx=5)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=6, padx=10, fill='x')
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self.btn_start = tk.Button(btn_frame, text="开始处理", command=self.start_resize, height=2);
        self.btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 2))

        self.btn_export = tk.Button(btn_frame, text="导出图片", command=self.export_img, height=2, state=tk.DISABLED)
        self.btn_export.grid(row=0, column=1, sticky="ew", padx=(2, 0))
        # -- 结束修改 --

        self.prog = ttk.Progressbar(self, length=360, mode="determinate");
        self.prog.pack(pady=4, padx=10, fill='x')
        self.lbl_img = tk.Label(self, text="请先选择一张图片", compound="top", pady=10);
        self.lbl_img.pack(pady=4, padx=10, expand=True, fill='both')

    def _on_energy_select(self, event=None):
        if self.energy_combo.get() == "ALL":
            self.chk_ani_var.set(False)
            self.chk_ani_btn.config(state=tk.DISABLED)
        else:
            self.chk_ani_btn.config(state=tk.NORMAL)

    def open_img(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if not path: return
        try:
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            if img is None: raise IOError("无法解码图像")
            self.img_orig = img
            self.img_processed = None
            self.btn_export.config(state=tk.DISABLED)
            self._show(img)
            self.fname_var.set(os.path.basename(path))
        except Exception as e:
            messagebox.showerror("无法打开", f"打开文件失败: {path}\n错误: {e}")
            return

    def _show(self, img: np.ndarray, text=""):
        h, w = img.shape[:2];
        scale = min(800 / w, 600 / h, 1)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil = pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil)
        self.lbl_img.configure(image=tk_img, text=text);
        self.lbl_img.image = tk_img

    def _show_pair(self, orig: np.ndarray, proc: np.ndarray):
        h_max = max(orig.shape[0], proc.shape[0])
        orig_padded = cv2.copyMakeBorder(orig, 0, h_max - orig.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                         value=[240, 240, 240])
        proc_padded = cv2.copyMakeBorder(proc, 0, h_max - proc.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                         value=[240, 240, 240])
        gap = np.ones((h_max, 10, 3), np.uint8) * 240
        combined = np.hstack([orig_padded, gap, proc_padded])

        w1, h1 = proc.shape[:2]
        self._show(combined, text=f"原图 vs. 结果 ({w1}x{h1})")

    def _show_all_results(self, orig: np.ndarray, results: Dict[str, np.ndarray]):
        TARGET_H = 200
        GAP = 10
        LABEL_H = 25
        FONT_COLOR = "black"
        BG_COLOR = (240, 240, 240)

        # 保证固定的显示顺序
        energy_names = sorted(list(ENERGY_FUNC_MAP.keys()))
        images_to_render = [("Original", orig)]
        for name in energy_names:
            if name in results:
                images_to_render.append((name, results[name]))

        pil_images = []
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()

        for name, img_data in images_to_render:
            h, w = img_data.shape[:2]
            scale = TARGET_H / h
            new_w, new_h = int(w * scale), TARGET_H
            pil_img = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)).resize((new_w, new_h),
                                                                                        Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (new_w, new_h + LABEL_H), BG_COLOR)
            canvas.paste(pil_img, (0, 0))
            draw = ImageDraw.Draw(canvas)
            text_bbox = draw.textbbox((0, 0), name, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            draw.text(((new_w - text_w) / 2, new_h + 5), name, font=font, fill=FONT_COLOR)
            pil_images.append(canvas)

        if not pil_images:
            return

        cols = 3
        num_images = len(pil_images)
        rows = (num_images + cols - 1) // cols

        # 找到所有带标签图片的最大宽度和高度，以创建统一的单元格
        max_w = max(img.width for img in pil_images)
        max_h = max(img.height for img in pil_images)

        # 计算最终合并图像的总尺寸
        total_w = cols * max_w + (cols - 1) * GAP
        total_h = rows * max_h + (rows - 1) * GAP
        combined_img = Image.new("RGB", (total_w, total_h), BG_COLOR)

        # 将每个图片粘贴到网格中的对应位置
        for i, img in enumerate(pil_images):
            row_idx = i // cols
            col_idx = i % cols

            # 计算粘贴位置的左上角坐标
            x_offset = col_idx * (max_w + GAP)
            y_offset = row_idx * (max_h + GAP)

            # 将图片居中放置在单元格内
            x_paste = x_offset + (max_w - img.width) // 2
            y_paste = y_offset + (max_h - img.height) // 2

            combined_img.paste(img, (x_paste, y_paste))

        # 将组合图转换为OpenCV格式并保存，用于导出
        self.img_processed = cv2.cvtColor(np.array(combined_img), cv2.COLOR_RGB2BGR)

        tk_img = ImageTk.PhotoImage(combined_img)
        self.lbl_img.configure(image=tk_img, text="所有能量函数结果对比");
        self.lbl_img.image = tk_img

    def start_resize(self):
        if not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.prog['value'] = 0
        self.btn_export.config(state=tk.DISABLED)  # <-- [新增] 在处理时禁用导出
        if self.animator:
            try:
                self.animator.destroy()
            except tk.TclError:
                pass
            self.animator = None
        if self.img_orig is None: messagebox.showwarning("提示", "请先选择图片"); return
        try:
            sx, sy = float(self.sx_entry.get()), float(self.sy_entry.get())
        except ValueError:
            messagebox.showerror("输入错误", "比例必须是数字");
            return
        if not (0 < sx <= 2 and 0 < sy <= 2): messagebox.showerror("输入错误", "比例应在 (0, 2] 范围内"); return
        if abs(sx - 1) < 1e-6 and abs(sy - 1) < 1e-6: messagebox.showinfo("无需处理", "比例为1，图片尺寸不变"); return

        h0, w0 = self.img_orig.shape[:2]
        tgt_w = max(1, int(round(w0 * sx)))
        tgt_h = max(1, int(round(h0 * sy)))

        seams_to_process = abs(tgt_w - w0) + abs(tgt_h - h0)
        energy_name = self.energy_combo.get()

        if energy_name == "ALL":
            self.prog.config(maximum=seams_to_process * len(ENERGY_FUNC_MAP))
        else:
            self.prog.config(maximum=seams_to_process)
            if self.chk_ani_var.get():
                self.animator = AnimatorWindow(self)

        opt = ResizeOptions(
            tgt_w=tgt_w, tgt_h=tgt_h,
            batch=self.chk_batch_var.get(),
            animate=self.chk_ani_var.get() if energy_name != "ALL" else False,
            energy_name=energy_name
        )

        self.btn_start.config(state=tk.DISABLED, text="正在处理...")
        threading.Thread(target=self._worker, args=(self.img_orig.copy(), opt), daemon=True).start()

    def _worker(self, img_copy: np.ndarray, opt_base: ResizeOptions):
        try:
            def cb(step: int, frame: Optional[np.ndarray] = None):
                self.q.put(("progress", step))
                if frame is not None and self.animator:
                    self.q.put(("frame", frame))

            if opt_base.energy_name != "ALL":
                carver = SeamCarver(opt_base)
                carver.progress_callback = cb
                out = carver.resize(img_copy)
                self.q.put(("done", {"single_result": out}))
            else:
                all_results = {}
                for name in ENERGY_FUNC_MAP.keys():
                    print(f"Processing with energy function: {name}...")
                    opt = replace(opt_base, energy_name=name)
                    carver = SeamCarver(opt)
                    carver.progress_callback = cb
                    result = carver.resize(img_copy.copy())
                    all_results[name] = result
                self.q.put(("done", all_results))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.q.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                typ, data = self.q.get_nowait()
                if typ == "progress":
                    self.prog.step(data)
                elif typ == "frame":
                    if self.animator and self.animator.winfo_exists():
                        self.animator.add_frame(data)
                elif typ == "done":
                    # 处理完成后的逻辑
                    self.btn_start.config(state=tk.NORMAL, text="开始处理")
                    if "single_result" in data:
                        result_img = data["single_result"]
                        self.img_processed = result_img  # 保存结果用于导出
                        self._show_pair(self.img_orig, result_img)
                    else:
                        # 在 _show_all_results 内部会设置 self.img_processed
                        self._show_all_results(self.img_orig, data)

                    self.btn_export.config(state=tk.NORMAL)  # 启用导出按钮
                    messagebox.showinfo("完成", "处理完成！")
                elif typ == "error":
                    self.btn_start.config(state=tk.NORMAL, text="开始处理")
                    messagebox.showerror("处理失败", f"发生了一个错误：\n{data}")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # 导出图片方法 -->
    def export_img(self):
        if self.img_processed is None:
            messagebox.showwarning("无图片", "没有可导出的已处理图片。")
            return

        # 建议一个文件名
        original_basename = os.path.splitext(self.fname_var.get())[0]
        suggested_filename = f"{original_basename}_carved.png"

        path = filedialog.asksaveasfilename(
            initialfile=suggested_filename,
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("BMP Image", "*.bmp")]
        )

        if not path:
            return

        try:
            ext = os.path.splitext(path)[1]
            is_success, buffer = cv2.imencode(ext, self.img_processed)
            if not is_success:
                raise IOError("无法编码图像")
            with open(path, 'wb') as f:
                f.write(buffer)
            messagebox.showinfo("成功", f"图片已保存至:\n{path}")
        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存图片。\n错误: {e}")


if __name__ == "__main__":
    if not os.path.exists("models/u2netp.pth"):
        print("'Deep' 能量模式将不可用。")

    app = MainWindow()
    app.mainloop()
