"""Interactive wheat kernel segmentation with SAM2.

1. Draw one ROI per treatment (CT, LN, HN) on the pseudo-RGB image.
2. SAM2 generates kernel masks inside each ROI.
3. Reject merged or broken kernels in the review window.
4. Save each kernel as a 50 x 50 x bands cube (zero outside the mask).
"""

import os, json, glob, re, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.widgets import Button
import torch

mpl.rcParams["axes.unicode_minus"] = False

PAD_SIZE   = 50       # output size (pixels)
MIN_AREA   = 300      # mask area limits (pixels)
MAX_AREA   = 8000
MIN_ASPECT = 0.3      # minimum short/long side ratio

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(ROOT_DIR, "sam2_checkpoints", "sam2.1_hiera_large.pt")
MODEL_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"

CAL_DIR   = os.path.join(ROOT_DIR, "dataset", "hyperspectral_wheat_calibrated")
OUT_DIR   = os.path.join(ROOT_DIR, "dataset", "hyperspectral_wheat_3d")
META_PATH = os.path.join(CAL_DIR, "metadata.json")

CLASS_NAMES  = ["CT", "LN", "HN"]
CLASS_COLORS = ["#4C9BE8", "#56C785", "#E8714C"]

BG      = "#f8f8f8"
PANEL   = "#ffffff"
GRID_C  = "#dddddd"
TEXT    = "#222222"
SUBTEXT = "#666666"

# SAM2 automatic mask generation, applied to each ROI crop
AMG_KWARGS = dict(
    points_per_side        = 48,
    pred_iou_thresh        = 0.72,
    stability_score_thresh = 0.82,
    crop_n_layers          = 0,
    min_mask_region_area   = MIN_AREA,
)

os.makedirs(OUT_DIR, exist_ok=True)

with open(META_PATH) as f:
    meta = json.load(f)
WL = np.array(meta["wavelength_nm"])
N_BANDS = len(WL)

def nearest_band(target):
    return int(np.argmin(np.abs(WL - target)))

R_IDX = nearest_band(660)
G_IDX = nearest_band(550)
B_IDX = nearest_band(460)

print(f"Bands: {N_BANDS}")
print(f"RGB: R={WL[R_IDX]:.0f}nm  G={WL[G_IDX]:.0f}nm  B={WL[B_IDX]:.0f}nm")


print("\nLoading SAM2 (Large)...")
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model   = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
mask_gen     = SAM2AutomaticMaskGenerator(sam2_model, **AMG_KWARGS)
print(f"SAM2 ready  [{device}]\n")


def _sort_key(p):
    m = re.search(r"VNIR_(\d+)-(\d+)", p)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

def _is_done(path):
    sname = os.path.splitext(os.path.basename(path))[0]
    return len(glob.glob(os.path.join(OUT_DIR, "CT", f"{sname}_CT_kernel_*.npy"))) > 0

ALL_FILES = sorted(
    glob.glob(os.path.join(CAL_DIR, "VNIR_*.npy")),
    key=_sort_key
)


def hsi_to_rgb_uint8(cube: np.ndarray) -> np.ndarray:
    rgb = np.stack([cube[:,:,R_IDX], cube[:,:,G_IDX], cube[:,:,B_IDX]], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    if hi > lo:
        rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
    return (rgb * 255).astype(np.uint8)


def run_sam_on_roi(cube: np.ndarray, roi: tuple) -> list:
    """Kernel masks inside roi = (r0, c0, r1, c1) that pass the area and shape filters."""
    r0, c0, r1, c1 = (int(x) for x in roi)
    crop_cube = cube[r0:r1, c0:c1, :]
    crop_rgb  = hsi_to_rgb_uint8(crop_cube)

    with torch.inference_mode():
        raw_masks = mask_gen.generate(crop_rgb)

    results = []
    H_crop, W_crop = crop_cube.shape[:2]

    for m in raw_masks:
        area = m["area"]
        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        seg = m["segmentation"]
        rows = np.where(seg.any(axis=1))[0]
        cols = np.where(seg.any(axis=0))[0]
        if len(rows) == 0:
            continue

        kh = rows[-1] - rows[0] + 1
        kw = cols[-1] - cols[0] + 1
        short_s = min(kh, kw)
        long_s  = max(kh, kw)

        if long_s == 0 or (short_s / long_s) < MIN_ASPECT:
            continue
        if kh > PAD_SIZE or kw > PAD_SIZE:
            continue

        seg_full = np.zeros((cube.shape[0], cube.shape[1]), dtype=bool)
        seg_full[r0:r1, c0:c1] = seg

        results.append({
            "segmentation": seg_full,
            "seg_crop":     seg,
            "area":         area,
            "crop_origin":  (r0, c0),
            "bbox_in_full": (r0 + rows[0], c0 + cols[0],
                             r0 + rows[-1]+1,  c0 + cols[-1]+1),
        })
    return results


def make_padded(cube: np.ndarray, mask_info: dict) -> np.ndarray:
    """Centre the masked kernel crop in a zero-padded PAD_SIZE x PAD_SIZE cube."""
    r0f, c0f, r1f, c1f = mask_info["bbox_in_full"]
    seg_full = mask_info["segmentation"]

    kernel_hsi  = cube[r0f:r1f, c0f:c1f, :]
    mask_crop   = seg_full[r0f:r1f, c0f:c1f].astype(np.float32)

    kH, kW = kernel_hsi.shape[:2]
    out = np.zeros((PAD_SIZE, PAD_SIZE, N_BANDS), dtype=np.float32)
    y0  = (PAD_SIZE - kH) // 2
    x0  = (PAD_SIZE - kW) // 2
    out[y0:y0+kH, x0:x0+kW, :] = kernel_hsi * mask_crop[:,:,np.newaxis]
    return out


def thumb_rgb(padded: np.ndarray) -> np.ndarray:
    rgb = np.stack([padded[:,:,R_IDX], padded[:,:,G_IDX], padded[:,:,B_IDX]], axis=-1)
    lo, hi = np.percentile(rgb[rgb > 0], [2, 98]) if rgb.any() else (0, 1)
    if hi > lo:
        rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
    return (rgb * 255).astype(np.uint8)


class WheatSAMExtractor:

    def __init__(self):
        self.file_list  = ALL_FILES
        self.file_idx   = 0
        self.cube       = None
        self.rgb_full   = None
        self.fname      = ""

        self.rois       = [None, None, None]
        self.cur_cls    = 0
        self.drag_start = None
        self._drag_rect = None

        self.masks_per_cls  = [[], [], []]
        self.padded_per_cls = [[], [], []]
        self.selected       = [set(), set(), set()]

        self.fig = plt.figure(figsize=(14, 8), facecolor=BG)
        self.fig.canvas.manager.set_window_title("Wheat SAM Extractor")

        self.ax_img  = self.fig.add_axes([0.02, 0.10, 0.68, 0.86])
        self.ax_info = self.fig.add_axes([0.72, 0.10, 0.26, 0.86])
        for ax in (self.ax_img, self.ax_info):
            ax.set_facecolor(PANEL)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID_C)

        btn_kw = dict(color="#e0e0e0", hovercolor="#cccccc")
        ax_skip   = self.fig.add_axes([0.02, 0.01, 0.10, 0.045])
        ax_reset  = self.fig.add_axes([0.14, 0.01, 0.10, 0.045])
        ax_run    = self.fig.add_axes([0.26, 0.01, 0.14, 0.045])
        self.btn_skip  = Button(ax_skip,  "Skip File",   **btn_kw)
        self.btn_reset = Button(ax_reset, "Reset ROI",   **btn_kw)
        self.btn_run   = Button(ax_run,   "Run SAM ▶",   **btn_kw)
        for btn in (self.btn_skip, self.btn_reset, self.btn_run):
            btn.label.set_color(TEXT); btn.label.set_fontsize(9)

        self.btn_skip.on_clicked(self._skip_file)
        self.btn_reset.on_clicked(self._reset_roi)
        self.btn_run.on_clicked(self._run_sam)

        self.fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

        self._load_next()
        plt.show()

    def _load_next(self):
        while self.file_idx < len(self.file_list) and \
              _is_done(self.file_list[self.file_idx]):
            print(f"[Skip] {os.path.basename(self.file_list[self.file_idx])} (already done)")
            self.file_idx += 1

        if self.file_idx >= len(self.file_list):
            self._show_finished()
            return

        path = self.file_list[self.file_idx]
        self.cube     = np.load(path).astype(np.float32)
        self.rgb_full = hsi_to_rgb_uint8(self.cube)
        self.fname    = os.path.basename(path)
        self._reset_state()
        self._refresh_image()
        self._update_info()

    def _reset_state(self):
        self.rois            = [None, None, None]
        self.cur_cls         = 0
        self.drag_start      = None
        self._drag_rect      = None
        self.masks_per_cls   = [[], [], []]
        self.padded_per_cls  = [[], [], []]
        self.selected        = [set(), set(), set()]

    def _skip_file(self, _):
        self.file_idx += 1
        self._load_next()

    def _reset_roi(self, _):
        self._reset_state()
        self._refresh_image()
        self._update_info()

    def _refresh_image(self):
        self.ax_img.cla()
        self.ax_img.set_facecolor(PANEL)
        self.ax_img.set_xticks([]); self.ax_img.set_yticks([])
        for sp in self.ax_img.spines.values():
            sp.set_edgecolor(GRID_C)

        if self.rgb_full is None:
            self.fig.canvas.draw_idle()
            return

        self.ax_img.imshow(self.rgb_full, aspect="auto")

        n_done  = sum(1 for p in self.file_list if _is_done(p))
        self.ax_img.set_title(
            f"{self.fname}   [{self.file_idx+1}/{len(self.file_list)}]  done: {n_done}",
            color=TEXT, fontsize=9, pad=5)

        for i, roi in enumerate(self.rois):
            if roi is None:
                continue
            r0, c0, r1, c1 = roi
            rect = mpatches.Rectangle(
                (c0, r0), c1-c0, r1-r0,
                linewidth=2, edgecolor=CLASS_COLORS[i],
                facecolor=CLASS_COLORS[i], alpha=0.15)
            self.ax_img.add_patch(rect)
            self.ax_img.text(c0+3, r0+3, CLASS_NAMES[i],
                             color=CLASS_COLORS[i], fontsize=8,
                             fontweight="bold", va="top")

        for cls_i, masks in enumerate(self.masks_per_cls):
            for ki, m in enumerate(masks):
                if ki not in self.selected[cls_i]:
                    continue
                seg = m["segmentation"]
                from scipy.ndimage import binary_erosion
                edge = seg & ~binary_erosion(seg)
                ys, xs = np.where(edge)
                if len(ys):
                    self.ax_img.scatter(xs, ys, s=0.3,
                                        c=CLASS_COLORS[cls_i], alpha=0.6,
                                        linewidths=0)

        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax_img or self.cube is None:
            return
        if self.cur_cls >= 3 or event.button != 1:
            return
        self.drag_start = (event.ydata, event.xdata)

    def _on_motion(self, event):
        if self.drag_start is None or event.inaxes != self.ax_img:
            return
        if self._drag_rect is not None:
            try: self._drag_rect.remove()
            except: pass
        r0, c0 = self.drag_start
        r1, c1 = event.ydata, event.xdata
        rect = mpatches.Rectangle(
            (min(c0,c1), min(r0,r1)),
            abs(c1-c0), abs(r1-r0),
            linewidth=1.5, edgecolor=CLASS_COLORS[self.cur_cls],
            facecolor=CLASS_COLORS[self.cur_cls], alpha=0.20,
            linestyle="--")
        self.ax_img.add_patch(rect)
        self._drag_rect = rect
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self.drag_start is None or event.inaxes != self.ax_img:
            self.drag_start = None
            return
        if self._drag_rect is not None:
            try: self._drag_rect.remove()
            except: pass
            self._drag_rect = None

        r0s, c0s = self.drag_start
        r1e, c1e = event.ydata, event.xdata
        self.drag_start = None

        if abs(r1e - r0s) < 10 or abs(c1e - c0s) < 10:
            self.fig.canvas.draw_idle()
            return

        H, W, _ = self.cube.shape
        r0 = max(0, int(min(r0s, r1e)))
        r1 = min(H, int(max(r0s, r1e)))
        c0 = max(0, int(min(c0s, c1e)))
        c1 = min(W, int(max(c0s, c1e)))

        self.rois[self.cur_cls] = (r0, c0, r1, c1)
        self.cur_cls += 1
        self._refresh_image()
        self._update_info()

    def _run_sam(self, _):
        if self.cube is None:
            return
        if any(r is None for r in self.rois):
            print("[!] Draw an ROI for all three treatments first.")
            return

        self.ax_img.set_title("Running SAM2...",
                              color="tomato", fontsize=10, pad=5)
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

        total = 0
        for cls_i, roi in enumerate(self.rois):
            print(f"  [{CLASS_NAMES[cls_i]}] running SAM2...")
            masks = run_sam_on_roi(self.cube, roi)
            self.masks_per_cls[cls_i]  = masks
            self.padded_per_cls[cls_i] = [make_padded(self.cube, m) for m in masks]
            total += len(masks)
            print(f"    {len(masks)} kernels detected")

        print(f"  {total} kernels detected in total\n")
        self._refresh_image()
        self._update_info()

        self._open_review()

    def _open_review(self):
        """Review the kernels of each treatment in turn."""
        total = sum(len(m) for m in self.masks_per_cls)
        if total == 0:
            print("[!] No kernels detected.")
            return
        self._review_cls = 0
        self._open_review_for(0)

    _N_COL    = 10
    _N_ROW    = 8
    _PAGE_SZ  = _N_COL * _N_ROW

    def _open_review_for(self, cls_i):
        self._rev_page = 0
        self._draw_review_page(cls_i)

    def _draw_review_page(self, cls_i):
        n      = len(self.masks_per_cls[cls_i])
        color  = CLASS_COLORS[cls_i]
        name   = CLASS_NAMES[cls_i]
        page   = self._rev_page
        n_pages = max(1, math.ceil(n / self._PAGE_SZ))
        start  = page * self._PAGE_SZ
        end    = min(start + self._PAGE_SZ, n)

        if hasattr(self, "rev_fig") and plt.fignum_exists(self.rev_fig.number):
            plt.close(self.rev_fig)

        self.rev_fig = plt.figure(figsize=(20, 13), facecolor=BG)
        self.rev_fig.canvas.manager.set_window_title(
            f"Kernel review  [{cls_i+1}/3]  {name}  page {page+1}/{n_pages}")
        try:
            self.rev_fig.canvas.manager.window.state("zoomed")
        except Exception:
            pass

        sel = len(self.selected[cls_i])
        self.rev_title = self.rev_fig.suptitle(
            f"{self.fname}   |   {name} ({cls_i+1}/3)   |   "
            f"{n} kernels  selected: {sel}   |   "
            f"page {page+1}/{n_pages}  ({start+1}-{end})   |   click to toggle",
            fontsize=11, color=color, fontweight="bold", y=0.987)

        TOP  = 0.955
        BOT  = 0.062
        L0   = 0.004
        R0   = 0.996
        cell_w = (R0 - L0) / self._N_COL
        cell_h = (TOP - BOT) / self._N_ROW
        pad    = 0.0025

        self.rev_axes_cur  = []
        self._rev_ki_map   = []

        for slot, ki in enumerate(range(start, end)):
            row = slot // self._N_COL
            col = slot  % self._N_COL
            l = L0  + col * cell_w + pad
            b = TOP - (row + 1) * cell_h + pad
            w = cell_w - 2 * pad
            h = cell_h - 2 * pad

            ax = self.rev_fig.add_axes([l, b, w, h])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(1.8)

            ax.imshow(thumb_rgb(self.padded_per_cls[cls_i][ki]),
                      aspect="equal", interpolation="nearest")
            ax.text(0.02, 0.97, str(ki), transform=ax.transAxes,
                    fontsize=7, color="white", va="top", ha="left",
                    fontweight="bold",
                    bbox=dict(fc="black", alpha=0.4, pad=0.5, lw=0))

            self.rev_axes_cur.append(ax)
            self._rev_ki_map.append(ki)

        btn_kw = dict(color="#e0e0e0", hovercolor="#cccccc")
        self._rev_extra_btns = []   # keep references to the buttons

        if page > 0:
            ax_prev = self.rev_fig.add_axes([0.01, 0.005, 0.12, 0.048])
            btn_prev = Button(ax_prev, "◀ Previous", **btn_kw)
            btn_prev.label.set_fontsize(9)
            btn_prev.on_clicked(lambda _,
                                c=cls_i: self._rev_turn_page(c, -1))
            self._rev_extra_btns.append(btn_prev)

        if page < n_pages - 1:
            ax_next = self.rev_fig.add_axes([0.14, 0.005, 0.12, 0.048])
            btn_next = Button(ax_next, "Next ▶", **btn_kw)
            btn_next.label.set_fontsize(9)
            btn_next.on_clicked(lambda _,
                                c=cls_i: self._rev_turn_page(c, +1))
            self._rev_extra_btns.append(btn_next)

        label = "Confirm → next treatment ▶" if cls_i < 2 else "Confirm & save ✓"
        b_col = "#56C785" if cls_i < 2 else "#E8714C"
        b_hov = "#3da86a" if cls_i < 2 else "#c95a38"
        ax_ok = self.rev_fig.add_axes([0.38, 0.005, 0.24, 0.048])
        self.rev_btn = Button(ax_ok, label, color=b_col, hovercolor=b_hov)
        self.rev_btn.label.set_color("white")
        self.rev_btn.label.set_fontsize(10)
        self.rev_btn.label.set_fontweight("bold")
        self.rev_btn.on_clicked(self._on_review_confirm)

        self.rev_fig.canvas.mpl_connect("button_press_event", self._on_review_click)

        self._refresh_review_cur(cls_i)
        self.rev_fig.show()

    def _rev_turn_page(self, cls_i, delta):
        self._rev_page += delta
        self._draw_review_page(cls_i)

    def _refresh_review_cur(self, cls_i):
        color = CLASS_COLORS[cls_i]
        n     = len(self.masks_per_cls[cls_i])
        sel   = len(self.selected[cls_i])
        n_pages = max(1, math.ceil(n / self._PAGE_SZ))
        page  = self._rev_page
        start = page * self._PAGE_SZ
        end   = min(start + self._PAGE_SZ, n)

        self.rev_title.set_text(
            f"{self.fname}   |   {CLASS_NAMES[cls_i]} ({cls_i+1}/3)   |   "
            f"{n} kernels  selected: {sel}   |   "
            f"page {page+1}/{n_pages}  ({start+1}-{end})   |   click to toggle")

        for ax, ki in zip(self.rev_axes_cur, self._rev_ki_map):
            for t in ax.texts[:]:
                t.remove()
            ax.text(0.02, 0.97, str(ki), transform=ax.transAxes,
                    fontsize=7, color="white", va="top", ha="left",
                    fontweight="bold",
                    bbox=dict(fc="black", alpha=0.4, pad=0.5, lw=0))
            if ki in self.selected[cls_i]:
                ax.set_facecolor("#ddffdd")
                ax.text(0.5, 0.5, "✓", transform=ax.transAxes,
                        fontsize=30, color="green", ha="center", va="center",
                        fontweight="bold", alpha=0.85)
                for sp in ax.spines.values():
                    sp.set_edgecolor("green"); sp.set_linewidth(3)
            else:
                ax.set_facecolor("#f0f0f0")
                for sp in ax.spines.values():
                    sp.set_edgecolor(color); sp.set_linewidth(1.8)

        self.rev_fig.canvas.draw_idle()

    def _on_review_click(self, event):
        if event.button != 1:
            return
        cls_i = self._review_cls
        for ax, ki in zip(self.rev_axes_cur, self._rev_ki_map):
            if event.inaxes == ax:
                if ki in self.selected[cls_i]:
                    self.selected[cls_i].discard(ki)
                else:
                    self.selected[cls_i].add(ki)
                self._refresh_review_cur(cls_i)
                return

    def _on_review_confirm(self, _):
        plt.close(self.rev_fig)
        next_cls = self._review_cls + 1
        if next_cls < 3:
            self._review_cls = next_cls
            self._open_review_for(next_cls)
        else:
            self._save_kernels()
            self.file_idx += 1
            self._load_next()

    def _save_kernels(self):
        sample = os.path.splitext(self.fname)[0]
        total = 0
        for cls_i, cls_name in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(OUT_DIR, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            saved = 0
            for ki, padded in enumerate(self.padded_per_cls[cls_i]):
                if ki not in self.selected[cls_i]:
                    continue
                out_path = os.path.join(cls_dir, f"{sample}_{cls_name}_kernel_{ki:03d}.npy")
                np.save(out_path, padded)
                saved += 1
            print(f"  [{cls_name}] {saved} kernels saved to {cls_dir}")
            total += saved
        print(f"  {total} kernels saved to {OUT_DIR}\n")
        self._update_info()

    def _show_finished(self):
        self.ax_img.cla()
        self.ax_img.set_facecolor(PANEL)
        self.ax_img.text(0.5, 0.5, "All files done!",
                         transform=self.ax_img.transAxes,
                         ha="center", va="center",
                         fontsize=18, color="green", fontweight="bold")
        self.ax_img.set_xticks([]); self.ax_img.set_yticks([])
        self._update_info(finished=True)
        self.fig.canvas.draw_idle()

    def _update_info(self, finished=False):
        self.ax_info.cla()
        self.ax_info.set_facecolor(PANEL)
        self.ax_info.set_xticks([]); self.ax_info.set_yticks([])
        for sp in self.ax_info.spines.values():
            sp.set_edgecolor(GRID_C)

        n_done  = sum(1 for p in self.file_list if _is_done(p))
        n_total = len(self.file_list)

        lines = [
            ("■ Wheat SAM Extractor", TEXT, 11, "bold"),
            ("", TEXT, 7, "normal"),
            (f"Progress: {n_done} / {n_total}",  SUBTEXT, 9, "normal"),
            (f"Bands: {N_BANDS}",  SUBTEXT, 8, "normal"),
            (f"Output: {PAD_SIZE}×{PAD_SIZE}×{N_BANDS}",   SUBTEXT, 8, "normal"),
            ("", TEXT, 7, "normal"),
            ("── ROI ──",  SUBTEXT, 8, "normal"),
        ]

        for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            roi = self.rois[i]
            if roi:
                r0, c0, r1, c1 = roi
                status = f"[v]  ({c0},{r0})→({c1},{r1})"
            else:
                status = "[ ]  not set"
            n_k = len(self.masks_per_cls[i]) if self.masks_per_cls[i] else 0
            sel = len(self.selected[i])
            extra = f"  detected:{n_k}" + (f" selected:{sel}" if sel else "")
            lines.append((f"  {name}: {status}{extra}", color, 8, "normal"))

        lines += [
            ("", TEXT, 7, "normal"),
            ("── Controls ──", SUBTEXT, 8, "normal"),
            ("  Drag      draw ROI", TEXT, 7, "normal"),
            ("  Run SAM   run SAM2", TEXT, 7, "normal"),
            ("  Review    click thumbnails to select", TEXT, 7, "normal"),
            ("  Confirm   save and go to next", TEXT, 7, "normal"),
            ("  Skip      skip this file", TEXT, 7, "normal"),
            ("  Reset     clear ROIs", TEXT, 7, "normal"),
        ]

        if finished:
            lines += [("", TEXT, 7, "normal"),
                      ("All files done!", "green", 12, "bold")]
        elif self.cur_cls < 3:
            cur_name  = CLASS_NAMES[self.cur_cls]
            cur_color = CLASS_COLORS[self.cur_cls]
            lines += [
                ("", TEXT, 7, "normal"),
                (f">> Draw the {cur_name} ROI", cur_color, 10, "bold"),
            ]
        else:
            lines += [
                ("", TEXT, 7, "normal"),
                (">> ROIs set. Click Run SAM", "forestgreen", 10, "bold"),
            ]

        y = 0.97
        for text, color, size, weight in lines:
            self.ax_info.text(0.05, y, text, transform=self.ax_info.transAxes,
                              color=color, fontsize=size, fontweight=weight,
                              va="top")
            y -= 0.052

        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    print(f"Files: {len(ALL_FILES)}")
    print(f"Output: {OUT_DIR}\n")
    WheatSAMExtractor()
