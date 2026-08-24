# -*- coding: utf-8 -*-
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import scipy.io as sio
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import uniform_filter
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import spectral
import warnings
import xml.etree.ElementTree as ET
warnings.filterwarnings('ignore')

try:
    import tifffile
    HAS_TIFF = True
except ImportError:
    HAS_TIFF = False

CLASS_NAMES = ['Асфальт', 'Луга', 'Гравий', 'Деревья', 'Металлические листы',
               'Голая почва', 'Битум', 'Кирпичная кладка', 'Тени']
N = 9

COL_BG, COL_PANEL, COL_BORDER = '#f2f2f2', '#ffffff', '#d9d9d9'
COL_TEXT, COL_MUTED, COL_ACCENT = '#2b2b2b', '#808080', '#4a76a8'
COL_SEL = '#e6e6e6'
import sys as _sys
_FONT = 'DejaVu Sans' if _sys.platform.startswith('linux') else 'Segoe UI'
FONT_BASE  = (_FONT, 10)
FONT_BOLD  = (_FONT, 10, 'bold')
FONT_TITLE = (_FONT, 12, 'bold')

VIEWS = [
    ('src', 'Исходное изображение',   False),
    ('res', 'Результат кластеризации', False),
    ('sp',  'Спектры кластеров',       False),
    ('gt',  'Эталонная маска',         True),
    ('cmp', 'GT + Результат',          True),
    ('cg',  'Кластер + Эталон',        True),
    ('gts', 'Спектры GT',              True),
    ('hm',  'Карта сходства',          True),
]


def parse_wavelengths_spp(path):
    try:
        tree = ET.parse(path, parser=ET.XMLParser(encoding='windows-1251'))
    except Exception:
        tree = ET.parse(path)
    root = tree.getroot()
    entries = []
    for wl in root.iter('WaveLength'):
        ch  = wl.findtext('ChannelNumber')
        val = wl.findtext('WaveLen')
        if ch is not None and val is not None:
            entries.append((int(ch), float(val)))
    if not entries:
        return None
    entries.sort(key=lambda x: x[0])
    return np.array([v for _, v in entries])

def load_wavelengths(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.spp', '.xml'):
        return parse_wavelengths_spp(path)
    return None

def bands_axis(wavelengths, n_bands):
    if wavelengths is not None and len(wavelengths) == n_bands:
        return wavelengths, 'Длина волны, нм'
    return np.arange(n_bands), 'Канал (band)'

def _load_hsi_mat(path):
    mat = sio.loadmat(path)
    for key in ['paviaU', 'pavia', 'data', 'HSI']:
        if key in mat:
            return mat[key]
    arrays = {k: v for k, v in mat.items()
              if not k.startswith('_') and isinstance(v, np.ndarray) and v.ndim == 3}
    if arrays:
        return next(iter(arrays.values()))
    raise ValueError(f"HSI-массив не найден в {path}")

def _load_hsi_tiff(path, progress_cb=None):
    if not HAS_TIFF:
        raise ImportError("Установите tifffile: pip install tifffile")
    if progress_cb:
        progress_cb('Чтение TIFF (может занять 1-2 мин)…', 1)
    try:
        with tifffile.TiffFile(path) as tif:
            arr = tif.pages[0].asarray() if len(tif.pages) == 1 else tif.asarray()
    except Exception:
        arr = tifffile.imread(path)
    if progress_cb:
        progress_cb('TIFF прочитан, определяем структуру…', 3)
    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[2] or (arr.shape[0] < 500 and arr.shape[2] >= 500):
            arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if progress_cb:
        progress_cb(f'TIFF загружен: {arr.shape[0]}×{arr.shape[1]}, {arr.shape[2]} каналов', 4)
    return arr

def _load_gt_mat(path):
    mat = sio.loadmat(path)
    data_keys = [k for k, v in mat.items()
                 if not k.startswith('_') and isinstance(v, np.ndarray)]
    for key in ['paviaU_gt', 'pavia_gt', 'gt', 'GT', 'labels',
                'groundTruth', 'ground_truth', 'mask', 'reference']:
        if key in mat:
            return mat[key].squeeze()
    for k in data_keys:
        v = mat[k].squeeze()
        if v.ndim == 2:
            return v
    raise ValueError(f"GT не найден в {path}. Ключи: {data_keys}")

def _load_gt_tiff(path):
    if not HAS_TIFF:
        raise ImportError("Установите tifffile: pip install tifffile")
    arr = tifffile.imread(path)
    if arr.ndim > 2:
        arr = arr[:, :, 0]
    return arr.astype(np.int32)

def load_hsi(path, progress_cb=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        return _load_hsi_tiff(path, progress_cb=progress_cb)
    if progress_cb:
        progress_cb('Чтение MAT файла…', 1)
    arr = _load_hsi_mat(path)
    if progress_cb:
        progress_cb('MAT файл прочитан', 4)
    return arr

def load_gt(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        return _load_gt_tiff(path)
    return _load_gt_mat(path)

def match(y_pred, y_true, n):
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = -np.sum((y_pred == i) & (y_true == j))
    r, c = linear_sum_assignment(cost)
    m = dict(zip(r, c))
    return np.array([m[l] for l in y_pred])

def prepare_hsi(HSI, progress_cb=None):
    def cb(msg, pct):
        if progress_cb:
            progress_cb(msg, pct)

    h, w, bands = HSI.shape
    n_pix  = h * w
    n_comp = min(20, bands)
    large  = n_pix > 500_000

    cb(f'Подготовка данных ({h}×{w}, {bands} каналов)…', 2)
    flat = HSI.reshape(-1, bands).astype(np.float32)

    cb('PCA — обучение…', 5)
    pca = IncrementalPCA(n_components=n_comp)
    if large:
        rng = np.random.default_rng(42)
        sample = flat[rng.choice(n_pix, size=min(200_000, n_pix), replace=False)]
        batches = [sample[i:i+10_000] for i in range(0, len(sample), 10_000)]
        for bi, b in enumerate(batches):
            pca.partial_fit(b)
            cb(f'PCA — обучение… {bi+1}/{len(batches)}',
               5 + int(bi / len(batches) * 15))
    else:
        batches = np.array_split(flat, 256)
        for bi, b in enumerate(batches):
            pca.partial_fit(b)
            if bi % 30 == 0:
                cb('PCA — обучение…', 5 + int(bi / len(batches) * 20))

    cb('PCA — трансформация…', 22)
    chunks  = [flat[i:i+50_000] for i in range(0, n_pix, 50_000)]
    X_parts = []
    for ci, chunk in enumerate(chunks):
        X_parts.append(pca.transform(chunk).astype(np.float32))
        cb(f'PCA — трансформация… {ci+1}/{len(chunks)}',
           22 + int(ci / len(chunks) * 20))
    X_pca = np.concatenate(X_parts, axis=0)
    del flat; X_parts.clear()

    X3 = X_pca.reshape(h, w, n_comp)
    if large:
        cb('Пространственное сглаживание пропущено (большой снимок)…', 44)
        Xss = X3.reshape(-1, n_comp)
    else:
        cb('Пространственное сглаживание…', 44)
        sm = np.zeros_like(X3)
        for ch in range(n_comp):
            sm[:, :, ch] = uniform_filter(X3[:, :, ch], size=5)
        Xss = np.concatenate([X3, sm], axis=2).reshape(-1, n_comp * 2)

    cb('Нормализация спектров…', 60)
    raw  = HSI.reshape(-1, bands).astype(np.float32)
    Xraw = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-8)).astype(np.float32)
    del raw

    cb('Подготовка RGB…', 80)
    b_r = min(int(bands * 0.55), bands - 1)
    b_g = min(int(bands * 0.35), bands - 1)
    b_b = min(int(bands * 0.10), bands - 1)
    rgb = HSI[:, :, [b_r, b_g, b_b]].astype(float)
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)

    cb('Готово!', 100)
    return {
        'Xss': Xss, 'Xraw': Xraw,
        'h': h, 'w': w, 'bands': bands,
        'rgb': rgb,
        'GT': None, 'mask': None, 'y': None,
    }

def attach_gt(data, GT):
    mask = GT.flatten() > 0
    data['GT']          = GT
    data['mask']        = mask
    data['y']           = GT.flatten()[mask] - 1
    data['Xraw_masked'] = data['Xraw'][mask]
    data['Xss_masked']  = data['Xss'][mask]
    return data

def get_metrics(y_true, y_pred):
    p = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    r = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return p, r, f

def colormap_full(pred_full, h, w):
    spy = np.array(spectral.spy_colors) / 255.0
    img = np.zeros((h * w, 3))
    for k in range(N):
        img[pred_full == k] = spy[k + 1]
    return img.reshape(h, w, 3)

def cluster_colors():
    spy = np.array(spectral.spy_colors) / 255.0
    return [spy[i + 1] for i in range(N)]

def gt_colormap(GT):
    spy = np.array(spectral.spy_colors) / 255.0
    h, w = GT.shape
    img  = np.zeros((h * w, 3))
    for c in range(1, 10):
        img[GT.flatten() == c] = spy[c]
    return img.reshape(h, w, 3)

def _normalize01(D):
    mn, mx = D.min(), D.max()
    return (D - mn) / (mx - mn + 1e-12)

def spectral_distance(X, C, metric):
    if metric == 'euclid':
        return cdist(X, C, 'euclidean')
    if metric == 'corr':
        return cdist(X, C, 'correlation')
    if metric == 'sam':
        return np.arccos(np.clip(1 - cdist(X, C, 'cosine'), -1, 1))
    raise ValueError(metric)

def spectral_cluster(X, n_clusters, metric='euclid', weights=(1, 1, 1),
                     max_iter=30, seed=None, progress_cb=None):
    rng    = np.random.default_rng(seed)
    C      = X[rng.choice(len(X), n_clusters, replace=False)].astype(float).copy()
    labels = np.full(len(X), -1)
    for it in range(max_iter):
        if metric == 'combo':
            we, ws, wc = weights
            total = we + ws + wc or 1
            D = (_normalize01(spectral_distance(X, C, 'euclid')) * we +
                 _normalize01(spectral_distance(X, C, 'sam'))    * ws +
                 _normalize01(spectral_distance(X, C, 'corr'))   * wc) / total
        else:
            D = spectral_distance(X, C, metric)
        new_labels = D.argmin(axis=1)
        changed    = int(np.sum(new_labels != labels))
        labels     = new_labels
        for k in range(n_clusters):
            pts = X[labels == k]
            if len(pts) > 0:
                C[k] = pts.mean(axis=0)
        if progress_cb:
            progress_cb(it + 1, max_iter)
        if changed == 0 and it > 0:
            break
    return labels


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Кластеризация изображений")
        self.configure(bg=COL_BG)
        self.minsize(1150, 700)
        self._maximize()

        self.hsi_path    = None
        self.meta_path   = None
        self.wavelengths = None
        self.data        = None
        self.pred_full   = None
        self.last_name   = None
        self._busy       = False

        self.view_frames  = {}
        self.nav_buttons  = {}
        self.current_view = None

        self._init_style()
        self._build()

    def _maximize(self):
        try:
            self.state('zoomed')
        except tk.TclError:
            try:
                self.attributes('-zoomed', True)
            except tk.TclError:
                sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
                self.geometry(f"{sw}x{sh}+0+0")

    def _init_style(self):
        style = ttk.Style(self)
        for _theme in ('clam', 'alt', 'default'):
            try:
                style.theme_use(_theme)
                break
            except tk.TclError:
                continue
        self.configure(bg=COL_BG)
        self.option_add('*Font',              FONT_BASE)
        self.option_add('*Background',        COL_BG)
        self.option_add('*Foreground',        COL_TEXT)
        self.option_add('*activeBackground',  COL_SEL)
        self.option_add('*activeForeground',  COL_TEXT)
        self.option_add('*highlightThickness', 0)
        cfg = {
            '.':                              dict(font=FONT_BASE, background=COL_BG, foreground=COL_TEXT),
            'TFrame':                         dict(background=COL_BG),
            'Panel.TFrame':                   dict(background=COL_PANEL),
            'TLabel':                         dict(background=COL_BG, foreground=COL_TEXT),
            'Panel.TLabel':                   dict(background=COL_PANEL, foreground=COL_TEXT),
            'Muted.TLabel':                   dict(background=COL_PANEL, foreground=COL_MUTED, font=(_FONT, 8)),
            'Title.TLabel':                   dict(background=COL_BG, foreground=COL_TEXT, font=FONT_TITLE),
            'Metric.TLabel':                  dict(background=COL_PANEL, foreground=COL_MUTED, font=(_FONT, 9)),
            'MetricVal.TLabel':               dict(background=COL_PANEL, foreground=COL_TEXT, font=FONT_BOLD),
            'Accent.TButton':                 dict(font=FONT_BASE, padding=(10, 7),
                                                   background=COL_ACCENT, foreground='white', borderwidth=0),
            'Plain.TButton':                  dict(font=FONT_BASE, padding=(10, 7),
                                                   background=COL_PANEL, foreground=COL_TEXT, borderwidth=1),
            'Nav.TButton':                    dict(font=FONT_BASE, padding=(12, 8), anchor='w',
                                                   background=COL_BG, foreground=COL_TEXT, borderwidth=0),
            'NavActive.TButton':              dict(font=FONT_BOLD, padding=(12, 8), anchor='w',
                                                   background=COL_SEL, foreground=COL_TEXT, borderwidth=0),
            'NavGT.TButton':                  dict(font=FONT_BASE, padding=(12, 8), anchor='w',
                                                   background=COL_BG, foreground=COL_MUTED, borderwidth=0),
            'MetricActive.TButton':           dict(font=FONT_BOLD, padding=(10, 7),
                                                   background=COL_SEL, foreground=COL_TEXT, borderwidth=1),
            'Horizontal.TScale':              dict(background=COL_PANEL, troughcolor=COL_SEL),
            'Accent.Horizontal.TProgressbar': dict(background=COL_ACCENT, troughcolor=COL_SEL),
            'Treeview':             dict(background=COL_PANEL, foreground=COL_TEXT,
                                         fieldbackground=COL_PANEL, rowheight=24),
            'Treeview.Heading':     dict(background=COL_SEL, foreground=COL_TEXT,
                                         font=FONT_BOLD, relief='flat'),
            'TSeparator':           dict(background=COL_BORDER),
            'TScrollbar':           dict(background=COL_SEL, troughcolor=COL_BG,
                                         borderwidth=0, arrowsize=12),
        }
        for name, opts in cfg.items():
            style.configure(name, **opts)
        style.map('Accent.TButton',       background=[('active', '#3d6491'), ('disabled', '#b7c4d3')])
        style.map('Plain.TButton',        background=[('active', COL_SEL)])
        style.map('MetricActive.TButton', background=[('active', COL_SEL)])
        style.map('Nav.TButton',          background=[('active', COL_SEL)])
        style.map('NavGT.TButton',        background=[('active', COL_SEL)])

    def _build(self):
        top = ttk.Frame(self)
        top.pack(side='top', fill='x')
        ttk.Label(top, text="Кластеризация изображений",
                  style='Title.TLabel').pack(anchor='w', padx=16, pady=12)
        ttk.Separator(self).pack(fill='x')

        body = ttk.Frame(self)
        body.pack(side='top', fill='both', expand=True)

        sidebar = ttk.Frame(body, style='Panel.TFrame', width=240)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="ФАЙЛЫ", style='Muted.TLabel').pack(anchor='w', padx=14, pady=(14, 2))

        self.btn_hsi = ttk.Button(sidebar, text="Открыть HSI…",
                                  style='Plain.TButton', command=self._open_hsi)
        self.btn_hsi.pack(fill='x', padx=14, pady=(0, 2))
        self.lbl_hsi = ttk.Label(sidebar, text="", style='Muted.TLabel', wraplength=210)
        self.lbl_hsi.pack(anchor='w', padx=16)

        self.btn_meta = ttk.Button(sidebar, text="Открыть метаданные (.spp/.xml)…",
                                   style='Plain.TButton', command=self._open_meta)
        self.btn_meta.pack(fill='x', padx=14, pady=(2, 2))
        self.lbl_meta = ttk.Label(sidebar, text="", style='Muted.TLabel', wraplength=210)
        self.lbl_meta.pack(anchor='w', padx=16)

        self.btn_gt = ttk.Button(sidebar, text="Открыть GT…",
                                 style='Plain.TButton', command=self._open_gt_file)
        self.btn_gt.pack(fill='x', padx=14, pady=(2, 2))
        self.lbl_gt = ttk.Label(sidebar, text="GT не загружен", style='Muted.TLabel', wraplength=210)
        self.lbl_gt.pack(anchor='w', padx=16, pady=(0, 4))

        self.btn_prepare = ttk.Button(sidebar, text="Загрузить данные",
                                      style='Accent.TButton', command=self._prepare)
        self.btn_prepare.pack(fill='x', padx=14, pady=(4, 2))
        self.btn_clear = ttk.Button(sidebar, text="Очистить",
                                    style='Plain.TButton', command=self._clear_all)
        self.btn_clear.pack(fill='x', padx=14, pady=(0, 0))
        ttk.Separator(sidebar).pack(fill='x', padx=14, pady=10)

        ttk.Label(sidebar, text="АЛГОРИТМ", style='Muted.TLabel').pack(anchor='w', padx=14, pady=(0, 4))
        self.algo_var     = tk.StringVar(value='km')
        self.algo_buttons = {}
        for val, txt in [('km', 'K-Means'), ('gmm', 'GMM'), ('kms', 'K-Means + seed…')]:
            b = ttk.Button(sidebar, text=txt, style='Plain.TButton',
                           command=lambda v=val: self._set_algo(v))
            b.pack(fill='x', padx=14, pady=2)
            self.algo_buttons[val] = b
        self._cluster_buttons = list(self.algo_buttons.values())
        ttk.Separator(sidebar).pack(fill='x', padx=14, pady=10)

        ttk.Label(sidebar, text="МЕТРИКА РАССТОЯНИЯ", style='Muted.TLabel').pack(anchor='w', padx=14, pady=(0, 4))
        self.metric_var     = tk.StringVar(value='euclid')
        self.metric_buttons = {}
        for val, txt in [('euclid', 'Евклидово расстояние'),
                          ('sam',    'Спектральный угол (SAM)'),
                          ('corr',   'Корреляция'),
                          ('combo',  'Комбо (с весами)')]:
            b = ttk.Button(sidebar, text=txt, style='Plain.TButton',
                           command=lambda v=val: self._set_metric(v))
            b.pack(fill='x', padx=14, pady=2)
            self.metric_buttons[val] = b

        self.weights_frame = ttk.Frame(sidebar, style='Panel.TFrame')
        self.weights_frame.pack(fill='x', padx=14, pady=(4, 0))
        self.w_euclid      = tk.DoubleVar(value=1.0)
        self.w_sam         = tk.DoubleVar(value=1.0)
        self.w_corr        = tk.DoubleVar(value=1.0)
        self.weight_scales = []
        for lbl, var in [("Евклид", self.w_euclid), ("SAM", self.w_sam), ("Корреляция", self.w_corr)]:
            row = ttk.Frame(self.weights_frame, style='Panel.TFrame')
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=lbl, style='Metric.TLabel', width=9).pack(side='left')
            val_lbl = ttk.Label(row, text=f"{var.get():.2f}", style='MetricVal.TLabel',
                                width=4, anchor='e')
            val_lbl.pack(side='right')
            sc = ttk.Scale(row, from_=0, to=1, orient='horizontal', variable=var,
                           command=lambda v, l=val_lbl: l.config(text=f"{float(v):.2f}"))
            sc.pack(side='left', fill='x', expand=True, padx=(4, 6))
            self.weight_scales.append(sc)

        self._cluster_buttons.extend(self.metric_buttons.values())
        self._toggle_weights()

        self.btn_run = ttk.Button(sidebar, text="Запустить кластеризацию",
                                  style='Accent.TButton', command=self._run_current)
        self.btn_run.pack(fill='x', padx=14, pady=(10, 4))
        self._cluster_buttons.append(self.btn_run)
        ttk.Separator(sidebar).pack(fill='x', padx=14, pady=10)

        results = ttk.Frame(sidebar, style='Panel.TFrame')
        results.pack(fill='x', padx=14)
        self.lbl_method_val = self._metric_row(results, "Метод")
        self.lbl_prec_val   = self._metric_row(results, "Точность")
        self.lbl_rec_val    = self._metric_row(results, "Полнота")
        self.lbl_f1_val     = self._metric_row(results, "F1-мера")
        self.progress_lbl   = ttk.Label(results, text="", style='Muted.TLabel')
        self.progress_lbl.pack(anchor='w', pady=(8, 2))
        self.progress_bar   = ttk.Progressbar(results, mode='determinate', maximum=100,
                                               style='Accent.Horizontal.TProgressbar')
        self.progress_bar.pack(fill='x')

        ttk.Separator(body, orient='vertical').pack(side='left', fill='y')

        nav = ttk.Frame(body, width=190)
        nav.pack(side='left', fill='y')
        nav.pack_propagate(False)
        for key, title, needs_gt in VIEWS:
            style_name = 'NavGT.TButton' if needs_gt else 'Nav.TButton'
            b = ttk.Button(nav, text=title, style=style_name,
                           command=lambda k=key, g=needs_gt: self._nav_click(k, g))
            b.pack(fill='x')
            self.nav_buttons[key] = b

        ttk.Separator(body, orient='vertical').pack(side='left', fill='y')

        self.content = ttk.Frame(body)
        self.content.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.fig_src, self.ax_src,  self.cv_src = self._make_tab('src')
        self.fig_res, self.ax_res2, self.cv_res = self._make_tab('res')
        self.fig_sp,  self.ax_sp,   self.cv_sp  = self._make_grid_tab('sp', 3, 3)
        self.fig_gt2, self.ax_gt2,  self.cv_gt2 = self._make_tab('gt')
        self.fig_cmp, self.ax_cmp,  self.cv_cmp = self._make_grid_tab('cmp', 1, 3)
        self.fig_cg,  self.ax_cg,   self.cv_cg  = self._make_grid_tab('cg', 3, 3)
        self.fig_gts, self.ax_gts,  self.cv_gts = self._make_tab('gts')
        self.fig_hm,  self.ax_hm,   self.cv_hm  = self._make_tab('hm')

        self._set_algo('km')
        self.metric_var.set('euclid')
        self._show_view('src')
        self.after(200, self._setup_spectra_tooltip)
        self.after(200, self._setup_result_tooltip)

    def _metric_row(self, parent, label):
        row = ttk.Frame(parent, style='Panel.TFrame')
        row.pack(fill='x', pady=2)
        ttk.Label(row, text=label, style='Metric.TLabel').pack(side='left')
        val = ttk.Label(row, text="—", style='MetricVal.TLabel')
        val.pack(side='right')
        return val

    def _clear_all(self):
        if self._busy:
            return
        self.hsi_path    = None
        self.meta_path   = None
        self.wavelengths = None
        self.data        = None
        self.pred_full   = None
        self.last_name   = None
        if hasattr(self, '_pending_gt_path'):
            self._pending_gt_path = None

        self.lbl_hsi.config(text="")
        self.lbl_meta.config(text="")
        self.lbl_gt.config(text="GT не загружен")
        self.lbl_method_val.config(text="—")
        self.lbl_prec_val.config(text="—")
        self.lbl_rec_val.config(text="—")
        self.lbl_f1_val.config(text="—")
        self._reset_progress()

        for ax in [self.ax_src, self.ax_res2, self.ax_gt2, self.ax_gts, self.ax_hm]:
            ax.clear(); ax.axis('off')
        for ax in self.ax_sp + self.ax_cg + self.ax_cmp:
            ax.clear(); ax.axis('off')
        for cv in [self.cv_src, self.cv_res, self.cv_sp, self.cv_gt2,
                   self.cv_cmp, self.cv_cg, self.cv_gts, self.cv_hm]:
            cv.draw()

        for key, _, needs_gt in VIEWS:
            style = 'NavGT.TButton' if needs_gt else 'Nav.TButton'
            self.nav_buttons[key].config(style=style)

        self._show_view('src')

    def _set_busy(self, busy):
        self._busy = busy
        state = 'disabled' if busy else 'normal'
        for b in [self.btn_hsi, self.btn_meta, self.btn_gt, self.btn_prepare] + self._cluster_buttons:
            b.config(state=state)

    def _set_algo(self, val):
        self.algo_var.set(val)
        for k, b in self.algo_buttons.items():
            b.config(style='MetricActive.TButton' if k == val else 'Plain.TButton')

    def _set_metric(self, val):
        self.metric_var.set(val)
        for k, b in self.metric_buttons.items():
            b.config(style='MetricActive.TButton' if k == val else 'Plain.TButton')
        self._toggle_weights()

    def _toggle_weights(self):
        state = 'normal' if self.metric_var.get() == 'combo' else 'disabled'
        for sc in self.weight_scales:
            sc.config(state=state)

    def _run_current(self):
        metric = self.metric_var.get()
        algo   = self.algo_var.get()
        if metric in ('sam', 'corr', 'combo'):
            self._run_spectral()
        else:
            self._run(algo)

    def _update_progress(self, it, total):
        pct = int(it / total * 100)
        self.progress_bar['value'] = pct
        self.progress_lbl.config(text=f"Кластеризация: {pct}% (итерация {it}/{total})")

    def _reset_progress(self):
        self.progress_bar['value'] = 0
        self.progress_lbl.config(text="")

    def _unlock_nav(self, include_gt=False):
        has_gt = include_gt or (self.data is not None and self.data.get('GT') is not None)
        for key, _, needs_gt in VIEWS:
            if not needs_gt or has_gt:
                self.nav_buttons[key].config(style='Nav.TButton')

    def _nav_click(self, key, needs_gt):
        if needs_gt and (self.data is None or self.data.get('GT') is None):
            messagebox.showinfo("GT не загружен",
                                "Этот вид требует эталонной маски (GT).\n"
                                "Нажмите «Открыть GT…» и выберите файл.")
            return
        self._show_view(key)

    def _show_view(self, key):
        self.view_frames[key].tkraise()
        for k, b in self.nav_buttons.items():
            _, _, needs_gt = next(v for v in VIEWS if v[0] == k)
            if k == key:
                b.config(style='NavActive.TButton')
            elif needs_gt and (self.data is None or self.data.get('GT') is None):
                b.config(style='NavGT.TButton')
            else:
                b.config(style='Nav.TButton')
        self.current_view = key

    def _make_tab(self, key):
        f   = ttk.Frame(self.content)
        f.grid(row=0, column=0, sticky='nsew')
        self.view_frames[key] = f
        fig = plt.Figure(tight_layout=True)
        ax  = fig.add_subplot(111)
        ax.axis('off')
        cv  = FigureCanvasTkAgg(fig, master=f)
        cv.get_tk_widget().pack(fill='both', expand=True)
        return fig, ax, cv

    def _make_grid_tab(self, key, rows, cols):
        f   = ttk.Frame(self.content)
        f.grid(row=0, column=0, sticky='nsew')
        self.view_frames[key] = f
        fig  = plt.Figure(tight_layout=True)
        axes = [fig.add_subplot(rows, cols, i + 1) for i in range(rows * cols)]
        for ax in axes:
            ax.axis('off')
        cv = FigureCanvasTkAgg(fig, master=f)
        cv.get_tk_widget().pack(fill='both', expand=True)
        return fig, axes, cv

    def _setup_spectra_tooltip(self):
        tooltip_lbl = tk.Label(
            self._panels['sp'] if hasattr(self, '_panels') else self.view_frames['sp'],
            text='', bg='#fffbe6', fg='#333', relief='solid', bd=1,
            font=(_FONT, 9), padx=6, pady=3
        )

        def on_move(event):
            if event.inaxes is None:
                tooltip_lbl.place_forget()
                return
            for i, ax in enumerate(self.ax_sp):
                if ax is event.inaxes:
                    colors = cluster_colors()
                    c      = colors[i]
                    hex_c  = '#{:02x}{:02x}{:02x}'.format(
                        int(c[0]*255), int(c[1]*255), int(c[2]*255))
                    n_pix = int((self.pred_full == i).sum()) if self.pred_full is not None else 0
                    parts = [f'Кластер {i}']
                    if event.xdata is not None and self.data is not None:
                        wl = self.data.get('wavelengths')
                        if wl is not None and len(wl) > 0:
                            parts.append(f'{event.xdata:.1f} нм')
                        else:
                            parts.append(f'канал {int(event.xdata)}')
                    if n_pix > 0:
                        parts.append(f'{n_pix:,} пикс.')
                    tooltip_lbl.config(text='   |   '.join(parts),
                                       fg='#222')
                    widget = self.cv_sp.get_tk_widget()
                    x = int(event.guiEvent.x) + 14
                    y = int(event.guiEvent.y) - 30
                    tooltip_lbl.place(in_=widget, x=x, y=y)
                    tooltip_lbl.lift()
                    return
            tooltip_lbl.place_forget()

        self.fig_sp.canvas.mpl_connect('motion_notify_event', on_move)

    def _setup_result_tooltip(self):
        panel  = self.view_frames['res']
        widget = self.cv_res.get_tk_widget()

        tip = tk.Label(panel, text='', bg='#fffbe6', fg='#333',
                       relief='solid', bd=1, font=(_FONT, 9),
                       padx=6, pady=3)

        def on_move(event):
            if self.pred_full is None or self.data is None:
                tip.place_forget()
                return
            ax = self.ax_res2
            if not ax.get_images():
                tip.place_forget()
                return
            contains, _ = ax.contains(event)
            if not contains or event.xdata is None or event.ydata is None:
                tip.place_forget()
                return

            col = int(round(event.xdata))
            row = int(round(event.ydata))
            h, w = self.data['h'], self.data['w']
            if not (0 <= col < w and 0 <= row < h):
                tip.place_forget()
                return

            pix_idx   = row * w + col
            cluster   = int(self.pred_full[pix_idx])
            colors    = cluster_colors()
            c         = colors[cluster]
            hex_c     = '#{:02x}{:02x}{:02x}'.format(
                int(c[0]*255), int(c[1]*255), int(c[2]*255))
            n_pix     = int((self.pred_full == cluster).sum())
            parts     = [f'Кластер {cluster}', f'{n_pix:,} пикс.']
            if self.data.get('GT') is not None:
                GT_val = int(self.data['GT'][row, col])
                if GT_val > 0:
                    parts.append(f'GT: {CLASS_NAMES[GT_val - 1]}')
            tip.config(text='   |   '.join(parts),
                       fg='#222')
            x = int(event.guiEvent.x) + 14
            y = int(event.guiEvent.y) - 30
            tip.place(in_=widget, x=x, y=y)
            tip.lift()

        def on_leave(event):
            tip.place_forget()

        self.fig_res.canvas.mpl_connect('motion_notify_event', on_move)
        self.fig_res.canvas.mpl_connect('axes_leave_event', on_leave)

    def _open_hsi(self):
        ft = [("HSI файлы", "*.mat *.tif *.tiff"), ("MAT", "*.mat"),
              ("TIFF", "*.tif *.tiff"), ("Все", "*.*")]
        p  = filedialog.askopenfilename(filetypes=ft)
        if p:
            self.hsi_path = p
            self.lbl_hsi.config(text=os.path.basename(p))
            self.meta_path   = None
            self.wavelengths = None
            self.lbl_meta.config(text="")
            self.data        = None
            self.pred_full   = None
            self.last_name   = None
            if hasattr(self, '_pending_gt_path'):
                self._pending_gt_path = None
            self.lbl_gt.config(text="GT не загружен")
            self.lbl_method_val.config(text="—")
            self.lbl_prec_val.config(text="—")
            self.lbl_rec_val.config(text="—")
            self.lbl_f1_val.config(text="—")
            self._reset_progress()
            for key, _, needs_gt in VIEWS:
                style = 'NavGT.TButton' if needs_gt else 'Nav.TButton'
                self.nav_buttons[key].config(style=style)

    def _open_meta(self):
        ft = [("Метаданные", "*.spp *.xml"), ("SPP", "*.spp"),
              ("XML", "*.xml"), ("Все", "*.*")]
        p  = filedialog.askopenfilename(filetypes=ft)
        if not p:
            return
        try:
            wl = load_wavelengths(p)
            if wl is not None:
                self.wavelengths = wl
                self.meta_path   = p
                self.lbl_meta.config(
                    text=f"{os.path.basename(p)}  ({len(wl)} кан., {wl[0]:.0f}–{wl[-1]:.0f} нм)")
                if self.data is not None:
                    self.data['wavelengths'] = wl
                    if self.pred_full is not None:
                        self._draw_cluster_spectra(self.pred_full)
                    if self.data.get('y') is not None:
                        self._draw_gt_spectra()
            else:
                self.lbl_meta.config(text="Длины волн не найдены в файле")
        except Exception as e:
            messagebox.showerror("Ошибка метаданных", str(e))

    def _open_gt_file(self):
        ft = [("GT файлы", "*.mat *.tif *.tiff"), ("MAT", "*.mat"),
              ("TIFF", "*.tif *.tiff"), ("Все", "*.*")]
        p  = filedialog.askopenfilename(filetypes=ft)
        if not p:
            return
        if self.data is None:
            self._pending_gt_path = p
            self.lbl_gt.config(text=f"{os.path.basename(p)} (загрузится после HSI)")
            return
        self._apply_gt(p)

    def _apply_gt(self, path):
        try:
            GT   = load_gt(path)
            h, w = self.data['h'], self.data['w']
            if GT.shape != (h, w):
                messagebox.showerror("Ошибка",
                    f"Размер GT {GT.shape} не совпадает с HSI ({h}×{w}).")
                return
            attach_gt(self.data, GT)
            self.lbl_gt.config(text=os.path.basename(path))

            self.ax_gt2.clear()
            self.ax_gt2.imshow(gt_colormap(GT))
            self.ax_gt2.axis('off')
            self.cv_gt2.draw()
            self._draw_gt_spectra()
            self._unlock_nav(include_gt=True)

            if self.pred_full is not None:
                pred_masked = self.pred_full[self.data['mask']]
                _, col = linear_sum_assignment(
                    -np.array([[np.sum((pred_masked == i) & (self.data['y'] == j))
                                for j in range(N)] for i in range(N)]))
                mapping = {i: col[i] for i in range(N)}
                remapped = np.array([mapping[l] for l in self.pred_full])
                self.pred_full = remapped

                self.ax_res2.clear()
                self.ax_res2.imshow(colormap_full(remapped, h, w))
                self.ax_res2.axis('off')
                self.cv_res.draw()

                self._draw_cluster_spectra(remapped)
                self._evaluate_with_gt()
                self._show_view('gt')
            else:
                self._show_view('gt')
        except Exception as e:
            messagebox.showerror("Ошибка загрузки GT", str(e))

    def _open_gt_and_evaluate(self):
        self._open_gt_file()

    def _prepare(self):
        if self._busy:
            return
        if not self.hsi_path:
            messagebox.showwarning("Нет файла", "Выберите HSI-файл.")
            return

        def work():
            try:
                def cb(msg, pct):
                    self.after(0, lambda m=msg, p=pct: (
                        self.progress_lbl.config(text=m),
                        self.progress_bar.config(value=p)
                    ))
                HSI        = load_hsi(self.hsi_path, progress_cb=cb)
                self.data  = prepare_hsi(HSI, progress_cb=cb)
                self.data['wavelengths'] = self.wavelengths
                self.after(0, self._on_hsi_ready)
                self.after(0, lambda: self._set_busy(False))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
                self.after(0, lambda: self.progress_lbl.config(text='Ошибка загрузки'))
                self.after(0, lambda: self._set_busy(False))

        self._set_busy(True)
        self.progress_lbl.config(text="Загрузка…")
        threading.Thread(target=work, daemon=True).start()

    def _on_hsi_ready(self):
        d = self.data
        self.ax_src.clear()
        self.ax_src.imshow(d['rgb'])
        self.ax_src.axis('off')
        self.cv_src.draw()
        self.progress_bar.config(value=100)
        self.progress_lbl.config(
            text=f"✓ Загружено: {d['h']}×{d['w']}, {d['bands']} каналов")
        self._unlock_nav(include_gt=False)
        self._show_view('src')

        if hasattr(self, '_pending_gt_path') and self._pending_gt_path:
            path = self._pending_gt_path
            self._pending_gt_path = None
            self._apply_gt(path)

    def _run(self, mode):
        if self._busy:
            return
        if self.data is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные.")
            return
        seeds = None
        if mode == 'kms':
            seeds = self._seed_dialog()
            if seeds is None:
                return

        pulse_active = [True]

        def _pulse(step=0):
            if not pulse_active[0]:
                return
            self.progress_bar.config(value=10 + (step % 80))
            self.after(200, lambda: _pulse(step + 5))

        def work():
            try:
                d = self.data
                X = d['Xss']
                if mode == 'km':
                    self.after(0, lambda: self.progress_lbl.config(text='K-Means — вычисление…'))
                    self.after(0, _pulse)
                    pred_full = KMeans(n_clusters=N, n_init=10, random_state=42).fit_predict(X)
                    name      = "K-Means"
                elif mode == 'gmm':
                    self.after(0, lambda: self.progress_lbl.config(text='GMM — вычисление…'))
                    self.after(0, _pulse)
                    pred_full = GaussianMixture(n_components=N, n_init=3, random_state=42).fit_predict(X)
                    name      = "GMM"
                else:
                    self.after(0, lambda: self.progress_lbl.config(text='K-Means seed — вычисление…'))
                    self.after(0, _pulse)
                    if seeds == 'random':
                        rng  = np.random.default_rng()
                        init = X[rng.choice(len(X), N, replace=False)]
                        name = "K-Means (случайные seed)"
                    else:
                        init = np.zeros((N, X.shape[1]))
                        for i, idxs in enumerate(seeds):
                            init[i] = X[idxs].mean(0) if idxs else \
                                       X[[np.random.randint(len(X))]].mean(0)
                        name = "K-Means (ручные seed)"
                    pred_full = KMeans(n_clusters=N, init=init, n_init=1, random_state=42).fit_predict(X)

                pulse_active[0] = False

                cur = self.data   # свежее состояние data на момент завершения
                if cur.get('y') is not None:
                    self.after(0, lambda: self.progress_lbl.config(text='Сопоставление кластеров с GT…'))
                    _, col = linear_sum_assignment(
                        -np.array([[np.sum((pred_full[cur['mask']] == i) & (cur['y'] == j))
                                    for j in range(N)] for i in range(N)]))
                    mapping   = {i: col[i] for i in range(N)}
                    pred_full = np.array([mapping[l] for l in pred_full])

                self.pred_full = pred_full
                self.last_name = name
                self.after(0, lambda: self.progress_bar.config(value=100))
                self.after(0, lambda n=name: self.progress_lbl.config(text=f'✓ {n} завершён'))
                self.after(0, lambda: self._finish_clustering(name))
                self.after(0, lambda: self._set_busy(False))
            except Exception as e:
                pulse_active[0] = False
                self.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
                self.after(0, self._reset_progress)
                self.after(0, lambda: self._set_busy(False))

        self._set_busy(True)
        threading.Thread(target=work, daemon=True).start()

    def _run_spectral(self):
        if self._busy:
            return
        if self.data is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные.")
            return
        metric  = self.metric_var.get()
        weights = (self.w_euclid.get(), self.w_sam.get(), self.w_corr.get())
        names   = {'euclid': 'Евклидово расстояние',
                   'sam':    'Спектральный угол (SAM)',
                   'corr':   'Корреляция'}

        def progress_cb(it, total):
            self.after(0, lambda: self._update_progress(it, total))

        def work():
            try:
                d         = self.data
                pred_full = spectral_cluster(d['Xraw'], N, metric=metric, weights=weights,
                                             max_iter=30, progress_cb=progress_cb)
                if metric == 'combo':
                    we, ws, wc = weights
                    name = f"Комбо (Евкл={we:.2f}, SAM={ws:.2f}, Корр={wc:.2f})"
                else:
                    name = names[metric]

                cur = self.data
                if cur.get('y') is not None:
                    _, col = linear_sum_assignment(
                        -np.array([[np.sum((pred_full[cur['mask']] == i) & (cur['y'] == j))
                                    for j in range(N)] for i in range(N)]))
                    mapping   = {i: col[i] for i in range(N)}
                    pred_full = np.array([mapping[l] for l in pred_full])

                self.pred_full = pred_full
                self.last_name = name
                self.after(0, lambda: self.progress_bar.config(value=100))
                self.after(0, lambda n=name: self.progress_lbl.config(text=f'✓ {n} завершён'))
                self.after(0, lambda: self._finish_clustering(name))
                self.after(0, lambda: self._set_busy(False))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
                self.after(0, self._reset_progress)
                self.after(0, lambda: self._set_busy(False))

        self._set_busy(True)
        self.progress_lbl.config(text="Кластеризация: 0%")
        self.progress_bar['value'] = 0
        threading.Thread(target=work, daemon=True).start()

    def _finish_clustering(self, name):
        d    = self.data   # всегда читаем свежий self.data
        pred = self.pred_full
        has_gt = d is not None and d.get('GT') is not None

        self.ax_res2.clear()
        self.ax_res2.imshow(colormap_full(pred, d['h'], d['w']))
        self.ax_res2.axis('off')
        self.cv_res.draw()

        self._draw_cluster_spectra(pred)
        self._unlock_nav(include_gt=has_gt)

        if has_gt:
            self._evaluate_with_gt()
        else:
            self.lbl_method_val.config(text=name)
            self.lbl_prec_val.config(text="—")
            self.lbl_rec_val.config(text="—")
            self.lbl_f1_val.config(text="—")

        self._show_view('res')

    def _evaluate_with_gt(self):
        d           = self.data
        pred        = self.pred_full
        name        = self.last_name
        pred_masked = pred[d['mask']]
        p, r, f     = get_metrics(d['y'], pred_masked)

        self.lbl_method_val.config(text=name or "—")
        self.lbl_prec_val.config(text=f"{p:.4f}")
        self.lbl_rec_val.config(text=f"{r:.4f}")
        self.lbl_f1_val.config(text=f"{f:.4f}")

        imgs   = [d['rgb'], gt_colormap(d['GT']), colormap_full(pred, d['h'], d['w'])]
        titles = ['Исходное RGB', 'Эталонная маска', f'Результат: {name}']
        for ax, img, t in zip(self.ax_cmp, imgs, titles):
            ax.clear(); ax.imshow(img); ax.axis('off'); ax.set_title(t, fontsize=9)
        self.fig_cmp.tight_layout(); self.cv_cmp.draw()

        self._draw_cluster_vs_gt(pred_masked)
        self._draw_heatmap(pred_masked)
        self._show_metrics_window(name, d['y'], pred_masked, p, r, f)

    def _show_metrics_window(self, name, y_true, y_pred, p, r, f):
        win = tk.Toplevel(self)
        win.title("Оценка качества кластеризации")
        win.configure(bg=COL_BG)
        win.geometry("700x560")
        win.minsize(700, 400)
        win.transient(self)

        ttk.Label(win, text=f"Метод: {name}", style='Title.TLabel').pack(anchor='w', padx=16, pady=(14, 4))
        summary = ttk.Frame(win, style='Panel.TFrame')
        summary.pack(fill='x', padx=14, pady=(0, 10))
        for lbl, val in [("Точность (Precision)", f"{p:.4f}"),
                          ("Полнота (Recall)",     f"{r:.4f}"),
                          ("F1-мера",              f"{f:.4f}")]:
            row = ttk.Frame(summary, style='Panel.TFrame')
            row.pack(fill='x', pady=3, padx=10)
            ttk.Label(row, text=lbl, style='Metric.TLabel').pack(side='left')
            ttk.Label(row, text=val, style='MetricVal.TLabel').pack(side='right')

        ttk.Separator(win).pack(fill='x', padx=14, pady=4)
        ttk.Label(win, text="По классам:", style='Muted.TLabel').pack(anchor='w', padx=16, pady=(0, 4))

        frame_tree = tk.Frame(win, bg=COL_BG)
        frame_tree.pack(fill='both', expand=True, padx=14, pady=(0, 4))
        sb   = ttk.Scrollbar(frame_tree, orient='vertical')
        cols = ('Класс', 'Precision', 'Recall', 'F1', 'Пикселей')
        tree = ttk.Treeview(frame_tree, columns=cols, show='headings',
                             yscrollcommand=sb.set)
        sb.config(command=tree.yview)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')
        tree.column('Класс', width=160, anchor='w')

        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                        output_dict=True, zero_division=0)
        for cn in CLASS_NAMES:
            row = report.get(cn, {})
            tree.insert('', 'end', values=(
                cn,
                f"{row.get('precision', 0):.3f}",
                f"{row.get('recall', 0):.3f}",
                f"{row.get('f1-score', 0):.3f}",
                int(row.get('support', 0))
            ))

        sb.pack(side='right', fill='y')
        tree.pack(side='left', fill='both', expand=True)

        btn_frame = tk.Frame(win, bg=COL_BG)
        btn_frame.pack(fill='x', padx=14, pady=(0, 10))
        ttk.Button(btn_frame, text="Закрыть", style='Plain.TButton',
                   command=win.destroy).pack(side='right', ipadx=20)

    def _draw_cluster_spectra(self, pred_full):
        d               = self.data
        X               = d['Xraw']
        xvals, xlabel   = bands_axis(d.get('wavelengths'), X.shape[1])
        has_gt          = d.get('y') is not None
        mask            = d.get('mask')
        colors          = cluster_colors()

        for i, ax in enumerate(self.ax_sp):
            ax.clear(); ax.axis('on')
            c = colors[i]
            m = pred_full == i
            if m.sum() > 0:
                sp         = X[m]
                mean, std  = sp.mean(0), sp.std(0)
                ax.plot(xvals, mean, color=c, lw=2, label='Средний спектр')
                ax.fill_between(xvals, mean - std, mean + std, alpha=0.25, color=c, label='±СКО')
                title = f'Кластер {i}'
                if has_gt and mask is not None:
                    m_masked = pred_full[mask] == i
                    if m_masked.sum() > 0:
                        dom    = CLASS_NAMES[np.bincount(d['y'][m_masked]).argmax()]
                        pur    = np.bincount(d['y'][m_masked]).max() / m_masked.sum()
                        title += f'  |  {dom} ({pur:.1%})'
                ax.set_title(title, fontsize=8,
                             bbox=dict(facecolor=c, alpha=0.2, edgecolor=c, pad=3))
                for spine in ax.spines.values():
                    spine.set_edgecolor(c)
                    spine.set_linewidth(1.5)
            else:
                ax.set_title(f'Кластер {i} (пустой)', fontsize=8)
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_ylabel('Отражение', fontsize=7)
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=6)
        self.fig_sp.tight_layout()
        self.cv_sp.draw()

    def _draw_cluster_vs_gt(self, pred_masked):
        d             = self.data
        X, y          = d['Xraw_masked'], d['y']
        xvals, xlabel = bands_axis(d.get('wavelengths'), X.shape[1])
        for i, ax in enumerate(self.ax_cg):
            ax.clear(); ax.axis('on')
            for arr, color, ls, lbl in [(pred_masked == i, 'steelblue', '-',  'Кластер'),
                                         (y == i,           'tomato',    '--', 'Эталон')]:
                if arr.sum() > 0:
                    sp        = X[arr]
                    mean, std = sp.mean(0), sp.std(0)
                    ax.plot(xvals, mean, color=color, ls=ls, lw=1.5, label=lbl)
                    ax.fill_between(xvals, mean - std, mean + std, alpha=0.2, color=color)
            m_cl = pred_masked == i
            m_gt = y == i
            if m_cl.sum() > 0 and m_gt.sum() > 0:
                v_cl = X[m_cl].mean(0)
                v_gt = X[m_gt].mean(0)
                cos  = np.dot(v_cl, v_gt) / (np.linalg.norm(v_cl) * np.linalg.norm(v_gt) + 1e-9)
                ax.text(0.97, 0.05, f'cos={cos:.3f}', transform=ax.transAxes,
                        ha='right', fontsize=7, color='steelblue', fontweight='bold')
            ax.set_title(CLASS_NAMES[i], fontsize=8)
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_ylabel('Отражение', fontsize=7)
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=6)
        self.fig_cg.tight_layout()
        self.cv_cg.draw()

    def _draw_gt_spectra(self):
        if self.data is None or self.data.get('y') is None:
            return
        d             = self.data
        X, y          = d['Xraw_masked'], d['y']
        xvals, xlabel = bands_axis(d.get('wavelengths'), X.shape[1])
        colors        = plt.cm.tab10(np.linspace(0, 1, N))
        ax            = self.ax_gts
        ax.clear(); ax.axis('on')
        for c in range(N):
            m = y == c
            if m.sum() == 0:
                continue
            sp        = X[m]
            mean, std = sp.mean(0), sp.std(0)
            ax.plot(xvals, mean, color=colors[c], lw=1.5,
                    label=f'{CLASS_NAMES[c]} ({m.sum()})')
            ax.fill_between(xvals, mean - std, mean + std, alpha=0.10, color=colors[c])
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Нормализованное отражение', fontsize=9)
        ax.legend(fontsize=7, ncol=3, loc='upper right')
        ax.grid(alpha=0.3)
        self.fig_gts.tight_layout()
        self.cv_gts.draw()

    def _draw_heatmap(self, pred_masked):
        d    = self.data
        X, y = d['Xraw_masked'], d['y']
        mat  = np.zeros((N, N))
        for gt_c in range(N):
            m_gt = y == gt_c
            if m_gt.sum() == 0:
                continue
            v_gt = X[m_gt].mean(0)
            for cl in range(N):
                m_cl = pred_masked == cl
                if m_cl.sum() == 0:
                    continue
                v_cl            = X[m_cl].mean(0)
                mat[gt_c, cl]   = np.dot(v_gt, v_cl) / (
                    np.linalg.norm(v_gt) * np.linalg.norm(v_cl) + 1e-9)
        self.fig_hm.clf()
        ax       = self.fig_hm.add_subplot(111)
        self.ax_hm = ax
        im = ax.imshow(mat, vmin=0.7, vmax=1.0, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(N))
        ax.set_xticklabels([f'Кл.{i}' for i in range(N)], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(N))
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        self.fig_hm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for r in range(N):
            for c in range(N):
                ax.text(c, r, f'{mat[r, c]:.2f}', ha='center', va='center',
                        fontsize=7, color='black' if mat[r, c] > 0.85 else 'white')
        self.fig_hm.tight_layout()
        self.cv_hm.draw()

    def _seed_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("Seed-точки K-Means")
        dlg.configure(bg=COL_PANEL)
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()
        result = [None]
        mode   = tk.StringVar(value="random")

        def toggle():
            s = 'normal' if mode.get() == 'manual' else 'disabled'
            for e in entries:
                e.config(state=s)

        ttk.Radiobutton(dlg, text="Случайные seed-точки", variable=mode,
                        value="random", command=toggle).pack(anchor='w', padx=16, pady=(14, 2))
        ttk.Radiobutton(dlg, text="Ручные seed-точки", variable=mode,
                        value="manual", command=toggle).pack(anchor='w', padx=16)

        frame = ttk.LabelFrame(dlg, text="Индексы пикселей через запятую", padding=8)
        frame.pack(fill='x', padx=14, pady=10)
        n_pix = self.data['h'] * self.data['w']
        ttk.Label(frame, text=f"Допустимо: 0 – {n_pix - 1}",
                  foreground=COL_MUTED).pack(anchor='w')

        entries = []
        for cn in CLASS_NAMES:
            row = ttk.Frame(frame)
            row.pack(fill='x', pady=1)
            ttk.Label(row, text=cn + ':', width=18).pack(side='left')
            e = ttk.Entry(row, width=32, state='disabled')
            e.pack(side='left')
            entries.append(e)

        def ok():
            if mode.get() == 'random':
                result[0] = 'random'
                dlg.destroy()
                return
            seeds = []
            for i, e in enumerate(entries):
                txt = e.get().strip()
                if not txt:
                    seeds.append([np.random.randint(n_pix)])
                    continue
                try:
                    idxs = [int(x) for x in txt.split(',') if x.strip()]
                    for idx in idxs:
                        if not (0 <= idx < n_pix):
                            messagebox.showerror("Ошибка", f"Индекс {idx} вне диапазона")
                            return
                    seeds.append(idxs)
                except ValueError:
                    messagebox.showerror("Ошибка", f"Неверный формат: {CLASS_NAMES[i]}")
                    return
            result[0] = seeds
            dlg.destroy()

        btns = ttk.Frame(dlg, style='Panel.TFrame')
        btns.pack(fill='x', padx=14, pady=(0, 14))
        ttk.Button(btns, text="Отмена", style='Plain.TButton',
                   command=dlg.destroy).pack(side='right', padx=(6, 0))
        ttk.Button(btns, text="Запустить", style='Accent.TButton',
                   command=ok).pack(side='right')
        dlg.wait_window()
        return result[0]


if __name__ == '__main__':
    if not os.environ.get('DISPLAY'):
        os.environ['DISPLAY'] = 'host.docker.internal:0'
    App().mainloop()
