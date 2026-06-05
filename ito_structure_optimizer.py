"""
ITO废料闪蒸焦耳热铟回收 — 结构创新版AI优化可视化系统
========================================================
匹配专利技术交底书（结构创新版）

核心可视化：
  ① 梯度浓度反应床剖面结构
  ② 同轴一体化反应-冷凝腔（横截面 + 纵截面）
  ③ 梯度温控冷凝管温度分布
  ④ 两段式脉冲温度曲线
  ⑤ InCl₃/SnCl₄/SnCl₂ 蒸气压曲线
  ⑥ 整体工艺流程图
  ⑦ XGBoost 参数预测
  ⑧ 贝叶斯自学习优化
  ⑨ 批次数据管理与反馈学习

依赖安装：pip install xgboost scikit-learn gradio numpy pandas scipy matplotlib
启动方式：python ito_structure_optimizer.py
"""

import json, os, datetime, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm

# ───────────────────── 全局配置 ─────────────────────
DATA_DIR   = Path(__file__).parent / "ito_ai_data"
MODEL_PATH = DATA_DIR / "batch_history.json"
DATA_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
                                   "Noto Sans CJK SC", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

# 颜色方案
C = {
    "red":    "#E74C3C",
    "orange": "#F39C12",
    "yellow": "#F1C40F",
    "green":  "#27AE60",
    "blue":   "#2980B9",
    "purple": "#8E44AD",
    "gray":   "#7F8C8D",
    "dark":   "#2C3E50",
    "white":  "#ECF0F1",
    "bg":     "#FAFAFA",
}

# ═══════════════════════════════════════════════════════
#  1. 热力学模型
# ═══════════════════════════════════════════════════════
INCL3_SUBLIMATION  = {"A": 26.5, "B": 16800, "T_range_K": (450, 750)}
SNCL4_VAPORIZATION = {"A": 22.8, "B": 5200,  "T_range_K": (200, 500)}
SNCL2_VAPORIZATION = {"A": 25.0, "B": 13500, "T_range_K": (500, 1000)}


def _cond_temp(params, P_Pa):
    A, B = params["A"], params["B"]
    P = max(P_Pa, 1.0)
    T = B / (A - np.log(P))
    lo, hi = params["T_range_K"]
    return np.clip(T, lo, hi)


def calc_condensation_temps(pressure_Pa):
    T_incl3 = _cond_temp(INCL3_SUBLIMATION, pressure_Pa) - 273.15
    T_sncl4 = _cond_temp(SNCL4_VAPORIZATION, pressure_Pa) - 273.15
    T_sncl2 = _cond_temp(SNCL2_VAPORIZATION, pressure_Pa) - 273.15
    return {
        "InCl3 凝结温度(C)": round(T_incl3, 1),
        "SnCl4 凝结温度(C)": round(T_sncl4, 1),
        "SnCl2 凝结温度(C)": round(T_sncl2, 1),
        "InCl3-SnCl4 温差(C)": round(T_incl3 - T_sncl4, 1),
        "推荐-固体颗粒拦截段(C)": round(T_incl3 + 120, 1),
        "推荐-InCl3主沉积段(C)": round(T_incl3 - 30, 1),
        "推荐-In金属沉积段(C)": round((T_sncl4 + T_incl3) / 2, 1),
        "推荐-副产物收集段(C)": round(max(T_sncl4 - 20, -10), 1),
        "推荐-尾端冷阱(C)": -5.0,
    }


def vapor_pressure_curve(params, T_min_C, T_max_C, n=200):
    T_C = np.linspace(T_min_C, T_max_C, n)
    T_K = T_C + 273.15
    A, B = params["A"], params["B"]
    lnP = A - B / T_K
    P_Pa = np.exp(lnP)
    return T_C, P_Pa


# ═══════════════════════════════════════════════════════
#  2. 合成数据 + ML 模型
# ═══════════════════════════════════════════════════════
def _generate_synthetic(n=200, seed=42):
    rng = np.random.RandomState(seed)
    in2o3 = rng.uniform(55, 95, n)
    sno2  = np.clip(95 - in2o3 + rng.normal(0, 3, n), 2, 35)
    imp   = 100 - in2o3 - sno2
    d50   = rng.uniform(3, 50, n)
    c_ratio  = np.clip(0.15 + 0.002*in2o3 + rng.normal(0, 0.03, n), 0.05, 0.40)
    nh_ratio = np.clip(0.02 + 0.001*sno2 + 0.0005*in2o3 + rng.normal(0, 0.005, n), 0.01, 0.15)
    t1 = np.clip(350 + 0.5*sno2 - 0.3*d50 + rng.normal(0, 20, n), 200, 500)
    d1 = np.clip(3 + 0.05*sno2 + 0.02*d50 + rng.normal(0, 1, n), 1, 30)
    i1 = np.clip(8 + 0.1*in2o3 + rng.normal(0, 2, n), 5, 25)
    t2 = np.clip(700 + 1.5*in2o3 - 0.8*sno2 + rng.normal(0, 30, n), 500, 1200)
    d2 = np.clip(2 + 0.01*in2o3 - 0.005*sno2 + rng.normal(0, 0.5, n), 0.1, 30)
    i2 = np.clip(50 + 0.3*in2o3 + rng.normal(0, 5, n), 30, 100)
    pr = np.clip(30 + 2*sno2 + rng.normal(0, 10, n), 10, 500)
    ci = np.clip(260 + 0.3*pr + rng.normal(0, 10, n), 200, 350)
    rec = np.clip(85 + 0.1*in2o3 - 0.15*sno2 + 0.05*(nh_ratio*100) + rng.normal(0, 3, n), 50, 99)
    pur = np.clip(95 - 0.2*sno2 + 0.05*in2o3 + rng.normal(0, 2, n), 70, 99.5)
    return pd.DataFrame({
        "in2o3_pct": np.round(in2o3, 2), "sno2_pct": np.round(sno2, 2),
        "impurity_pct": np.round(imp, 2), "d50_um": np.round(d50, 1),
        "carbon_ratio": np.round(c_ratio, 4), "nh4cl_ratio": np.round(nh_ratio, 4),
        "current_s1": np.round(i1, 1), "time_s1": np.round(d1, 2),
        "temp_s1": np.round(t1, 1), "current_s2": np.round(i2, 1),
        "time_s2": np.round(d2, 2), "temp_s2": np.round(t2, 1),
        "pressure_Pa": np.round(pr, 1), "cond_incl3_C": np.round(ci, 1),
        "recovery_pct": np.round(rec, 2), "purity_pct": np.round(pur, 2),
    })


INPUT_COLS  = ["in2o3_pct", "sno2_pct", "impurity_pct", "d50_um"]
OUTPUT_COLS = [
    "carbon_ratio", "nh4cl_ratio", "current_s1", "time_s1", "temp_s1",
    "current_s2", "time_s2", "temp_s2", "pressure_Pa", "cond_incl3_C",
    "recovery_pct", "purity_pct",
]
OUTPUT_BOUNDS = {
    "carbon_ratio": (0.05, 0.40), "nh4cl_ratio":  (0.01, 0.15),
    "current_s1":   (5, 25),      "time_s1":      (1, 30),
    "temp_s1":      (200, 500),   "current_s2":   (30, 100),
    "time_s2":      (0.1, 30),    "temp_s2":      (500, 1200),
    "pressure_Pa":  (10, 500),    "cond_incl3_C": (200, 350),
    "recovery_pct": (50, 99),     "purity_pct":   (70, 99.5),
}


def _load_history():
    if MODEL_PATH.exists():
        return json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    return []

def _save_history(records):
    MODEL_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

def add_batch_record(record):
    records = _load_history()
    record["timestamp"] = datetime.datetime.now().isoformat()
    record["id"] = hashlib.md5(record["timestamp"].encode()).hexdigest()[:8]
    records.append(record)
    _save_history(records)
    return record["id"]

def get_all_records():
    records = _load_history()
    return pd.DataFrame(records) if records else pd.DataFrame()


from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


class ITOParameterPredictor:
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        base = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42, verbosity=0,
        )
        self.model = MultiOutputRegressor(base)
        self._trained = False

    def _build_data(self):
        syn = _generate_synthetic(200)
        real = get_all_records()
        if not real.empty:
            needed = INPUT_COLS + OUTPUT_COLS
            avail = [c for c in needed if c in real.columns]
            if len(avail) == len(needed):
                rc = real[needed].dropna()
                rw = pd.concat([rc] * 3, ignore_index=True)
                return pd.concat([syn, rw], ignore_index=True)
        return syn

    def train(self):
        df = self._build_data()
        X, y = df[INPUT_COLS], df[OUTPUT_COLS]
        self.scaler_X.fit(X); self.scaler_y.fit(y)
        self.model.fit(self.scaler_X.transform(X), self.scaler_y.transform(y))
        self._trained = True

    def predict(self, in2o3, sno2, impurity, d50):
        if not self._trained:
            self.train()
        X = pd.DataFrame([{"in2o3_pct": in2o3, "sno2_pct": sno2,
                           "impurity_pct": impurity, "d50_um": d50}])
        y = self.scaler_y.inverse_transform(
            self.model.predict(self.scaler_X.transform(X)))
        result = {}
        for i, c in enumerate(OUTPUT_COLS):
            v = float(y[0, i])
            lo, hi = OUTPUT_BOUNDS[c]
            result[c] = round(float(np.clip(v, lo, hi)), 2)
        return result


predictor = ITOParameterPredictor()


# ═══════════════════════════════════════════════════════
#  3. 贝叶斯优化
# ═══════════════════════════════════════════════════════
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel


class BayesianBatchOptimizer:
    PARAM_SPACE = {
        "carbon_ratio": (0.05, 0.40), "nh4cl_ratio":  (0.01, 0.15),
        "temp_s1": (200, 500), "time_s1": (1, 30),
        "temp_s2": (500, 1200), "time_s2": (0.1, 30),
        "pressure_Pa": (10, 500),
    }
    PARAM_NAMES = list(PARAM_SPACE.keys())

    def __init__(self):
        self.gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0)*Matern(nu=2.5),
            alpha=1e-4, n_restarts_optimizer=5, random_state=42)

    def _observed(self):
        df = get_all_records()
        if df.empty or len(df) < 3:
            return np.empty((0, len(self.PARAM_NAMES))), np.array([])
        req = self.PARAM_NAMES + ["recovery_pct"]
        if not all(c in df.columns for c in req):
            return np.empty((0, len(self.PARAM_NAMES))), np.array([])
        X, y = df[self.PARAM_NAMES].values, df["recovery_pct"].values
        m = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        return X[m], y[m]

    def suggest_next(self):
        X, y = self._observed()
        if len(X) < 5:
            rng = np.random.RandomState(len(X)*7+13)
            s = {n: round(rng.uniform(lo, hi), 2)
                 for n, (lo, hi) in self.PARAM_SPACE.items()}
            s["_strategy"] = "Latin Hypercube Exploration (suggest 5-10 experiments first)"
            return s
        lo_arr = np.array([v[0] for v in self.PARAM_SPACE.values()])
        hi_arr = np.array([v[1] for v in self.PARAM_SPACE.values()])
        Xn = (X - lo_arr) / (hi_arr - lo_arr)
        self.gp.fit(Xn, y)
        best_y = y.max()
        best_c, best_ei = None, -1
        rng = np.random.RandomState(42)
        for _ in range(5000):
            c = np.array([rng.uniform(lo, hi) for _, (lo, hi) in self.PARAM_SPACE.items()])
            cn = (c - lo_arr) / (hi_arr - lo_arr)
            mu, sigma = self.gp.predict(cn.reshape(1, -1), return_std=True)
            sigma = max(float(sigma), 1e-8)
            z = (float(mu) - best_y) / sigma
            ei = (float(mu) - best_y) * 0.5*(1+np.math.erf(z/np.sqrt(2))) + \
                 sigma * np.exp(-0.5*z*z)/np.sqrt(2*np.pi)
            if ei > best_ei:
                best_ei, best_c = ei, c
        s = {}
        for i, n in enumerate(self.PARAM_NAMES):
            lo, hi = self.PARAM_SPACE[n]
            s[n] = round(float(np.clip(best_c[i], lo, hi)), 2)
        s["_strategy"] = f"GP-Expected Improvement ({len(X)} data points)"
        return s


bo_optimizer = BayesianBatchOptimizer()


# ═══════════════════════════════════════════════════════
#  4. 可视化函数
# ═══════════════════════════════════════════════════════

def _fig_to_numpy(fig):
    fig.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img


# ── 4.1 梯度浓度反应床剖面 ──
def plot_gradient_bed():
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 13)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Gradient Concentration Reaction Bed", fontsize=15, fontweight="bold", pad=15)

    # 电极
    for y, label, color in [(0.4, "Bottom Electrode (-)", C["gray"]),
                             (11.4, "Top Electrode (+)", C["dark"])]:
        rect = FancyBboxPatch((2.5, y), 7, 0.6, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor=C["dark"], lw=2)
        ax.add_patch(rect)
        ax.text(6, y+0.3, label, ha="center", va="center", fontsize=10,
                fontweight="bold", color="white")

    # 三层反应床
    layers = [
        (1.2, 3.5, "#E74C3C", "Bottom Layer: Pre-chlorination\nNH4Cl 10% + Carbon 10%",
         "NH4Cl decomposes at 300C\nHCl reacts with In2O3"),
        (4.9, 3.0, "#F39C12", "Middle Layer: Transition\nNH4Cl 4% + Carbon 20%",
         "Buffer zone 400-800C\nChlorination -> reduction"),
        (8.1, 3.1, "#F1C40F", "Top Layer: Flash Volatilization\nNH4Cl 2% + Carbon 28%",
         "Highest carbon content\n800-1800C rapid reduction"),
    ]
    for y, h, color, label, _ in layers:
        rect = FancyBboxPatch((2.5, y), 7, h, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor=C["dark"], lw=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(6, y+h/2, label, ha="center", va="center", fontsize=9, fontweight="bold")

    # 右侧注释
    for y, h, _, _, desc in layers:
        ax.annotate(desc, xy=(9.7, y+h/2), xytext=(10.5, y+h/2),
                    fontsize=7.5, va="center",
                    arrowprops=dict(arrowstyle="->", color=C["dark"], lw=1.2))

    # 温度梯度色条
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(gradient, aspect="auto", cmap="coolwarm",
              extent=[1.0, 2.0, 1.2, 11.2], origin="lower")
    ax.text(0.7, 3, "Low T", fontsize=8, ha="center", va="center", color=C["blue"], fontweight="bold")
    ax.text(0.7, 9.5, "High T", fontsize=8, ha="center", va="center", color=C["red"], fontweight="bold")
    ax.text(0.7, 6, "~", fontsize=14, ha="center", va="center", color=C["gray"])

    # 电流方向箭头
    ax.annotate("", xy=(1.6, 10.5), xytext=(1.6, 2.0),
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=3))
    ax.text(1.6, 6.5, "Current", ha="center", va="center", fontsize=9,
            color=C["blue"], fontweight="bold", rotation=90)

    return _fig_to_numpy(fig)


# ── 4.2 同轴一体化反应-冷凝腔（横截面） ──
def plot_coaxial_cross():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-7, 7); ax.set_ylim(-7, 7)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Coaxial Reactor-Condenser Cross Section", fontsize=14, fontweight="bold", pad=12)

    theta = np.linspace(0, 2*np.pi, 100)
    # 外壳
    ax.fill(6*np.cos(theta), 6*np.sin(theta), color="#E0E0E0", edgecolor=C["dark"], lw=2.5)
    ax.plot(6*np.cos(theta), 6*np.sin(theta), color=C["dark"], lw=2.5)
    # 冷凝壁面
    ax.fill(4.8*np.cos(theta), 4.8*np.sin(theta), color=C["green"], alpha=0.25,
            edgecolor=C["green"], lw=2)
    # 蒸气通道
    ax.fill(3.5*np.cos(theta), 3.5*np.sin(theta), color="#FFF3E0", edgecolor=C["orange"], lw=2)
    # 反应床
    ax.fill(2.2*np.cos(theta), 2.2*np.sin(theta), color=C["red"], alpha=0.65,
            edgecolor=C["dark"], lw=2)
    # 中心电极
    ax.fill(0.8*np.cos(theta), 0.8*np.sin(theta), color=C["gray"],
            edgecolor=C["dark"], lw=1.5)

    # 标注（左侧）
    labels = [
        (0, 0, "Electrode", "white", 8),
        (0, -1.5, "Reaction Bed\n(Gradient Layers)", "white", 7.5),
        (0, -2.8, "Vapor Channel\n(10-20mm)", C["dark"], 7),
        (0, -4.2, "Condenser Wall\n(Gradient T Control)", C["dark"], 7),
        (0, -5.5, "Vacuum Shell", C["dark"], 7),
    ]
    for x, y, t, c, s in labels:
        ax.text(x, y, t, ha="center", va="center", fontsize=s, fontweight="bold", color=c)

    # 径向扩散箭头 (多条)
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = np.radians(angle)
        ax.annotate("",
            xy=(4.5*np.cos(rad), 4.5*np.sin(rad)),
            xytext=(2.5*np.cos(rad), 2.5*np.sin(rad)),
            arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.5, alpha=0.6))

    ax.text(0, 5.8, "InCl3 radial diffusion -> condensation", fontsize=8,
            ha="center", color=C["blue"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 冷凝壁面温控段标注
    for i, angle in enumerate([30, 90, 150, 210, 270, 330]):
        rad = np.radians(angle)
        ax.plot(4.8*np.cos(rad), 4.8*np.sin(rad), 'o', color=C["green"], markersize=6)
    ax.text(-6.5, 0, "8-zone PID\nT control", fontsize=8, ha="center",
            color=C["green"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["green"], alpha=0.15))

    return _fig_to_numpy(fig)


# ── 4.3 同轴一体化反应-冷凝腔（纵截面） ──
def plot_coaxial_long():
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(-5, 5); ax.set_ylim(-1, 14)
    ax.set_aspect("auto"); ax.axis("off")
    ax.set_title("Coaxial Reactor-Condenser Longitudinal Section", fontsize=14, fontweight="bold", pad=12)

    # 外壳
    ax.add_patch(FancyBboxPatch((-3.8, 0), 7.6, 12, boxstyle="round,pad=0.15",
                                facecolor="#F0F0F0", edgecolor=C["dark"], lw=2.5))
    # 冷凝壁面
    ax.add_patch(FancyBboxPatch((-3.0, 0.5), 6.0, 11, boxstyle="round,pad=0.05",
                                facecolor=C["green"], alpha=0.15, edgecolor=C["green"], lw=1.5))
    # 环形蒸气通道
    ax.add_patch(FancyBboxPatch((-2.2, 1.0), 4.4, 10, boxstyle="round,pad=0.05",
                                facecolor="#FFF3E0", edgecolor=C["orange"], lw=1))

    # 梯度反应床（三层）
    bed_layers = [
        (1.2, 2.8, C["red"],    "Bottom\nPre-chlorination\nNH4Cl 10%"),
        (4.2, 2.4, C["orange"], "Middle\nTransition\nNH4Cl 4%"),
        (6.8, 2.6, C["yellow"], "Top\nFlash Volatilization\nCarbon 28%"),
    ]
    for y, h, color, label in bed_layers:
        ax.add_patch(FancyBboxPatch((-1.3, y), 2.6, h, boxstyle="round,pad=0.03",
                                    facecolor=color, alpha=0.75, edgecolor=C["dark"], lw=1))
        ax.text(0, y+h/2, label, ha="center", va="center", fontsize=8, fontweight="bold")

    # 层间碳纸
    for y in [4.1, 6.7]:
        ax.plot([-1.3, 1.3], [y, y], color=C["dark"], lw=1.5, ls="--")
        ax.text(1.5, y, "Carbon paper", fontsize=6, va="center", color=C["gray"])

    # 电极
    for y, label in [(0.5, "Bottom\nElectrode"), (9.8, "Top\nElectrode")]:
        ax.add_patch(FancyBboxPatch((-1.3, y), 2.6, 0.7, boxstyle="round,pad=0.03",
                                    facecolor=C["gray"], edgecolor=C["dark"], lw=1.5))
        ax.text(0, y+0.35, label, ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # 径向扩散箭头
    for y in [2.5, 5.4, 8.0]:
        ax.annotate("", xy=(2.5, y), xytext=(1.5, y),
                    arrowprops=dict(arrowstyle="->", color=C["blue"], lw=2))

    ax.text(3.2, 5.5, "InCl3 vapor\nradial diffusion", fontsize=8, color=C["blue"],
            fontweight="bold", ha="center")

    # 冷凝壁面各温控段（右侧标注）
    cond_zones = [
        (10.5, "550C  Solid particle trap"),
        (9.0,  "400C"),
        (7.5,  "280C  InCl3 deposit"),
        (6.0,  "180C  In metal deposit"),
        (4.5,  "100C"),
        (3.0,  "50C   SnCl4 collection"),
        (1.5,  "-5C   Cold trap"),
    ]
    for y, label in cond_zones:
        ax.plot([3.2], [y], 's', color=C["green"], markersize=5)
        ax.text(3.5, y, label, fontsize=7, va="center", color=C["green"])

    # 温度梯度箭头
    ax.annotate("", xy=(3.0, 1.5), xytext=(3.0, 10.5),
                arrowprops=dict(arrowstyle="<->", color=C["red"], lw=1.5))
    ax.text(4.5, 6, "T gradient\nalong axis", fontsize=7, ha="center",
            color=C["red"], fontweight="bold")

    # 左侧标注
    ax.text(-4.5, 6, "Condenser\nWall", fontsize=8, ha="center", color=C["green"], fontweight="bold")
    ax.text(4.5, 11.5, "Vacuum\nShell", fontsize=8, color=C["dark"], fontweight="bold")

    # 热电偶
    for y in [2.6, 5.4, 8.1]:
        ax.plot([-0.1], [y], 'x', color=C["blue"], markersize=8, mew=2)
    ax.text(-2.0, 5.4, "Thermocouple\narray", fontsize=7, ha="center", color=C["blue"])

    return _fig_to_numpy(fig)


# ── 4.4 梯度温控冷凝管温度分布 ──
def plot_condensation_gradient(pressure):
    fig, ax = plt.subplots(figsize=(10, 5))

    cond = calc_condensation_temps(pressure)
    T_incl3 = cond["InCl3 凝结温度(C)"]
    T_sncl4 = cond["SnCl4 凝结温度(C)"]

    # 梯度温度曲线 (指数衰减)
    x = np.linspace(0, 100, 500)
    T_hot = 550; T_cold = -10; alpha = 0.7
    T_profile = T_hot + (T_cold - T_hot) * (x / 100) ** alpha

    ax.plot(x, T_profile, color=C["red"], lw=2.5, label="Wall Temperature Profile")
    ax.fill_between(x, T_profile, T_cold, alpha=0.1, color=C["red"])

    # InCl3 和 SnCl4 凝结温度线
    ax.axhline(T_incl3, color=C["green"], ls="--", lw=2, label=f"InCl3 condensation: {T_incl3:.0f}C")
    ax.axhline(T_sncl4, color=C["blue"],  ls="--", lw=2, label=f"SnCl4 condensation: {T_sncl4:.0f}C")

    # 功能分区
    idx_incl3 = np.argmin(np.abs(T_profile - T_incl3))
    idx_sncl4 = np.argmin(np.abs(T_profile - T_sncl4))
    idx_in_metal = np.argmin(np.abs(T_profile - 157))

    zones = [
        (0, x[idx_incl3-30], "#FFE0B2", "Solid Particle\nTrap Zone"),
        (x[max(0,idx_incl3-30)], x[min(len(x)-1,idx_incl3+20)], "#C8E6C9", "InCl3\nDeposit Zone"),
        (x[idx_incl3+20], x[idx_in_metal+15], "#BBDEFB", "In Metal\nDeposit Zone"),
        (x[idx_in_metal+15], x[idx_sncl4+20], "#E1BEE7", "SnCl4\nCollection Zone"),
        (x[idx_sncl4+20], 100, "#E0E0E0", "Cold\nTrap Zone"),
    ]
    for x0, x1, color, label in zones:
        if x1 > x0:
            ax.axvspan(x0, x1, alpha=0.2, color=color)
            ax.text((x0+x1)/2, T_hot*0.85, label, ha="center", va="center",
                    fontsize=8, fontweight="bold")

    ax.set_xlabel("Position along condenser (%)", fontsize=11)
    ax.set_ylabel("Temperature (C)", fontsize=11)
    ax.set_title(f"Gradient Condenser Wall Temperature Profile (P = {pressure} Pa)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 100); ax.set_ylim(-30, 600)
    ax.grid(True, alpha=0.3)

    return _fig_to_numpy(fig)


# ── 4.5 两段式脉冲温度曲线 ──
def plot_pulse_curve(res):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    t1, d1 = res["temp_s1"], res["time_s1"]
    t2, d2 = res["temp_s2"], res["time_s2"]

    # 构建温度曲线
    time_pts = [0, d1*0.15, d1*0.7, d1, d1+0.5, d1+0.5+d2*0.2,
                d1+0.5+d2*0.8, d1+0.5+d2, d1+0.5+d2+1, d1+d2+3]
    temp_pts = [25, t1*0.4, t1*0.85, t1, t1*0.95, t2*0.7,
                t2*0.95, t2, t2*0.4, 25]

    ax.plot(time_pts, temp_pts, color=C["red"], lw=2.5, zorder=5)
    ax.fill_between(time_pts, temp_pts, alpha=0.12, color=C["red"])

    # 反应阶段背景
    ax.axvspan(0, d1, alpha=0.08, color=C["blue"])
    ax.axvspan(d1+0.5, d1+0.5+d2, alpha=0.08, color=C["orange"])

    # 阶段标注
    ax.annotate(f"Stage 1: Pre-chlorination\n{t1:.0f}C / {d1:.1f}s\n(5-25 A/cm2)",
                xy=(d1/2, t1), xytext=(d1/2, t1+200), fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.5),
                bbox=dict(boxstyle="round,pad=0.4", facecolor=C["blue"], alpha=0.15))

    ax.annotate(f"Stage 2: Flash Volatilization\n{t2:.0f}C / {d2:.1f}s\n(30-120 A/cm2)",
                xy=(d1+0.5+d2/2, t2), xytext=(d1+0.5+d2/2, t2+200),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.5),
                bbox=dict(boxstyle="round,pad=0.4", facecolor=C["orange"], alpha=0.15))

    # 间隔标注
    ax.annotate("Interval\n0.1-2s", xy=(d1+0.25, (t1+t2*0.7)/2),
                fontsize=8, ha="center", color=C["purple"], fontweight="bold")

    # 温度区间标注
    ax.axhspan(200, 500, alpha=0.06, color=C["blue"], label="Pre-chlorination window")
    ax.axhspan(800, 1800, alpha=0.06, color=C["orange"], label="Flash volatilization window")

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Reaction Bed Temperature (C)", fontsize=11)
    ax.set_title("Two-Stage Pulse Heating Curve (synergy with gradient bed)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, max(t2*1.3, 1400))
    ax.grid(True, alpha=0.3)

    return _fig_to_numpy(fig)


# ── 4.6 蒸气压曲线 ──
def plot_vapor_pressure():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    T_in, P_in = vapor_pressure_curve(INCL3_SUBLIMATION, 300, 700)
    T_sn, P_sn = vapor_pressure_curve(SNCL4_VAPORIZATION, 0, 200)
    T_sn2, P_sn2 = vapor_pressure_curve(SNCL2_VAPORIZATION, 400, 800)

    ax.semilogy(T_in, P_in, color=C["green"], lw=2.5, label="InCl3 (sublimation)")
    ax.semilogy(T_sn, P_sn, color=C["blue"],  lw=2.5, label="SnCl4 (vaporization)")
    ax.semilogy(T_sn2, P_sn2, color=C["purple"], lw=2.5, label="SnCl2 (vaporization)")

    for P, label, color in [(100, "100 Pa", C["red"]),
                             (500, "500 Pa", C["orange"]),
                             (101325, "1 atm", C["gray"])]:
        ax.axhline(P, color=color, ls=":", lw=1.2, alpha=0.7)
        ax.text(50, P*1.4, label, fontsize=8, color=color)

    # 标注温度差区间
    ax.annotate("InCl3-SnCl4 separation\nwindow (>200C at 100Pa)",
                xy=(250, 300), xytext=(400, 500),
                fontsize=9, fontweight="bold", color=C["green"],
                arrowprops=dict(arrowstyle="->", color=C["green"]),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C["green"], alpha=0.1))

    ax.set_xlabel("Temperature (C)", fontsize=11)
    ax.set_ylabel("Vapor Pressure (Pa)", fontsize=11)
    ax.set_title("InCl3 / SnCl4 / SnCl2 Vapor Pressure - Temperature", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0, 800)

    return _fig_to_numpy(fig)


# ── 4.7 整体工艺流程图 ──
def plot_process_flow():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Process Flow: Gradient Bed + Coaxial Condenser FJH System", fontsize=14, fontweight="bold", pad=15)

    # 流程步骤框
    boxes = [
        (0.5, 7, 2.2, 1.2, C["blue"],   "S1: Material Prep\n& Gradient\nBatching"),
        (3.5, 7, 2.2, 1.2, C["red"],    "S2: Build Gradient\nConcentration\nReaction Bed"),
        (6.5, 7, 2.2, 1.2, C["orange"], "S3: Load into\nCoaxial Reactor\n& Vacuum"),
        (9.5, 7, 2.2, 1.2, C["purple"], "S4: Two-Stage\nPulse FJH\nReaction"),
        (12.0, 5.2, 1.5, 1.0, C["green"], "S5: Gradient\nCondenser\nCollection"),
        (9.5, 3.2, 2.2, 1.0, C["green"], "S6: In Enrichment\nProduct"),
        (6.5, 3.2, 2.2, 1.0, C["blue"],  "S7: Refining\n(Electrolysis)"),
        (3.5, 3.2, 2.2, 1.0, C["gray"],  "S8: Carbon\nRecovery\n& Recycling"),
    ]

    for x, y, w, h, color, label in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                               facecolor=color, alpha=0.2, edgecolor=color, lw=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold")

    # 流程箭头
    arrow_style = dict(arrowstyle="-|>", color=C["dark"], lw=1.8)
    arrows = [
        ((2.8, 7.6), (3.4, 7.6)),
        ((5.8, 7.6), (6.4, 7.6)),
        ((8.8, 7.6), (9.4, 7.6)),
        ((10.6, 7.0), (12.7, 6.3)),
        ((12.7, 5.2), (10.6, 4.3)),
        ((9.5, 3.7), (8.8, 3.7)),
        ((6.5, 3.7), (5.8, 3.7)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=arrow_style)

    # 反馈循环箭头
    ax.annotate("Bayesian\nFeedback", xy=(4.6, 7.0), xytext=(4.6, 4.3),
                fontsize=8, ha="center", color=C["purple"], fontweight="bold",
                arrowprops=dict(arrowstyle="<->", color=C["purple"], lw=2, ls="--"))

    # 碳循环箭头
    ax.annotate("Carbon\nRecycle", xy=(3.5, 7.0), xytext=(3.5, 4.3),
                fontsize=8, ha="center", color=C["gray"], fontweight="bold",
                arrowprops=dict(arrowstyle="<->", color=C["gray"], lw=2, ls="--"))

    # ML 预测框
    ml_box = FancyBboxPatch((0.3, 4.5), 2.5, 1.5, boxstyle="round,pad=0.15",
                             facecolor=C["purple"], alpha=0.15, edgecolor=C["purple"], lw=1.5)
    ax.add_patch(ml_box)
    ax.text(1.55, 5.25, "XGBoost Parameter\nPrediction +\nBayesian Optimization", ha="center",
            va="center", fontsize=8, fontweight="bold", color=C["purple"])

    ax.annotate("", xy=(1.55, 7.0), xytext=(1.55, 6.1),
                arrowprops=dict(arrowstyle="<->", color=C["purple"], lw=1.5, ls="--"))

    # 标注梯度反应床结构 (小示意图)
    mini_box = FancyBboxPatch((11.8, 7.5), 1.8, 1.2, boxstyle="round,pad=0.1",
                               facecolor="white", edgecolor=C["dark"], lw=1)
    ax.add_patch(mini_box)
    # 三层小色块
    for i, (color, label) in enumerate([(C["red"], ""), (C["orange"], ""), (C["yellow"], "")]):
        ax.add_patch(FancyBboxPatch((12.0, 7.6+i*0.35), 1.4, 0.3,
                                    facecolor=color, alpha=0.6, edgecolor="none"))
    ax.text(12.7, 8.85, "Bed", fontsize=7, ha="center", fontweight="bold")

    return _fig_to_numpy(fig)


# ── 4.8 贝叶斯优化流程图 ──
def plot_bayesian_flow():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Self-Learning Optimization Loop", fontsize=14, fontweight="bold", pad=12)

    # 流程框
    flow = [
        (0.3, 2.5, 2.0, 1.5, C["blue"], "ITO Feedstock\nComposition\n(In2O3, SnO2,\nimpurity, D50)"),
        (3.0, 2.5, 2.0, 1.5, C["purple"], "XGBoost\nParameter\nPrediction\n(12 outputs)"),
        (5.7, 2.5, 2.0, 1.5, C["orange"], "FJH Experiment\n(Two-stage pulse\n+ Gradient bed\n+ Coaxial condenser)"),
        (8.4, 2.5, 2.0, 1.5, C["green"], "Product Analysis\n(Recovery %\nPurity %)"),
        (5.7, 0.3, 2.0, 1.2, C["red"], "Bayesian\nOptimization\n(GP + EI)"),
    ]

    for x, y, w, h, color, label in flow:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                               facecolor=color, alpha=0.15, edgecolor=color, lw=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold")

    # 箭头
    style = dict(arrowstyle="-|>", color=C["dark"], lw=1.8)
    for s, e in [((2.4, 3.25), (2.9, 3.25)),
                 ((5.1, 3.25), (5.6, 3.25)),
                 ((7.8, 3.25), (8.3, 3.25))]:
        ax.annotate("", xy=e, xytext=s, arrowprops=style)

    # 贝叶斯反馈箭头
    ax.annotate("New experiment\nsuggestion", xy=(6.7, 1.55), xytext=(6.7, 2.4),
                fontsize=8, ha="center", color=C["red"], fontweight="bold",
                arrowprops=dict(arrowstyle="<-", color=C["red"], lw=2))
    ax.annotate("", xy=(5.6, 3.0), xytext=(5.0, 1.6),
                arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=1.5, ls="--",
                                connectionstyle="arc3,rad=-0.3"))

    # 数据反馈
    ax.annotate("Batch data\nfeedback", xy=(4.0, 2.5), xytext=(9.4, 2.0),
                fontsize=8, ha="center", color=C["green"], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=1.5, ls="--",
                                connectionstyle="arc3,rad=-0.3"))

    # 注释
    ax.text(10.8, 4.5, "Loop continues:\n-> Model improves\n-> Recovery increases\n-> Converges in\n   10-15 experiments",
            fontsize=8.5, va="top", ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C["purple"], alpha=0.1))

    return _fig_to_numpy(fig)


# ═══════════════════════════════════════════════════════
#  5. Gradio 界面
# ═══════════════════════════════════════════════════════
def create_ui():
    import gradio as gr

    predictor.train()

    # ── Tab 1: 参数预测 ──
    def predict_params(in2o3, sno2, impurity, d50):
        if abs(in2o3 + sno2 + impurity - 100) > 0.5:
            return "In2O3 + SnO2 + impurity should be ~100%", None, None
        res = predictor.predict(in2o3, sno2, impurity, d50)
        cond = calc_condensation_temps(res["pressure_Pa"])
        lines = [
            "=" * 55,
            "  AI Recommended Process Parameters",
            "=" * 55, "",
            "-- Gradient Bed Batching --",
            f"  Bottom layer: NH4Cl 10% + Carbon {res['carbon_ratio']*100*0.5:.1f}%",
            f"  Middle layer: NH4Cl 4%  + Carbon {res['carbon_ratio']*100*0.75:.1f}%",
            f"  Top layer:    NH4Cl 2%  + Carbon {res['carbon_ratio']*100:.1f}%", "",
            "-- Stage 1: Pre-chlorination --",
            f"  Current density:  {res['current_s1']:.1f} A/cm2",
            f"  Duration:         {res['time_s1']:.1f} s",
            f"  Target temp:      {res['temp_s1']:.0f} C", "",
            "-- Stage 2: Flash Volatilization --",
            f"  Current density:  {res['current_s2']:.1f} A/cm2",
            f"  Duration:         {res['time_s2']:.1f} s",
            f"  Target temp:      {res['temp_s2']:.0f} C", "",
            "-- Operating Pressure --",
            f"  Chamber pressure: {res['pressure_Pa']:.0f} Pa", "",
            "=" * 55,
            "  Condensation Zone Design (Gradient)",
            "=" * 55, "",
        ]
        for k, v in cond.items():
            lines.append(f"  {k}: {v} C")
        lines += ["", "=" * 55,
                   "  Predicted Results (need experimental verification)",
                   "=" * 55,
                   f"  InCl3 recovery:   ~{res['recovery_pct']:.1f}%",
                   f"  InCl3 purity:     ~{res['purity_pct']:.1f}%"]
        pulse_fig = plot_pulse_curve(res)
        return "\n".join(lines), pulse_fig, res

    # ── Tab 2: 梯度反应床 ──
    def show_bed():
        return plot_gradient_bed()

    # ── Tab 3: 同轴反应-冷凝腔 ──
    def show_coaxial_cross():
        return plot_coaxial_cross()

    def show_coaxial_long():
        return plot_coaxial_long()

    # ── Tab 4: 脉冲曲线 ──
    def show_pulse(t1, d1, t2, d2):
        res = {"temp_s1": t1, "time_s1": d1, "temp_s2": t2, "time_s2": d2}
        return plot_pulse_curve(res)

    # ── Tab 5: 冷凝系统 ──
    def show_condensation(pressure):
        cond = calc_condensation_temps(pressure)
        text = "\n".join(f"  {k}: {v} C" for k, v in cond.items())
        tube_fig = plot_condensation_gradient(pressure)
        return text, tube_fig

    # ── Tab 6: 蒸气压曲线 ──
    def show_vp():
        return plot_vapor_pressure()

    # ── Tab 7: 工艺流程 ──
    def show_flow():
        return plot_process_flow()

    # ── Tab 8: 贝叶斯优化流程 ──
    def show_bayesian():
        return plot_bayesian_flow()

    # ── Tab 9: 贝叶斯优化建议 ──
    def get_bo():
        s = bo_optimizer.suggest_next()
        st = s.pop("_strategy")
        lines = ["=" * 55,
                  "  Bayesian Optimization - Next Experiment Suggestion",
                  "=" * 55, "",
                  f"  Strategy: {st}", "",
                  "-- Batching --",
                  f"  Carbon ratio:    {s['carbon_ratio']*100:.1f} wt%",
                  f"  NH4Cl ratio:     {s['nh4cl_ratio']*100:.1f} wt%", "",
                  "-- Stage 1 --",
                  f"  Temp: {s['temp_s1']:.0f}C  Time: {s['time_s1']:.1f}s", "",
                  "-- Stage 2 --",
                  f"  Temp: {s['temp_s2']:.0f}C  Time: {s['time_s2']:.1f}s", "",
                  "-- Pressure --",
                  f"  Operating pressure: {s['pressure_Pa']:.0f} Pa", "",
                  "Tip: Submit results in 'Batch Records' tab after experiment."]
        return "\n".join(lines)

    # ── Tab 10: 批次记录 ──
    def submit_batch(in2o3, sno2, imp, d50, carbon, nh4cl,
                     cs1, ts1, tmp1, cs2, ts2, tmp2, pres, rec, pur):
        rec_d = {"in2o3_pct": in2o3, "sno2_pct": sno2, "impurity_pct": imp,
                 "d50_um": d50, "carbon_ratio": carbon, "nh4cl_ratio": nh4cl,
                 "current_s1": cs1, "time_s1": ts1, "temp_s1": tmp1,
                 "current_s2": cs2, "time_s2": ts2, "temp_s2": tmp2,
                 "pressure_Pa": pres, "cond_incl3_C": 0,
                 "recovery_pct": rec, "purity_pct": pur}
        bid = add_batch_record(rec_d)
        predictor.train()
        n = len(get_all_records())
        return f"Batch {bid} saved | Total: {n} | Model retrained"

    def view_history():
        df = get_all_records()
        if df.empty:
            return "No records yet"
        cols = [c for c in ["id","timestamp","in2o3_pct","sno2_pct",
                "carbon_ratio","nh4cl_ratio","temp_s1","temp_s2",
                "pressure_Pa","recovery_pct","purity_pct"] if c in df.columns]
        return df[cols].to_string(index=False)

    # ═══════ 构建界面 ═══════
    with gr.Blocks(title="ITO FJH Structure Innovation Optimizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ITO Waste FJH Indium Recovery - Structure Innovation Optimizer\n"
                     "Gradient Concentration Bed + Coaxial Integrated Reactor-Condenser")

        with gr.Tab("1 Parameter Prediction"):
            gr.Markdown("Input ITO composition -> AI recommended parameters + pulse curve")
            with gr.Row():
                in1 = gr.Number(label="In2O3 wt%", value=85)
                in2 = gr.Number(label="SnO2 wt%", value=10)
                in3 = gr.Number(label="Impurity wt%", value=5)
                in4 = gr.Number(label="D50 um", value=10)
            btn1 = gr.Button("Get Recommended Parameters", variant="primary")
            out_txt = gr.Textbox(label="Recommendations", lines=30)
            out_img = gr.Image(label="Pulse Temperature Curve", type="numpy")
            btn1.click(predict_params, inputs=[in1,in2,in3,in4],
                       outputs=[out_txt, out_img, gr.State()])

        with gr.Tab("2 Gradient Bed"):
            gr.Markdown("Gradient Concentration Reaction Bed cross-section structure")
            btn2 = gr.Button("Show Structure", variant="primary")
            out_bed = gr.Image(type="numpy", label="Gradient Reaction Bed")
            btn2.click(show_bed, outputs=out_bed)

        with gr.Tab("3 Coaxial Chamber"):
            gr.Markdown("Coaxial Integrated Reactor-Condenser (cross + longitudinal)")
            with gr.Row():
                btn3a = gr.Button("Cross Section", variant="primary")
                btn3b = gr.Button("Longitudinal Section", variant="primary")
            out_cx = gr.Image(type="numpy", label="Coaxial Chamber")
            btn3a.click(show_coaxial_cross, outputs=out_cx)
            btn3b.click(show_coaxial_long, outputs=out_cx)

        with gr.Tab("4 Pulse Curve"):
            gr.Markdown("Custom two-stage pulse heating curve")
            with gr.Row():
                pt1 = gr.Number(label="Stage 1 Temp (C)", value=400)
                pd1 = gr.Number(label="Stage 1 Time (s)", value=5)
                pt2 = gr.Number(label="Stage 2 Temp (C)", value=1600)
                pd2 = gr.Number(label="Stage 2 Time (s)", value=10)
            btn4 = gr.Button("Plot Curve", variant="primary")
            out_pulse = gr.Image(type="numpy", label="Pulse Curve")
            btn4.click(show_pulse, inputs=[pt1,pd1,pt2,pd2], outputs=out_pulse)

        with gr.Tab("5 Condenser"):
            gr.Markdown("Gradient temperature-controlled condenser wall distribution")
            p_in = gr.Number(label="Operating Pressure (Pa)", value=100)
            btn5 = gr.Button("Calculate Condensation Zones", variant="primary")
            out_ct = gr.Textbox(label="Zone Data", lines=10)
            out_ci = gr.Image(type="numpy", label="Temperature Profile")
            btn5.click(show_condensation, inputs=p_in, outputs=[out_ct, out_ci])

        with gr.Tab("6 Vapor Pressure"):
            gr.Markdown("InCl3 / SnCl4 / SnCl2 vapor pressure thermodynamics")
            btn6 = gr.Button("Show Vapor Pressure Curves", variant="primary")
            out_vp = gr.Image(type="numpy", label="Vapor Pressure")
            btn6.click(show_vp, outputs=out_vp)

        with gr.Tab("7 Process Flow"):
            gr.Markdown("Overall process flow diagram")
            btn7 = gr.Button("Show Process Flow", variant="primary")
            out_flow = gr.Image(type="numpy", label="Process Flow")
            btn7.click(show_flow, outputs=out_flow)

        with gr.Tab("8 Bayesian Flow"):
            gr.Markdown("Self-learning optimization loop diagram")
            btn8 = gr.Button("Show Optimization Loop", variant="primary")
            out_bf = gr.Image(type="numpy", label="Bayesian Flow")
            btn8.click(show_bayesian, outputs=out_bf)

        with gr.Tab("9 Bayesian Optimize"):
            gr.Markdown("Get next experiment suggestion from Bayesian optimizer")
            btn9 = gr.Button("Get Suggestion", variant="primary")
            out_bo = gr.Textbox(label="Optimization Suggestion", lines=18)
            btn9.click(get_bo, outputs=out_bo)

        with gr.Tab("10 Batch Records"):
            gr.Markdown("Submit experiment results -> model auto-learns")
            with gr.Accordion("Submit Batch Data", open=True):
                with gr.Row():
                    b1 = gr.Number(label="In2O3 wt%", value=85)
                    b2 = gr.Number(label="SnO2 wt%", value=10)
                    b3 = gr.Number(label="Impurity wt%", value=5)
                    b4 = gr.Number(label="D50 um", value=10)
                with gr.Row():
                    b5 = gr.Number(label="Carbon ratio (0-1)", value=0.20)
                    b6 = gr.Number(label="NH4Cl ratio (0-1)", value=0.04)
                with gr.Row():
                    b7 = gr.Number(label="Stage 1 current", value=12)
                    b8 = gr.Number(label="Stage 1 time", value=5)
                    b9 = gr.Number(label="Stage 1 temp", value=380)
                with gr.Row():
                    b10 = gr.Number(label="Stage 2 current", value=65)
                    b11 = gr.Number(label="Stage 2 time", value=10)
                    b12 = gr.Number(label="Stage 2 temp", value=1600)
                with gr.Row():
                    b13 = gr.Number(label="Pressure Pa", value=50)
                    b14 = gr.Number(label="Actual recovery %", value=90)
                    b15 = gr.Number(label="Actual purity %", value=95)
                btn10 = gr.Button("Save & Update Model", variant="primary")
                out_s = gr.Textbox(label="Result", lines=2)
                btn10.click(submit_batch,
                            inputs=[b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15],
                            outputs=out_s)
            with gr.Accordion("History Records", open=False):
                btn11 = gr.Button("Refresh")
                out_h = gr.Textbox(label="Batch List", lines=15)
                btn11.click(view_history, outputs=out_h)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
