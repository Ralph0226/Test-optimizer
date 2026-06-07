"""
ITO Waste Flash Joule Heating InCl3 Recovery Optimizer
Top-Condensation Design + Gradient Reaction Bed + Chlorination

Reaction: NH4Cl -> NH3 + HCl (300C), HCl + In2O3 -> InCl3 + H2O
Carbon is only a conductive additive (not a reductant).
Target product: InCl3 (condensed on top plate at ~280C).

Features:
  - Thermodynamic models (InCl3, SnCl4 vapor pressure)
  - XGBoost parameter prediction + Bayesian optimization
  - Gradient reaction bed visualization
  - Two-stage pulse heating curve
  - Process flow and comparison diagrams
  - Batch data management with feedback learning

Dependencies: pip install xgboost scikit-learn gradio numpy pandas scipy matplotlib
Run: python ito_optimizer_en.py
"""

import json, os, datetime, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DATA_DIR = Path(__file__).parent / "ito_ai_data"
MODEL_PATH = DATA_DIR / "batch_history.json"
DATA_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200

C = {"red":"#E74C3C","orange":"#F39C12","yellow":"#F1C40F","green":"#27AE60",
     "blue":"#2980B9","purple":"#8E44AD","gray":"#7F8C8D","dark":"#2C3E50"}

# ================================================================
#  1. Thermodynamic Models (Clausius-Clapeyron)
# ================================================================

INCL3 = {"A": 26.5, "B": 16800, "T_lo": 450, "T_hi": 750}
SNCL4 = {"A": 22.8, "B": 5200,  "T_lo": 200, "T_hi": 500}
SNCL2 = {"A": 25.0, "B": 13500, "T_lo": 500, "T_hi": 1000}

def _cond_temp(params, P_pa):
    T = params["B"] / (params["A"] - np.log(max(P_pa, 1.0)))
    return np.clip(T, params["T_lo"], params["T_hi"])

def calc_condensation_temps(P_pa):
    T_in = _cond_temp(INCL3, P_pa) - 273.15
    T_sn = _cond_temp(SNCL4, P_pa) - 273.15
    return {
        "InCl3 condensation (C)": round(T_in, 1),
        "SnCl4 condensation (C)": round(T_sn, 1),
        "InCl3-SnCl4 gap (C)": round(T_in - T_sn, 1),
        "Recommended condenser plate (C)": round(T_in - 20, 1),
    }

def vapor_curve(params, T_lo_C, T_hi_C, n=200):
    T = np.linspace(T_lo_C, T_hi_C, n)
    P = np.exp(params["A"] - params["B"] / (T + 273.15))
    return T, P

# ================================================================
#  2. Synthetic Data + XGBoost Model
# ================================================================

def _gen_synthetic(n=200, seed=42):
    """
    Synthetic data for chlorination pathway.
    Key differences from carbon-reduction model:
    - Carbon ratio is lower (5-20%, conductive additive only)
    - NH4Cl ratio is higher (5-20%, main reagent)
    - Reaction temperatures are lower (300-1200C)
    - Recovery depends on NH4Cl amount and temperature
    """
    rng = np.random.RandomState(seed)
    in2o3 = rng.uniform(55, 95, n)
    sno2 = np.clip(95 - in2o3 + rng.normal(0, 3, n), 2, 35)
    imp = 100 - in2o3 - sno2
    d50 = rng.uniform(3, 50, n)

    # Carbon: only for conductivity (5-20%)
    c_ratio = np.clip(0.08 + 0.001*in2o3 + rng.normal(0, 0.02, n), 0.03, 0.20)
    # NH4Cl: main chlorination reagent (5-20%)
    nh_ratio = np.clip(0.08 + 0.001*sno2 + rng.normal(0, 0.02, n), 0.03, 0.20)

    # Stage 1: low-temp pre-chlorination (NH4Cl decomposition)
    t1 = np.clip(300 + 0.3*sno2 + rng.normal(0, 15, n), 200, 500)
    d1 = np.clip(3 + 0.04*sno2 + rng.normal(0, 1, n), 1, 30)
    i1 = np.clip(5 + 0.08*in2o3 + rng.normal(0, 2, n), 3, 20)

    # Stage 2: high-temp chlorination + InCl3 volatilization
    t2 = np.clip(500 + 0.8*in2o3 - 0.3*sno2 + rng.normal(0, 30, n), 400, 1100)
    d2 = np.clip(3 + 0.02*in2o3 + rng.normal(0, 0.5, n), 0.5, 30)
    i2 = np.clip(30 + 0.2*in2o3 + rng.normal(0, 5, n), 20, 80)

    pr = np.clip(30 + 2*sno2 + rng.normal(0, 10, n), 10, 500)

    # Recovery depends on NH4Cl (more NH4Cl = more HCl = more chlorination)
    # and temperature (higher T = faster reaction)
    rec = np.clip(80 + 0.1*in2o3 - 0.1*sno2 + 0.3*(nh_ratio*100) + rng.normal(0, 3, n), 50, 98)
    pur = np.clip(93 - 0.15*sno2 + 0.05*in2o3 + rng.normal(0, 2, n), 70, 99.5)

    return pd.DataFrame({
        "in2o3_pct": np.round(in2o3,2), "sno2_pct": np.round(sno2,2),
        "impurity_pct": np.round(imp,2), "d50_um": np.round(d50,1),
        "carbon_ratio": np.round(c_ratio,4), "nh4cl_ratio": np.round(nh_ratio,4),
        "current_s1": np.round(i1,1), "time_s1": np.round(d1,2),
        "temp_s1": np.round(t1,1), "current_s2": np.round(i2,1),
        "time_s2": np.round(d2,2), "temp_s2": np.round(t2,1),
        "pressure_Pa": np.round(pr,1),
        "recovery_pct": np.round(rec,2), "purity_pct": np.round(pur,2),
    })

INPUT_COLS = ["in2o3_pct","sno2_pct","impurity_pct","d50_um"]
OUTPUT_COLS = ["carbon_ratio","nh4cl_ratio","current_s1","time_s1","temp_s1",
    "current_s2","time_s2","temp_s2","pressure_Pa","recovery_pct","purity_pct"]
BOUNDS = {
    "carbon_ratio": (0.03, 0.20),  # conductive additive only
    "nh4cl_ratio":  (0.03, 0.20),  # chlorination reagent
    "current_s1":   (3, 20),
    "time_s1":      (1, 30),
    "temp_s1":      (200, 500),
    "current_s2":   (20, 80),
    "time_s2":      (0.5, 30),
    "temp_s2":      (400, 1100),
    "pressure_Pa":  (10, 500),
    "recovery_pct": (50, 98),
    "purity_pct":   (70, 99.5),
}

def _load_hist():
    if MODEL_PATH.exists(): return json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    return []
def _save_hist(r):
    MODEL_PATH.write_text(json.dumps(r, indent=2), encoding="utf-8")
def add_batch(rec):
    r = _load_hist()
    rec["timestamp"] = datetime.datetime.now().isoformat()
    rec["id"] = hashlib.md5(rec["timestamp"].encode()).hexdigest()[:8]
    r.append(rec); _save_hist(r); return rec["id"]
def get_records():
    r = _load_hist()
    return pd.DataFrame(r) if r else pd.DataFrame()

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

class ParamPredictor:
    def __init__(self):
        self.sx, self.sy = StandardScaler(), StandardScaler()
        base = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        self.model = MultiOutputRegressor(base); self._ok = False
    def _data(self):
        syn = _gen_synthetic(200); real = get_records()
        if not real.empty and all(c in real.columns for c in INPUT_COLS+OUTPUT_COLS):
            rc = real[INPUT_COLS+OUTPUT_COLS].dropna()
            return pd.concat([syn, pd.concat([rc]*3)], ignore_index=True)
        return syn
    def train(self):
        df = self._data(); X, y = df[INPUT_COLS], df[OUTPUT_COLS]
        self.sx.fit(X); self.sy.fit(y)
        self.model.fit(self.sx.transform(X), self.sy.transform(y)); self._ok = True
    def predict(self, in2o3, sno2, imp, d50):
        if not self._ok: self.train()
        X = pd.DataFrame([{"in2o3_pct":in2o3,"sno2_pct":sno2,"impurity_pct":imp,"d50_um":d50}])
        y = self.sy.inverse_transform(self.model.predict(self.sx.transform(X)))
        return {c: round(float(np.clip(y[0,i], BOUNDS[c][0], BOUNDS[c][1])), 2)
                for i, c in enumerate(OUTPUT_COLS)}

predictor = ParamPredictor()

# ================================================================
#  3. Bayesian Optimization
# ================================================================

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimizer:
    SPACE = {
        "carbon_ratio": (0.03, 0.20), "nh4cl_ratio": (0.03, 0.20),
        "temp_s1": (200, 500), "time_s1": (1, 30),
        "temp_s2": (400, 1100), "time_s2": (0.5, 30),
        "pressure_Pa": (10, 500),
    }
    NAMES = list(SPACE.keys())
    def __init__(self):
        self.gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0)*Matern(nu=2.5),
            alpha=1e-4, n_restarts_optimizer=5, random_state=42)
    def suggest_next(self):
        df = get_records()
        if df.empty or len(df)<3 or not all(c in df.columns for c in self.NAMES+["recovery_pct"]):
            rng = np.random.RandomState(42)
            s = {n: round(rng.uniform(lo,hi),2) for n,(lo,hi) in self.SPACE.items()}
            s["_strategy"] = "Latin Hypercube (run 5-10 experiments first)"; return s
        X = df[self.NAMES].values; y = df["recovery_pct"].values
        m = ~np.isnan(X).any(axis=1) & ~np.isnan(y); X, y = X[m], y[m]
        if len(X) < 5:
            rng = np.random.RandomState(len(X)*7+13)
            s = {n: round(rng.uniform(lo,hi),2) for n,(lo,hi) in self.SPACE.items()}
            s["_strategy"] = "Latin Hypercube (need more data)"; return s
        lo = np.array([v[0] for v in self.SPACE.values()])
        hi = np.array([v[1] for v in self.SPACE.values()])
        self.gp.fit((X-lo)/(hi-lo), y); best_y = y.max()
        best_c, best_ei = None, -1; rng = np.random.RandomState(42)
        for _ in range(5000):
            c = np.array([rng.uniform(l,h) for l,h in self.SPACE.values()])
            cn = (c-lo)/(hi-lo)
            mu, sigma = self.gp.predict(cn.reshape(1,-1), return_std=True)
            sigma = max(float(sigma),1e-8); z = (float(mu)-best_y)/sigma
            ei = (float(mu)-best_y)*0.5*(1+np.math.erf(z/np.sqrt(2)))+sigma*np.exp(-0.5*z*z)/np.sqrt(2*np.pi)
            if ei > best_ei: best_ei, best_c = ei, c
        s = {n: round(float(np.clip(best_c[i],self.SPACE[n][0],self.SPACE[n][1])),2)
             for i,n in enumerate(self.NAMES)}
        s["_strategy"] = f"GP-EI ({len(X)} data points)"; return s

bo = BayesianOptimizer()

# ================================================================
#  4. Visualization
# ================================================================

def _to_img(fig):
    fig.tight_layout(); fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy(); plt.close(fig); return img

def plot_gradient_bed():
    """Gradient reaction bed: NH4Cl-rich layers for chlorination, carbon for conductivity."""
    fig, ax = plt.subplots(figsize=(6,8))
    ax.set_xlim(0,10); ax.set_ylim(0,12); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Gradient Concentration Reaction Bed (Chlorination)", fontsize=13, fontweight="bold", pad=12)
    for y,lb in [(0.3,"Bottom Electrode (-)"),(10.8,"Top Electrode (+)")]:
        ax.add_patch(FancyBboxPatch((2,y),6,0.5,boxstyle="round,pad=0.1",
            facecolor=C["gray"],edgecolor=C["dark"],lw=2))
        ax.text(5,y+0.25,lb,ha="center",va="center",fontsize=9,fontweight="bold",color="white")
    for y,h,color,lb in [(1.0,3.2,"#E74C3C","Bottom: Pre-chlorination\nNH4Cl 12% + Carbon 8%"),
        (4.4,2.6,"#F39C12","Middle: Transition\nNH4Cl 7% + Carbon 8%"),
        (7.2,3.3,"#F1C40F","Top: Chlorination+Volatilization\nNH4Cl 15% + Carbon 12%")]:
        ax.add_patch(FancyBboxPatch((2,y),6,h,boxstyle="round,pad=0.05",
            facecolor=color,edgecolor=C["dark"],lw=1.5,alpha=0.85))
        ax.text(5,y+h/2,lb,ha="center",va="center",fontsize=7.5,fontweight="bold")
    for y in [4.3,7.1]:
        ax.plot([2,8],[y,y],color=C["dark"],lw=1,ls="--")
        ax.text(8.2,y,"Carbon paper",fontsize=6,va="center",color=C["gray"])
    grad=np.linspace(0,1,256).reshape(-1,1)
    ax.imshow(grad,aspect="auto",cmap="coolwarm",extent=[0.8,1.5,1.0,10.5],origin="lower")
    ax.text(0.5,3,"Low T",fontsize=7,ha="center",color=C["blue"],fontweight="bold")
    ax.text(0.5,9,"High T",fontsize=7,ha="center",color=C["red"],fontweight="bold")
    ax.annotate("",xy=(1.2,10.0),xytext=(1.2,1.5),arrowprops=dict(arrowstyle="->",color=C["blue"],lw=2.5))
    ax.text(1.2,5.5,"Current",ha="center",va="center",fontsize=8,color=C["blue"],fontweight="bold",rotation=90)
    # Reaction annotation
    ax.text(5,0.3,"NH4Cl -> NH3 + HCl | HCl + In2O3 -> InCl3 + H2O",
        ha="center",va="center",fontsize=6.5,color=C["dark"],
        bbox=dict(boxstyle="round,pad=0.2",facecolor="white",alpha=0.9,edgecolor=C["dark"]))
    return _to_img(fig)

def plot_pulse_curve(res):
    """Two-stage pulse: low-temp NH4Cl decomposition + high-temp InCl3 volatilization."""
    fig, ax = plt.subplots(figsize=(9,5))
    t1,d1,t2,d2 = res["temp_s1"],res["time_s1"],res["temp_s2"],res["time_s2"]
    time=[0,d1*0.15,d1*0.7,d1,d1+0.5,d1+0.5+d2*0.2,d1+0.5+d2*0.8,d1+0.5+d2,d1+0.5+d2+1,d1+d2+3]
    temp=[25,t1*0.4,t1*0.85,t1,t1*0.95,t2*0.7,t2*0.95,t2,t2*0.4,25]
    ax.plot(time,temp,color=C["red"],lw=2.5,zorder=5)
    ax.fill_between(time,temp,alpha=0.12,color=C["red"])
    ax.axvspan(0,d1,alpha=0.08,color=C["blue"])
    ax.axvspan(d1+0.5,d1+0.5+d2,alpha=0.08,color=C["orange"])
    ax.axhspan(200,500,alpha=0.06,color=C["blue"],label="NH4Cl decomposition")
    ax.axhspan(500,1100,alpha=0.06,color=C["orange"],label="InCl3 volatilization")
    ax.annotate(f"Stage 1\nNH4Cl decomposition\n{t1:.0f}C / {d1:.1f}s",xy=(d1/2,t1),xytext=(d1/2,t1+150),fontsize=8,ha="center",
        arrowprops=dict(arrowstyle="->",color=C["blue"]),bbox=dict(boxstyle="round,pad=0.4",facecolor=C["blue"],alpha=0.15))
    ax.annotate(f"Stage 2\nInCl3 volatilization\n{t2:.0f}C / {d2:.1f}s",xy=(d1+0.5+d2/2,t2),xytext=(d1+0.5+d2/2,t2+150),fontsize=8,ha="center",
        arrowprops=dict(arrowstyle="->",color=C["orange"]),bbox=dict(boxstyle="round,pad=0.4",facecolor=C["orange"],alpha=0.15))
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)")
    ax.set_title("Two-Stage Pulse: NH4Cl Decomp + InCl3 Volatilization",fontsize=12,fontweight="bold")
    ax.legend(loc="upper left",fontsize=8); ax.set_ylim(0,max(t2*1.3,1200)); ax.grid(True,alpha=0.3)
    return _to_img(fig)

def plot_vapor_pressure():
    """Vapor pressure curves: InCl3 vs SnCl4 separation window."""
    fig, ax = plt.subplots(figsize=(8,5))
    for p,Tlo,Thi,c,lb in [(INCL3,300,700,C["green"],"InCl3 (sublimation)"),
        (SNCL4,0,200,C["blue"],"SnCl4 (vaporization)"),
        (SNCL2,400,800,C["purple"],"SnCl2 (vaporization)")]:
        T,P = vapor_curve(p,Tlo,Thi); ax.semilogy(T,P,color=c,lw=2.5,label=lb)
    for P,lb,c in [(100,"100 Pa",C["red"]),(500,"500 Pa",C["orange"])]:
        ax.axhline(P,color=c,ls=":",lw=1.2,alpha=0.7); ax.text(50,P*1.4,lb,fontsize=7,color=c)
    ax.annotate("InCl3-SnCl4 separation\nwindow (>200C at 100Pa)",
        xy=(250,300),xytext=(400,500),fontsize=8,fontweight="bold",color=C["green"],
        arrowprops=dict(arrowstyle="->",color=C["green"]),
        bbox=dict(boxstyle="round,pad=0.3",facecolor=C["green"],alpha=0.1))
    ax.set_xlabel("Temperature (C)"); ax.set_ylabel("Vapor Pressure (Pa)")
    ax.set_title("InCl3 / SnCl4 / SnCl2 Vapor Pressure",fontsize=12,fontweight="bold")
    ax.legend(); ax.grid(True,alpha=0.3,which="both"); ax.set_xlim(0,800)
    return _to_img(fig)

def plot_process_flow():
    """Process flow for InCl3 recovery via chlorination."""
    fig, ax = plt.subplots(figsize=(14,8)); ax.set_xlim(0,14); ax.set_ylim(0,8); ax.axis("off")
    for x,y,w,h,c,lb in [(0.3,5.5,2.5,1.5,C["blue"],"S1: Material Prep\n+ NH4Cl & Carbon\nBatching"),
        (3.5,5.5,2.5,1.5,C["red"],"S2: Build Gradient\nReaction Bed\n(NH4Cl-rich)"),
        (6.7,5.5,2.5,1.5,C["orange"],"S3: Load into\nTop-Condenser\n& Evacuate"),
        (9.9,5.5,2.5,1.5,C["purple"],"S4: Pulse FJH\nChlorination\n+ Resistance\nMonitoring"),
        (11.5,3.2,1.8,1.2,C["green"],"S5: InCl3\nCondensation\non Top Plate"),
        (8.5,3.2,2.2,1.2,C["green"],"S6: InCl3\nProduct"),
        (5.5,3.2,2.2,1.2,C["gray"],"S7: Optional\nFurther\nProcessing")]:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.15",facecolor=c,alpha=0.2,edgecolor=c,lw=2))
        ax.text(x+w/2,y+h/2,lb,ha="center",va="center",fontsize=7.5,fontweight="bold")
    for s,e in [((2.9,6.25),(3.4,6.25)),((6.1,6.25),(6.6,6.25)),((9.3,6.25),(9.8,6.25)),
        ((11.2,5.5),(12.4,4.5)),((11.2,3.8),(10.8,3.8)),((8.5,3.8),(7.8,3.8))]:
        ax.annotate("",xy=e,xytext=s,arrowprops=dict(arrowstyle="-|>",color=C["dark"],lw=1.5))
    ax.add_patch(FancyBboxPatch((0.3,2.5),2.5,2.0,boxstyle="round,pad=0.15",facecolor=C["purple"],alpha=0.15,edgecolor=C["purple"],lw=1.5))
    ax.text(1.55,3.5,"XGBoost +\nBayesian\nOptimization",ha="center",va="center",fontsize=8,fontweight="bold",color=C["purple"])
    ax.text(7,7.5,"Process Flow: Top-Condensation FJH InCl3 Recovery",fontsize=13,fontweight="bold",ha="center")
    return _to_img(fig)

def plot_bayesian_flow():
    fig, ax = plt.subplots(figsize=(11,5.5)); ax.set_xlim(0,11); ax.set_ylim(0,5.5); ax.axis("off")
    for x,y,w,h,c,lb in [(0.2,2.2,2.0,1.4,C["blue"],"ITO Composition"),
        (2.8,2.2,2.0,1.4,C["purple"],"XGBoost\nPrediction"),
        (5.4,2.2,2.0,1.4,C["orange"],"FJH Chlorination\nExperiment"),
        (8.0,2.2,2.0,1.4,C["green"],"InCl3 Product\nAnalysis"),
        (5.4,0.2,2.0,1.0,C["red"],"Bayesian\nOptimization")]:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.15",facecolor=c,alpha=0.15,edgecolor=c,lw=2))
        ax.text(x+w/2,y+h/2,lb,ha="center",va="center",fontsize=7.5,fontweight="bold")
    for s,e in [((2.3,2.9),(2.7,2.9)),((4.9,2.9),(5.3,2.9)),((7.5,2.9),(7.9,2.9))]:
        ax.annotate("",xy=e,xytext=s,arrowprops=dict(arrowstyle="-|>",color=C["dark"],lw=1.5))
    ax.annotate("Suggestion",xy=(6.4,1.2),xytext=(6.4,2.1),fontsize=7,ha="center",color=C["red"],fontweight="bold",
        arrowprops=dict(arrowstyle="<-",color=C["red"],lw=1.5))
    ax.annotate("Feedback",xy=(3.8,2.2),xytext=(9.0,1.8),fontsize=7,ha="center",color=C["green"],fontweight="bold",
        arrowprops=dict(arrowstyle="-|>",color=C["green"],lw=1.5,ls="--",connectionstyle="arc3,rad=-0.3"))
    ax.text(5.5,5.2,"Self-Learning Optimization Loop (InCl3 Recovery)",fontsize=12,fontweight="bold",ha="center")
    return _to_img(fig)

# ================================================================
#  5. Gradio UI
# ================================================================

def create_ui():
    import gradio as gr
    predictor.train()

    def predict_params(in2o3, sno2, impurity, d50):
        if abs(in2o3+sno2+impurity-100) > 0.5:
            return "In2O3+SnO2+impurity should sum to ~100%", None, None
        res = predictor.predict(in2o3, sno2, impurity, d50)
        cond = calc_condensation_temps(res["pressure_Pa"])
        lines = ["="*55,"  AI Recommended Parameters (InCl3 Recovery)","="*55,"",
            "-- Gradient Bed (Chlorination) --",
            f"  Bottom: NH4Cl 12% + Carbon {res['carbon_ratio']*80:.1f}%",
            f"  Middle: NH4Cl  7% + Carbon {res['carbon_ratio']*80:.1f}%",
            f"  Top:    NH4Cl 15% + Carbon {res['carbon_ratio']*100:.1f}%","",
            "-- Stage 1: NH4Cl Decomposition --",
            f"  Current: {res['current_s1']:.1f} A/cm2  Time: {res['time_s1']:.1f}s  Temp: {res['temp_s1']:.0f}C",
            "-- Stage 2: InCl3 Volatilization --",
            f"  Current: {res['current_s2']:.1f} A/cm2  Time: {res['time_s2']:.1f}s  Temp: {res['temp_s2']:.0f}C",
            "-- Pressure --", f"  Chamber: {res['pressure_Pa']:.0f} Pa","",
            "="*55,"  Condensation Design","="*55,""]
        for k,v in cond.items(): lines.append(f"  {k}: {v} C")
        lines += ["",f"  InCl3 Recovery: ~{res['recovery_pct']:.1f}%",f"  InCl3 Purity: ~{res['purity_pct']:.1f}%"]
        return "\n".join(lines), plot_pulse_curve(res), res

    def get_bo():
        s = bo.suggest_next(); st = s.pop("_strategy")
        return ("="*55+"\n  Bayesian Optimization\n"+"="*55+f"\n\n  Strategy: {st}\n\n"
            f"  Carbon: {s['carbon_ratio']*100:.1f}%  NH4Cl: {s['nh4cl_ratio']*100:.1f}%\n"
            f"  Stage 1: {s['temp_s1']:.0f}C / {s['time_s1']:.1f}s\n"
            f"  Stage 2: {s['temp_s2']:.0f}C / {s['time_s2']:.1f}s\n"
            f"  Pressure: {s['pressure_Pa']:.0f} Pa")

    def submit_batch(in2o3,sno2,imp,d50,carbon,nh4cl,cs1,ts1,tmp1,cs2,ts2,tmp2,pres,rec,pur):
        bid = add_batch({"in2o3_pct":in2o3,"sno2_pct":sno2,"impurity_pct":imp,"d50_um":d50,
            "carbon_ratio":carbon,"nh4cl_ratio":nh4cl,
            "current_s1":cs1,"time_s1":ts1,"temp_s1":tmp1,
            "current_s2":cs2,"time_s2":ts2,"temp_s2":tmp2,"pressure_Pa":pres,
            "recovery_pct":rec,"purity_pct":pur})
        predictor.train()
        return f"Batch {bid} saved | Total: {len(get_records())} | Model retrained"

    def view_history():
        df = get_records()
        return "No records yet" if df.empty else df.to_string(index=False)

    with gr.Blocks(title="ITO InCl3 Recovery Optimizer") as demo:
        gr.Markdown("# ITO FJH InCl3 Recovery Optimizer\n"
                     "Top-Condensation + Gradient Bed (Chlorination) + AI Optimization")
        with gr.Tab("1 Parameter Prediction"):
            gr.Markdown("Input ITO composition -> AI recommended parameters for InCl3 recovery")
            with gr.Row():
                in1=gr.Number(label="In2O3 wt%",value=85); in2=gr.Number(label="SnO2 wt%",value=10)
                in3=gr.Number(label="Impurity wt%",value=5); in4=gr.Number(label="D50 um",value=10)
            btn1=gr.Button("Predict",variant="primary")
            out_txt=gr.Textbox(label="Results",lines=22); out_img=gr.Image(label="Pulse Curve",type="numpy")
            btn1.click(predict_params,inputs=[in1,in2,in3,in4],outputs=[out_txt,out_img,gr.State()])
        with gr.Tab("2 Gradient Bed"):
            btn2=gr.Button("Show",variant="primary"); out_bed=gr.Image(type="numpy")
            btn2.click(plot_gradient_bed,outputs=out_bed)
        with gr.Tab("3 Pulse Curve"):
            with gr.Row():
                pt1=gr.Number(label="Stage 1 Temp",value=350); pd1=gr.Number(label="Stage 1 Time",value=5)
                pt2=gr.Number(label="Stage 2 Temp",value=800); pd2=gr.Number(label="Stage 2 Time",value=10)
            btn3=gr.Button("Plot",variant="primary"); out_p=gr.Image(type="numpy")
            btn3.click(lambda t1,d1,t2,d2: plot_pulse_curve({"temp_s1":t1,"time_s1":d1,"temp_s2":t2,"time_s2":d2}),
                inputs=[pt1,pd1,pt2,pd2],outputs=out_p)
        with gr.Tab("4 Vapor Pressure"):
            btn4=gr.Button("Show",variant="primary"); out_vp=gr.Image(type="numpy")
            btn4.click(plot_vapor_pressure,outputs=out_vp)
        with gr.Tab("5 Process Flow"):
            btn5=gr.Button("Show",variant="primary"); out_fl=gr.Image(type="numpy")
            btn5.click(plot_process_flow,outputs=out_fl)
        with gr.Tab("6 Bayesian Flow"):
            btn6=gr.Button("Show",variant="primary"); out_bf=gr.Image(type="numpy")
            btn6.click(plot_bayesian_flow,outputs=out_bf)
        with gr.Tab("7 Bayesian Optimize"):
            btn7=gr.Button("Get Suggestion",variant="primary")
            out_bo=gr.Textbox(label="Suggestion",lines=12)
            btn7.click(get_bo,outputs=out_bo)
        with gr.Tab("8 Batch Records"):
            with gr.Accordion("Submit",open=True):
                with gr.Row():
                    b1=gr.Number(label="In2O3",value=85); b2=gr.Number(label="SnO2",value=10)
                    b3=gr.Number(label="Imp",value=5); b4=gr.Number(label="D50",value=10)
                with gr.Row():
                    b5=gr.Number(label="Carbon (0-1)",value=0.08); b6=gr.Number(label="NH4Cl (0-1)",value=0.12)
                with gr.Row():
                    b7=gr.Number(label="S1 current",value=8); b8=gr.Number(label="S1 time",value=5); b9=gr.Number(label="S1 temp",value=350)
                with gr.Row():
                    b10=gr.Number(label="S2 current",value=50); b11=gr.Number(label="S2 time",value=10); b12=gr.Number(label="S2 temp",value=800)
                with gr.Row():
                    b13=gr.Number(label="Pressure",value=50); b14=gr.Number(label="InCl3 Recovery %",value=90); b15=gr.Number(label="InCl3 Purity %",value=95)
                btn8=gr.Button("Save",variant="primary"); out_s=gr.Textbox(label="Result",lines=2)
                btn8.click(submit_batch,inputs=[b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15],outputs=out_s)
            with gr.Accordion("History",open=False):
                btn9=gr.Button("Refresh"); out_h=gr.Textbox(label="Records",lines=15)
                btn9.click(view_history,outputs=out_h)
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, inbrowser=True, theme=gr.themes.Soft())
