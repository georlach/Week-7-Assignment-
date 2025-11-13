#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 00:36:14 2025

@author: giorgoslachanoudes
"""

# =============================================================================
# WEEK 7 ASSIGNMENT
# =============================================================================

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ---------------------------- Paths ------------------------------------------
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_path = os.path.join(desktop_path, "week7")
os.makedirs(output_path, exist_ok=True)
print(f"📂 Saving outputs to: {output_path}")

# ---------------------------- Universe ---------------------------------------
STOCK_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","ORCL","INTC","CSCO","ADBE",
    "WMT","COST","HD","MCD","NKE","SBUX","LOW","TGT","BKNG","NFLX",
    "JPM","BAC","WFC","C","MS","GS","BLK","AXP","SCHW","PYPL",
    "UNH","JNJ","PFE","MRK","ABBV","LLY","TMO","DHR","BMY","AMGN",
    "CAT","GE","BA","DE","XOM","CVX","COP","NEE","LIN","RIO"
]

# ------------------------- Download 3y prices/volume --------------------------
END_DATE = datetime.today().date()
START_DATE = END_DATE - timedelta(days=365*3 + 5)
print(f"\n⏱️ Downloading daily data: {START_DATE} → {END_DATE}")

raw = yf.download(
    STOCK_TICKERS, start=str(START_DATE), end=str(END_DATE),
    interval="1d", auto_adjust=False, progress=False, group_by="ticker"
)

close_frames, vol_frames = [], []
for t in STOCK_TICKERS:
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            s_close = raw[(t, "Adj Close")].rename(t)
            s_vol   = raw[(t, "Volume")].rename(t)
        else:
            s_close = raw["Adj Close"].rename(t)
            s_vol   = raw["Volume"].rename(t)
        close_frames.append(s_close)
        vol_frames.append(s_vol)
    except Exception as e:
        print(f"[WARN] Missing data for {t}: {e}")

prices = pd.concat(close_frames, axis=1)
volumes = pd.concat(vol_frames, axis=1)

# Reindex σε business days & basic fills
bd_index = pd.bdate_range(prices.index.min(), prices.index.max(), freq="C")
prices  = prices.reindex(bd_index).ffill().bfill().astype(float)
volumes = volumes.reindex(bd_index).fillna(0).astype(float)

# Save raw panels (προαιρετικό αλλά χρήσιμο)
prices.to_csv(os.path.join(output_path, "prices_stocks_adj_close_3y.csv"))
volumes.to_csv(os.path.join(output_path, "volume_stocks_3y.csv"))
print("💾 Saved raw panels.")

# -------------------------- Multi-window FEATURES ----------------------------
# Windows που ζήτησες
WINDOWS = [1, 3, 5, 10, 30]
TRADING_DAYS = 252.0
SQRT_252 = np.sqrt(TRADING_DAYS)

# Ημερήσιοι log-returns
logret = np.log(prices / prices.shift(1))

# Market cap series = sharesOutstanding * price (sharesOutstanding ~ σταθερό)
print("🔎 Fetching sharesOutstanding (this may take a bit)...")
shares_out = {}
for t in STOCK_TICKERS:
    so = None
    try:
        fi = yf.Ticker(t).fast_info
        so = fi.get("sharesOutstanding", None)
    except Exception:
        pass
    if so is None:
        try:
            info = yf.Ticker(t).get_info()
            so = info.get("sharesOutstanding", None)
        except Exception:
            so = None
    shares_out[t] = so

mcap_series = pd.DataFrame(index=prices.index, columns=prices.columns, dtype="float64")
for t in STOCK_TICKERS:
    so = shares_out.get(t, None)
    if so is not None:
        mcap_series[t] = prices[t] * float(so)
    else:
        mcap_series[t] = np.nan  # θα γεμίσει/μείνει NaN αναλόγως

# Helper: volatility proxy for window=1 (|last return| * sqrt(252))
def one_day_vol_proxy(series):
    r = series.dropna()
    if r.empty:
        return np.nan
    return float(abs(r.iloc[-1]) * SQRT_252)

# Φτιάχνουμε features: για κάθε παράθυρο w, παίρνουμε το ΤΕΛΕΥΤΑΙΟ διαθέσιμο rolling metric
feature_dict = {}
for w in WINDOWS:
    # Volatility (annualized): std over last w returns * sqrt(252)
    if w == 1:
        vol_w = logret.apply(one_day_vol_proxy).rename(f"volatility_ann_w{w}")
    else:
        vol_w = (logret.rolling(window=w).std().iloc[-1] * SQRT_252).rename(f"volatility_ann_w{w}")

    # Volume: rolling mean over w days (last value)
    volavg_w = volumes.rolling(window=w, min_periods=1).mean().iloc[-1].rename(f"volume_avg_w{w}")

    # Market Cap: rolling mean over w days (last value)
    mcapavg_w = mcap_series.rolling(window=w, min_periods=1).mean().iloc[-1].rename(f"market_cap_avg_w{w}")

    feature_dict[vol_w.name]   = vol_w
    feature_dict[volavg_w.name]= volavg_w
    feature_dict[mcapavg_w.name]= mcapavg_w

# Συναρμολόγηση feature matrix (assets x features)
features = pd.DataFrame(feature_dict)
features = features.reindex(STOCK_TICKERS)  # διατήρηση σειράς

# Drop assets που έχουν υπερβολικά NaNs (κρατάμε όσα έχουν ≥ 70% διαθέσιμα features)
min_non_na = int(np.ceil(0.7 * features.shape[1]))
keep_mask = features.notna().sum(axis=1) >= min_non_na
features = features.loc[keep_mask].copy()

# Validation: έλεγχος μηδενικής διακύμανσης ανά feature (δεν πρέπει να υπάρχουν σταθερές στήλες)
std_per_col = features.std(numeric_only=True)
zero_cols = std_per_col[std_per_col == 0].index.tolist()
if zero_cols:
    print(f"⚠️ Some features are constant across assets and will be dropped: {zero_cols}")
    features.drop(columns=zero_cols, inplace=True)

print(f"\n✅ Final feature matrix: {features.shape[0]} assets × {features.shape[1]} features")
feat_path = os.path.join(output_path, "features_multiwindow_matrix.csv")
features.to_csv(feat_path)
print(f"💾 Saved features → {feat_path}")

# ----------------------------- SimplePCA (prof) ------------------------------
class SimplePCA:
    """Professor's PCA helper class"""
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = None

    def fit(self, X, feature_names=None):
        X_scaled = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        self.feature_names = feature_names
        tv = self.pca.explained_variance_ratio_.sum()
        print(f"\n✅ PCA fitted successfully! Components: {self.pca.n_components_} | Total variance: {tv:.2%}")
        return self

    def transform(self, X):
        return self.pca.transform(self.scaler.transform(X))

    def fit_transform(self, X, feature_names=None):
        self.fit(X, feature_names)
        return self.transform(X)

    def get_variance_summary(self):
        var = self.pca.explained_variance_ratio_
        cum = np.cumsum(var)
        return pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(len(var))],
            "Variance_Explained": var,
            "Cumulative_Variance": cum,
            "Eigenvalue": self.pca.explained_variance_
        })

    def get_loadings(self):
        return pd.DataFrame(
            self.pca.components_.T,
            columns=[f"PC{i+1}" for i in range(self.pca.n_components_)],
            index=self.feature_names if self.feature_names is not None else range(self.pca.n_components_)
        )

    def plot_scree(self, figsize=(12,4)):
        df = self.get_variance_summary()
        fig, ax = plt.subplots(1,2,figsize=figsize)
        ax[0].bar(df["PC"], df["Variance_Explained"], alpha=.8)
        ax[0].plot(df["PC"], df["Variance_Explained"], "o-")
        ax[0].set_title("Scree (per PC)"); ax[0].set_ylabel("Variance")
        ax[1].plot(df["PC"], df["Cumulative_Variance"], "o-")
        ax[1].axhline(0.9, color="orange", ls="--", label="90%")
        ax[1].set_title("Cumulative"); ax[1].legend()
        plt.tight_layout()
        return fig

# -------------------------- PCA with ≥90% variance ---------------------------
# Impute τυχόν NaNs με median ανά στήλη
X = features.copy()
for c in X.columns:
    X[c] = X[c].fillna(X[c].median(skipna=True))

# 1) Fit με όλες για να αποφασίσουμε πόσες PCs χρειάζονται
pca_all = SimplePCA(n_components=None)
_ = pca_all.fit_transform(X.values, feature_names=X.columns.tolist())
var_all = pca_all.get_variance_summary()
print("\n📊 PCA variance (all PCs):")
print(var_all.round(4).to_string(index=False))

# 2) Επιλογή του ελάχιστου m με cumulative ≥ 0.90
cum = var_all["Cumulative_Variance"].values
m = int(np.argmax(cum >= 0.90) + 1)
print(f"\n🔎 Using n_components = {m} (to reach ≥90% cumulative variance)")

# 3) Refit με m PCs και αποθήκευση outputs
pca = SimplePCA(n_components=m)
X_pca = pca.fit_transform(X.values, feature_names=X.columns.tolist())

var_df = pca.get_variance_summary()
load_df = pca.get_loadings()
var_df.to_csv(os.path.join(output_path, "pca_variance_summary_multiwindow.csv"), index=False)
load_df.to_csv(os.path.join(output_path, "pca_feature_loadings_multiwindow.csv"))

fig = pca.plot_scree(figsize=(12,4))
fig.savefig(os.path.join(output_path, "pca_scree_multiwindow.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 4) Πρόχειρο scatter αν m>=2
if X_pca.shape[1] >= 2:
    pc1, pc2 = X_pca[:,0], X_pca[:,1]
    plt.figure(figsize=(7.2,6))
    plt.scatter(pc1, pc2, s=50)
    for name, x_, y_ in zip(features.index, pc1, pc2):
        plt.text(x_+0.02, y_+0.02, name, fontsize=7, alpha=0.7)
    plt.axhline(0, color="gray", lw=0.7); plt.axvline(0, color="gray", lw=0.7)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"PCA Scatter (PC1 vs PC2) — m={m}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "pca_scatter_pc1_pc2_multiwindow.png"), dpi=150)
    plt.close()

print("\n✅ Saved:")
print(" - features_multiwindow_matrix.csv")
print(" - pca_variance_summary_multiwindow.csv")
print(" - pca_feature_loadings_multiwindow.csv")
print(" - pca_scree_multiwindow.png")
if X_pca.shape[1] >= 2:
    print(" - pca_scatter_pc1_pc2_multiwindow.png")
print("\n🏁 Done.")

# =============================================================================
# K-MEANS πάνω στα PCA scores
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ---------- Paths / fallbacks ----------
try:
    output_path  # από το προηγούμενο κελί
except NameError:
    output_path = os.path.abspath("./week7")  # fallback

# Αν δεν υπάρχουν X_pca / features, τα ξαναφτιάχνουμε γρήγορα από το CSV
try:
    X_pca  # noqa: F821
    features  # noqa: F821
except NameError:
    # Φόρτωσε features
    feat_path = os.path.join(output_path, "features_multiwindow_matrix.csv")
    features = pd.read_csv(feat_path, index_col=0)

    # Impute & PCA (ίδια λογική με πριν: cumulative ≥ 90%)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X = features.copy()
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median(skipna=True))

    scaler = StandardScaler().fit(X.values)
    Xs = scaler.transform(X.values)
    pca_all = PCA(n_components=None).fit(Xs)
    cum = np.cumsum(pca_all.explained_variance_ratio_)
    m = int(np.argmax(cum >= 0.90) + 1)
    pca = PCA(n_components=m, whiten=True).fit(Xs)
    X_pca = pca.transform(Xs)  # PCs με μοναδιαία διακύμανση


    print(f"[Fallback] Recomputed PCA with n_components={m} (≥90% variance)")

# ---------- 1) Επιλογή k με Elbow (WCSS) & Silhouette & Davies-Bouldin ----------
k_grid = range(2, 11)
rows = []
for k in k_grid:
    km = KMeans(n_clusters=k, n_init=100, random_state=42)
    labels_k = km.fit_predict(X_pca)
    wcss = km.inertia_
    sil = silhouette_score(X_pca, labels_k)
    db  = davies_bouldin_score(X_pca, labels_k)
    rows.append({"k": k, "WCSS": wcss, "Silhouette": sil, "DaviesBouldin": db})

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(os.path.join(output_path, "kmeans_k_selection_metrics.csv"), index=False)
print("\n=== K selection metrics ===")
print(metrics_df.round(4).to_string(index=False))

# Βέλτιστο k: μεγιστοποίηση Silhouette, σε ισοβαθμία ελάχιστο Davies-Bouldin
best_row = metrics_df.sort_values(["Silhouette","DaviesBouldin"], ascending=[False, True]).iloc[0]
best_k = int(best_row["k"])
print(f"\n[INFO] Selected k = {best_k} (max Silhouette, tie-breaker min DB)")

# Plots επιλογής k
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(metrics_df["k"], metrics_df["WCSS"], marker="o")
ax[0].set_title("Elbow (WCSS)"); ax[0].set_xlabel("k"); ax[0].set_ylabel("WCSS"); ax[0].grid(alpha=.3)
ax[1].plot(metrics_df["k"], metrics_df["Silhouette"], marker="o")
ax[1].set_title("Silhouette vs k"); ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette"); ax[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "kmeans_k_selection_plots.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------- 2) Τελικό K-Means με best_k ----------
kmeans = KMeans(n_clusters=best_k, n_init=100, random_state=42)
labels = kmeans.fit_predict(X_pca)

labels_df = pd.DataFrame({"asset": features.index, "cluster": labels}).set_index("asset")
labels_df.to_csv(os.path.join(output_path, "kmeans_cluster_labels.csv"))
print(f"\nSaved cluster labels -> {os.path.join(output_path, 'kmeans_cluster_labels.csv')}")

# ---------- 3) Προφίλ clusters (στον χώρο των αρχικών features) ----------
profile = features.copy()
profile["cluster"] = labels

# Μέσοι όροι ανά feature/cluster
cluster_profile_mean = profile.groupby("cluster").mean(numeric_only=True).sort_index()
cluster_profile_mean.to_csv(os.path.join(output_path, "kmeans_cluster_profile_mean.csv"))

# Z-scores ανά feature (συγκριτικά ποιο cluster είναι “ψηλά/χαμηλά”)
feat_z = (features - features.mean(numeric_only=True)) / features.std(numeric_only=True, ddof=0)
feat_z["cluster"] = labels
cluster_profile_z = feat_z.groupby("cluster").mean(numeric_only=True).sort_index()
cluster_profile_z.to_csv(os.path.join(output_path, "kmeans_cluster_profile_zscore.csv"))

print("Saved profiles ->",
      "kmeans_cluster_profile_mean.csv, kmeans_cluster_profile_zscore.csv")

# ---------- 4) Συνθέσεις clusters (λίστα assets ανά cluster) ----------
compositions = []
for c in sorted(np.unique(labels)):
    members = labels_df.index[labels_df["cluster"] == c].tolist()
    compositions.append({"cluster": c, "size": len(members), "assets": ", ".join(members)})
comp_df = pd.DataFrame(compositions)
comp_df.to_csv(os.path.join(output_path, "kmeans_cluster_compositions.csv"), index=False)
print("Saved compositions -> kmeans_cluster_compositions.csv")

# ---------- 5) Visualizations: 2D & 3D ----------
# 2D scatter (PC1–PC2)
if X_pca.shape[1] >= 2:
    pc1, pc2 = X_pca[:, 0], X_pca[:, 1]
    plt.figure(figsize=(7.6, 6.4))
    sc = plt.scatter(pc1, pc2, c=labels, s=60, cmap="tab10")
    for name, x, y in zip(features.index, pc1, pc2):
        plt.text(x+0.02, y+0.02, name, fontsize=7, alpha=0.75)
    plt.axhline(0, color="gray", lw=0.7); plt.axvline(0, color="gray", lw=0.7)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"PCA Scatter (PC1 vs PC2) — KMeans k={best_k}")
    plt.colorbar(sc, label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "kmeans_pca_scatter_2d.png"), dpi=150, bbox_inches="tight")
    plt.close()

# 3D scatter (PC1–PC2–PC3) αν υπάρχουν ≥3 PCs
if X_pca.shape[1] >= 3:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8.0, 6.8))
    ax = fig.add_subplot(111, projection='3d')
    sc3d = ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=labels, s=50, cmap="tab10")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(f"PCA Scatter 3D — KMeans k={best_k}")
    fig.colorbar(sc3d, ax=ax, shrink=0.6, label="Cluster")
    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "kmeans_pca_scatter_3d.png"), dpi=150, bbox_inches="tight")
    plt.close()

print("\n✅ K-Means completed. Outputs saved in:", output_path)

# ---------- 6) Mini-report στην κονσόλα ----------
sizes = labels_df["cluster"].value_counts().sort_index()
print("\n=== Cluster sizes ===")
print(sizes.to_string())

print("\n=== Top 8 features (|z|) per cluster ===")
for c in sorted(np.unique(labels)):
    zrow = cluster_profile_z.loc[c].abs().sort_values(ascending=False).head(8)
    print(f"\nCluster {c}:")
    print(zrow.round(2).to_string())
    
# =============================================================================
#Equal-Weighted Portfolios per Cluster
# =============================================================================
import os
import numpy as np
import pandas as pd

# Paths / fallbacks
try:
    output_path  # από προηγούμενο κελί
except NameError:
    output_path = os.path.abspath("./week7")  # fallback

labels_path = os.path.join(output_path, "kmeans_cluster_labels.csv")
features_path = os.path.join(output_path, "features_multiwindow_matrix.csv")

# Φόρτωσε labels & features για έλεγχο universe
labels_df = pd.read_csv(labels_path, index_col=0)
try:
    features
except NameError:
    features = pd.read_csv(features_path, index_col=0)

# Βεβαιώσου ότι τα assets των labels υπάρχουν στα features (ίδιο index)
assets = labels_df.index.intersection(features.index)
labels_df = labels_df.loc[assets].copy()

# 1) Υπολογισμός ίσων βαρών ανά cluster
weights_rows = []
for cluster_id, group in labels_df.groupby("cluster"):
    members = group.index.tolist()
    n = len(members)
    if n == 0:
        continue
    w = 1.0 / n
    for asset in members:
        weights_rows.append({"cluster": cluster_id, "asset": asset, "weight": w})

weights_df = pd.DataFrame(weights_rows).sort_values(["cluster", "asset"]).reset_index(drop=True)

# 2) Validation: κάθε cluster πρέπει να αθροίζει σε 1.0 (εντός ανοχής)
check = weights_df.groupby("cluster")["weight"].sum().round(6)
bad = check[np.abs(check - 1.0) > 1e-6]
if not bad.empty:
    raise ValueError(f"❌ Κάποια clusters δεν αθροίζουν σε 1:\n{bad}")

# 3) Αποθήκευση
eq_path = os.path.join(output_path, "equal_weight_portfolios.csv")
weights_df.to_csv(eq_path, index=False)

# 4) Γρήγορες συνθέσεις ανά cluster
composition_rows = []
for cluster_id, group in labels_df.groupby("cluster"):
    members = group.index.tolist()
    composition_rows.append({
        "cluster": cluster_id,
        "size": len(members),
        "assets": ", ".join(members)
    })
composition_df = pd.DataFrame(composition_rows).sort_values("cluster")
comp_path = os.path.join(output_path, "equal_weight_portfolios_composition.csv")
composition_df.to_csv(comp_path, index=False)

print("\n✅ Equal-weighted portfolios created & saved:")
print(" -", eq_path)
print(" -", comp_path)

print("\n📌 Sanity check (sum of weights per cluster):")
print(check.to_string())

# =============================================================================
# Portfolio Performance Metrics
# =============================================================================
import os
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta

# -------------------- Load data --------------------
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_path = os.path.join(desktop_path, "week7")

prices_path = os.path.join(output_path, "prices_stocks_adj_close_3y.csv")
weights_path = os.path.join(output_path, "equal_weight_portfolios.csv")

prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
weights = pd.read_csv(weights_path)

# Optional: keep only assets in weights file
tickers_in_portfolio = weights["asset"].unique().tolist()
prices = prices[tickers_in_portfolio].copy()

# -------------------- Benchmark (S&P500) --------------------
END_DATE = prices.index[-1]
START_DATE = prices.index[0]
sp500_raw = yf.download("^GSPC", start=START_DATE, end=END_DATE, progress=False)
if "Adj Close" in sp500_raw.columns:
    sp500 = sp500_raw["Adj Close"]
elif "Close" in sp500_raw.columns:
    sp500 = sp500_raw["Close"]
else:
    raise KeyError("Neither 'Adj Close' nor 'Close' found in S&P500 data.")
sp500 = sp500.reindex(prices.index).ffill()
sp500_returns = sp500.pct_change().dropna()

sp500 = sp500.reindex(prices.index).ffill()
sp500_returns = sp500.pct_change().dropna()

# -------------------- Daily returns --------------------
returns = prices.pct_change().dropna()

# -------------------- Cluster portfolios --------------------
cluster_returns = {}
for cluster_id, group in weights.groupby("cluster"):
    assets = group["asset"].tolist()
    w = group.set_index("asset")["weight"]
    # Align weights and returns
    rets_sub = returns[assets]
    port_ret = (rets_sub * w).sum(axis=1)
    cluster_returns[cluster_id] = port_ret

cluster_returns_df = pd.DataFrame(cluster_returns)
cluster_returns_df.to_csv(os.path.join(output_path, "cluster_daily_returns.csv"))

# -------------------- Metrics functions --------------------
def cumulative_return(r):
    return (1 + r).prod() - 1

def annualized_volatility(r):
    return r.std() * np.sqrt(252)

def sharpe_ratio(r, rf=0.02):  # 2% annual RF
    mean_r = r.mean() * 252
    vol = annualized_volatility(r)
    return (mean_r - rf) / vol if vol > 0 else np.nan

def max_drawdown(r):
    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return dd.min()

def alpha_beta(port_r, bench_r):
    # Align both series
    df = pd.concat([port_r, bench_r], axis=1).dropna()
    if len(df) < 2:
        return np.nan, np.nan
    cov = np.cov(df.iloc[:,0], df.iloc[:,1])
    beta = cov[0,1] / cov[1,1]
    alpha = (df.iloc[:,0].mean() - beta * df.iloc[:,1].mean()) * 252
    return alpha, beta

# -------------------- Compute metrics --------------------
metrics = []
for cluster_id in cluster_returns_df.columns:
    r = cluster_returns_df[cluster_id].dropna()
    c_ret = cumulative_return(r)
    vol = annualized_volatility(r)
    sharpe = sharpe_ratio(r)
    mdd = max_drawdown(r)
    alpha, beta = alpha_beta(r, sp500_returns)
    
    metrics.append({
        "Cluster": cluster_id,
        "Cumulative_Return": c_ret,
        "Volatility_Ann": vol,
        "Sharpe_Ratio": sharpe,
        "Max_Drawdown": mdd,
        "Alpha": alpha,
        "Beta": beta
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(os.path.join(output_path, "portfolio_performance_metrics.csv"), index=False)

print("\n✅ Portfolio performance metrics saved:")
print(" - cluster_daily_returns.csv")
print(" - portfolio_performance_metrics.csv")

print("\n📊 Summary:")
print(metrics_df.round(4).to_string(index=False))

# =============================================================================
# Visualizations
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="husl")

# -------------------- Load Data --------------------
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_path = os.path.join(desktop_path, "week7")

returns_path = os.path.join(output_path, "cluster_daily_returns.csv")
metrics_path = os.path.join(output_path, "portfolio_performance_metrics.csv")
weights_path = os.path.join(output_path, "equal_weight_portfolios.csv")

cluster_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
metrics_df = pd.read_csv(metrics_path)
weights_df = pd.read_csv(weights_path)

# -------------------- 1) Portfolio Equity Curves --------------------
plt.figure(figsize=(10,6))
for cluster in cluster_returns.columns:
    equity = (1 + cluster_returns[cluster]).cumprod()
    plt.plot(equity, label=f"Cluster {cluster}")
plt.title("Portfolio Equity Curves per Cluster")
plt.xlabel("Date")
plt.ylabel("Cumulative Growth (× initial)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_path, "plot_equity_curves.png"), dpi=150)
plt.close()

# -------------------- 2) Returns Distribution --------------------
plt.figure(figsize=(10,6))
for cluster in cluster_returns.columns:
    sns.kdeplot(cluster_returns[cluster], label=f"Cluster {cluster}", fill=True, alpha=0.3)
plt.title("Distribution of Daily Returns per Cluster")
plt.xlabel("Daily Return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_path, "plot_returns_distribution.png"), dpi=150)
plt.close()

# -------------------- 3) Risk–Return Scatter --------------------
plt.figure(figsize=(8,6))
plt.scatter(metrics_df["Volatility_Ann"], metrics_df["Cumulative_Return"], s=120)
for _, row in metrics_df.iterrows():
    plt.text(row["Volatility_Ann"]+0.002, row["Cumulative_Return"], f"C{int(row['Cluster'])}", fontsize=9)
plt.title("Risk–Return Scatter (Volatility vs. Cumulative Return)")
plt.xlabel("Volatility (annualized)")
plt.ylabel("Cumulative Return")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "plot_risk_return_scatter.png"), dpi=150)
plt.close()

# -------------------- 4) Cluster Composition Breakdown --------------------
cluster_counts = weights_df.groupby("cluster")["asset"].count().reset_index(name="Count")
plt.figure(figsize=(7,5))
sns.barplot(x="cluster", y="Count", data=cluster_counts, palette="husl")
plt.title("Cluster Composition Breakdown (# of Assets per Cluster)")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Assets")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "plot_cluster_composition.png"), dpi=150)
plt.close()

print("\n✅ Visualization plots saved:")
print(" - plot_equity_curves.png")
print(" - plot_returns_distribution.png")
print(" - plot_risk_return_scatter.png")
print(" - plot_cluster_composition.png")





