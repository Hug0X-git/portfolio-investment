# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO INVESTMENT SYSTEM — Google Colab
# Shrinkage μ · Ledoit-Wolf Σ · Max Sharpe · Décomposition du risque
# ═══════════════════════════════════════════════════════════════════════════════
# USAGE : Copiez chaque cellule séparément dans Colab, ou exécutez en séquence.
# ═══════════════════════════════════════════════════════════════════════════════


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELLULE 1 — Installation des dépendances                                  │
# └─────────────────────────────────────────────────────────────────────────────┘

# !pip install yfinance numpy pandas matplotlib seaborn scipy -q


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELLULE 2 — Questionnaire & Configuration                                  │
# └─────────────────────────────────────────────────────────────────────────────┘

import numpy as np
import pandas as pd

# ── Questionnaire investisseur ───────────────────────────────────────────────
# Répondez aux 5 questions (échelles indiquées), votre profil sera calculé automatiquement.

Q_TOLERANCE_RISQUE    = 6   # 1 (très faible) → 10 (très élevée)    — poids 35%
Q_HORIZON_ANNEES      = 10  # 1 an            → 40 ans               — poids 25%
Q_STABILITE_REVENUS   = 7   # 1 (instables)   → 10 (très stables)   — poids 20%
Q_NIVEAU_PATRIMOINE   = 5   # 1 (faible)      → 10 (élevé)          — poids 10%
Q_CONNAISSANCES       = 6   # 1 (débutant)    → 10 (expert)         — poids 10%

def calculer_score():
    rt_s   = (Q_TOLERANCE_RISQUE  - 1) / 9 * 100
    hz_s   = min(Q_HORIZON_ANNEES / 40 * 100, 100)
    inc_s  = (Q_STABILITE_REVENUS - 1) / 9 * 100
    wlth_s = (Q_NIVEAU_PATRIMOINE - 1) / 9 * 100
    know_s = (Q_CONNAISSANCES     - 1) / 9 * 100
    return 0.35*rt_s + 0.25*hz_s + 0.20*inc_s + 0.10*wlth_s + 0.10*know_s

def profil_depuis_score(score):
    if   score <= 20: return 'Very Conservative'
    elif score <= 40: return 'Conservative'
    elif score <= 60: return 'Balanced'
    elif score <= 80: return 'Growth'
    else:             return 'Aggressive'

SCORE = calculer_score()
PROFIL_AUTO = profil_depuis_score(SCORE)

print(f"Score de risque calculé : {SCORE:.1f} / 100")
print(f"Profil recommandé       : {PROFIL_AUTO}")
print()

# ── Profils disponibles ──────────────────────────────────────────────────────
PROFILES = {
    'Very Conservative': {
        'lambda_': 8, 'vol': 0.05, 'max_eq': 0.30, 'crypto': False,
        'budgets': {'equity': 0.20, 'fixed_income': 0.65, 'commodity': 0.10, 'real_estate': 0.05, 'crypto': 0.00},
    },
    'Conservative': {
        'lambda_': 6, 'vol': 0.08, 'max_eq': 0.50, 'crypto': False,
        'budgets': {'equity': 0.35, 'fixed_income': 0.50, 'commodity': 0.10, 'real_estate': 0.05, 'crypto': 0.00},
    },
    'Balanced': {
        'lambda_': 4, 'vol': 0.12, 'max_eq': 0.70, 'crypto': True,
        'budgets': {'equity': 0.50, 'fixed_income': 0.30, 'commodity': 0.10, 'real_estate': 0.05, 'crypto': 0.05},
    },
    'Growth': {
        'lambda_': 2.5, 'vol': 0.18, 'max_eq': 0.85, 'crypto': True,
        'budgets': {'equity': 0.65, 'fixed_income': 0.15, 'commodity': 0.08, 'real_estate': 0.05, 'crypto': 0.07},
    },
    'Aggressive': {
        'lambda_': 1.5, 'vol': 0.25, 'max_eq': 1.00, 'crypto': True,
        'budgets': {'equity': 0.70, 'fixed_income': 0.05, 'commodity': 0.05, 'real_estate': 0.05, 'crypto': 0.15},
    },
}

# ── Paramètres ───────────────────────────────────────────────────────────────
SELECTED_PROFILE     = PROFIL_AUTO      # ou forcez manuellement : 'Balanced'
RISK_FREE_RATE       = 0.045            # taux sans risque annualisé (4.5%)
ESTIMATION_WINDOW_Y  = 5               # années d'historique pour estimation
SHRINKAGE_ALPHA      = 0.20            # α pour James-Stein : μ̂ = (1−α)·μ_hist + α·μ_grand_mean
MAX_CORRELATION      = 0.90            # filtre corrélation max
MIN_HISTORY_YEARS    = 3               # historique minimum requis par actif
MIN_ANNUAL_VOL       = 0.01            # volatilité annuelle minimale

# ── Univers d'actifs ─────────────────────────────────────────────────────────
ASSET_UNIVERSE = {
    'equity':       ['SPY', 'QQQ', 'VGK', 'EEM', 'VWO'],
    'fixed_income': ['AGG', 'TLT', 'IEF', 'LQD', 'HYG', 'EMB'],
    'commodity':    ['GLD', 'SLV', 'DJP'],
    'real_estate':  ['VNQ', 'VNQI'],
    'crypto':       ['BTC-USD', 'ETH-USD'],
}

CLASS_COLORS = {
    'equity': '#4a90d9', 'fixed_income': '#5cb85c',
    'commodity': '#e8a838', 'real_estate': '#9b59b6', 'crypto': '#f7931a',
}
CLASS_LABELS = {
    'equity': 'Actions', 'fixed_income': 'Obligations',
    'commodity': 'Matières premières', 'real_estate': 'Immobilier', 'crypto': 'Crypto',
}

profile = PROFILES[SELECTED_PROFILE]

# Affichage du profil sélectionné
print(f"Profil actif     : {SELECTED_PROFILE}")
print(f"Aversion (λ)     : {profile['lambda_']}")
print(f"Vol. cible       : {profile['vol']*100:.0f}%")
print(f"Crypto inclus    : {'Oui' if profile['crypto'] else 'Non'}")
print()
print("Budgets de risque par classe :")
for cls, b in profile['budgets'].items():
    bar = '█' * int(b * 40)
    print(f"  {CLASS_LABELS[cls]:<20} {bar:<40} {b*100:.0f}%")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELLULE 3 — Téléchargement des données Yahoo Finance                       │
# └─────────────────────────────────────────────────────────────────────────────┘

import yfinance as yf

def get_tickers():
    tickers = (
        ASSET_UNIVERSE['equity'] +
        ASSET_UNIVERSE['fixed_income'] +
        ASSET_UNIVERSE['commodity'] +
        ASSET_UNIVERSE['real_estate']
    )
    if profile['crypto']:
        tickers += ASSET_UNIVERSE['crypto']
    return list(dict.fromkeys(tickers))  # déduplique en gardant l'ordre

ALL_TICKERS = get_tickers()
print(f"Téléchargement de {len(ALL_TICKERS)} actifs sur {ESTIMATION_WINDOW_Y + 2} ans…")

raw = yf.download(
    ALL_TICKERS,
    period=f"{ESTIMATION_WINDOW_Y + 2}y",
    auto_adjust=True,
    progress=False,
)['Close']

# Filtrage : supprimer les actifs avec trop peu d'historique
min_obs = int(MIN_HISTORY_YEARS * 252)
raw = raw.dropna(axis=1, thresh=min_obs)
raw = raw.ffill().dropna()

print(f"Actifs retenus   : {list(raw.columns)}")
print(f"Période          : {raw.index[0].date()} → {raw.index[-1].date()} ({len(raw)} jours)")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELLULE 4 — Mathématiques du portefeuille                                  │
# └─────────────────────────────────────────────────────────────────────────────┘

# ── Rendements journaliers ───────────────────────────────────────────────────
returns = raw.pct_change().dropna()

# Fenêtre d'estimation
est_days = min(int(ESTIMATION_WINDOW_Y * 252), len(returns))
ret_est = returns.iloc[-est_days:]
tickers = list(ret_est.columns)
n = len(tickers)

print(f"Fenêtre d'estimation : {est_days} jours | {n} actifs")


# ── Covariance Ledoit-Wolf (cible identité) ──────────────────────────────────
def ledoit_wolf_cov(R, ann=252):
    """Shrinkage vers matrice identité scalée (Oracle Approximating Shrinkage)."""
    T, p = R.shape
    S = R.cov().values * ann                   # covariance sample annualisée
    tr_S = np.trace(S)
    mu_hat = tr_S / p                          # cible : μ̂ · I

    F = mu_hat * np.eye(p)                     # target matrix
    num   = np.sum(S**2) if False else 0
    denom = np.sum((S - F)**2)
    # estimation du coefficient optimal
    num = (p + 2) / T * np.sum(np.diag(S)**2)
    alpha = np.clip(num / (denom + 1e-10), 0, 1)

    return (1 - alpha) * S + alpha * F, alpha

cov, lw_alpha = ledoit_wolf_cov(ret_est)
print(f"Ledoit-Wolf α*  = {lw_alpha:.4f}  (0 = pur historique, 1 = identité)")


# ── Rendements attendus James-Stein ─────────────────────────────────────────
def shrinkage_mu(R, ann=252, alpha=0.20):
    """μ̂_i = (1−α)·μ_hist_i + α·μ_grand_mean"""
    mu_hist   = R.mean().values * ann
    grand_mean = mu_hist.mean()
    return (1 - alpha) * mu_hist + alpha * grand_mean

mu = shrinkage_mu(ret_est, ann=252, alpha=SHRINKAGE_ALPHA)
print(f"\nRendements attendus (shrinkage α={SHRINKAGE_ALPHA}) :")
for t, m in zip(tickers, mu):
    sign = '+' if m >= 0 else ''
    print(f"  {t:<12} {sign}{m*100:.2f}%")


# ── Optimisation Max Sharpe (gradient ascent) ────────────────────────────────
def proj_simplex(v, lo=0.001, hi=0.50):
    """Projection sur le simplexe [lo, hi]."""
    w = np.clip(v, lo, hi)
    return w / w.sum()

def max_sharpe_weights(mu, cov, rf, n_iter=3000):
    """Remonte le ratio de Sharpe par gradient ascent."""
    n = len(mu)
    w = np.ones(n) / n
    lr = 0.03

    for _ in range(n_iter):
        Sw    = cov @ w
        p_var = max(w @ Sw, 1e-10)
        p_vol = np.sqrt(p_var)
        p_ret = w @ mu
        sh    = (p_ret - rf) / p_vol

        grad  = ((mu - rf) - sh * Sw) / p_vol
        w_new = proj_simplex(w + lr * grad)

        p_var2 = max(w_new @ (cov @ w_new), 1e-10)
        p_ret2 = w_new @ mu
        sh2    = (p_ret2 - rf) / np.sqrt(p_var2)

        if sh2 > sh:
            w  = w_new
            lr = min(lr * 1.05, 0.2)
        else:
            lr *= 0.6

        if lr < 1e-7:
            break

    return w

rf_daily  = RISK_FREE_RATE / 252
weights   = max_sharpe_weights(mu, cov, RISK_FREE_RATE)

p_ret     = weights @ mu
p_vol     = np.sqrt(weights @ cov @ weights)
p_sharpe  = (p_ret - RISK_FREE_RATE) / max(p_vol, 1e-8)

print(f"\n── Résultats de l'optimisation ──────────────")
print(f"Rendement annuel  (μ) : {p_ret*100:.2f}%")
print(f"Volatilité annuelle(σ): {p_vol*100:.2f}%")
print(f"Ratio de Sharpe       : {p_sharpe:.3f}")


# ── Décomposition du risque (MRC / TRC / %RC) ────────────────────────────────
Sw    = cov @ weights
p_var = max(weights @ Sw, 1e-10)

MRC = Sw / np.sqrt(p_var)                # Marginal Risk Contribution
TRC = weights * MRC                       # Total Risk Contribution
RC  = TRC / np.sqrt(p_var)               # Risk Contribution (fraction)

print(f"\n── Décomposition du risque ──────────────────")
print(f"{'Ticker':<12} {'Poids':>7} {'MRC':>8} {'TRC':>8} {'%RC':>8}")
print("─" * 48)
idx_sorted = np.argsort(-weights)
for i in idx_sorted:
    t = tickers[i]
    print(f"{t:<12} {weights[i]*100:>6.1f}%  {MRC[i]*100:>7.3f}%  {TRC[i]*100:>7.3f}%  {RC[i]*100:>6.1f}%")


# ── Max Drawdown ─────────────────────────────────────────────────────────────
port_daily_ret = ret_est.values @ weights
nav = np.cumprod(1 + port_daily_ret)
peak = np.maximum.accumulate(nav)
drawdowns = (nav - peak) / peak
max_dd = drawdowns.min()

print(f"\nMax Drawdown          : {max_dd*100:.2f}%")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELLULE 5 — Visualisations                                                 │
# └─────────────────────────────────────────────────────────────────────────────┘

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

def get_class(ticker):
    for cls, tks in ASSET_UNIVERSE.items():
        if ticker in tks:
            return cls
    return 'equity'

colors = [CLASS_COLORS[get_class(t)] for t in tickers]

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 1. Allocation (donut) ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
wedges, texts, autotexts = ax1.pie(
    weights, labels=tickers, colors=colors,
    autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
    pctdistance=0.75, startangle=90,
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=1.5)
)
for t in autotexts: t.set_fontsize(8)
for t in texts:     t.set_fontsize(7.5)
ax1.set_title('Allocation du portefeuille', fontsize=11, fontweight='bold', pad=12)

# ── 2. Contributions au risque vs budgets ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1:])
budgets_arr = np.array([
    profile['budgets'].get(get_class(t), 0) /
    max(sum(1 for x in tickers if get_class(x) == get_class(t)), 1)
    for t in tickers
])
x = np.arange(n)
w_bar = 0.35
ax2.bar(x - w_bar/2, RC * 100,       w_bar, label='RC réelle',    color=colors, alpha=0.9, zorder=3)
ax2.bar(x + w_bar/2, budgets_arr*100, w_bar, label='Budget cible', color=colors, alpha=0.35, edgecolor=colors, linewidth=1, zorder=3)
ax2.set_xticks(x); ax2.set_xticklabels(tickers, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Contribution au risque (%)'); ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax2.set_title('Contributions au risque — réelle vs budget cible', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3, zorder=0); ax2.set_facecolor('#fafafa')

# ── 3. KPI summary ───────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')
kpis = [
    ('Rendement annuel (μ)', f"{p_ret*100:+.2f}%",  '#059669' if p_ret >= 0 else '#dc2626'),
    ('Volatilité annuelle (σ)', f"{p_vol*100:.2f}%", '#000'),
    ('Ratio de Sharpe',       f"{p_sharpe:.3f}",     '#059669' if p_sharpe >= 1 else '#555'),
    ('Max Drawdown',          f"{max_dd*100:.2f}%",  '#dc2626'),
    ('Profil',                SELECTED_PROFILE,       '#000'),
    ('Score questionnaire',   f"{SCORE:.1f} / 100",  '#000'),
]
for i, (label, val, color) in enumerate(kpis):
    y = 1 - i * 0.16
    ax3.text(0, y,      label, transform=ax3.transAxes, fontsize=9,  color='#888')
    ax3.text(1, y,      val,   transform=ax3.transAxes, fontsize=11, color=color,
             fontweight='bold', ha='right')
ax3.set_title('Indicateurs clés', fontsize=11, fontweight='bold')

# ── 4. Poids (barres horizontales) ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1:])
idx_s = np.argsort(-weights)
tickers_s = [tickers[i] for i in idx_s]
weights_s  = [weights[i] * 100 for i in idx_s]
colors_s   = [colors[i] for i in idx_s]
bars = ax4.barh(range(n), weights_s, color=colors_s, alpha=0.85, edgecolor='white', linewidth=0.5)
ax4.set_yticks(range(n)); ax4.set_yticklabels(tickers_s, fontsize=9)
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax4.set_title('Poids optimaux (Max Sharpe)', fontsize=11, fontweight='bold')
for bar, val in zip(bars, weights_s):
    ax4.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=8)
ax4.grid(axis='x', alpha=0.3); ax4.set_facecolor('#fafafa'); ax4.invert_yaxis()

# ── 5. Matrice de corrélation (heatmap) ─────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :])
corr_matrix = ret_est.corr()
mask = np.zeros_like(corr_matrix, dtype=bool)
np.fill_diagonal(mask, False)
sns.heatmap(
    corr_matrix, ax=ax5, annot=True, fmt='.2f', annot_kws={'size': 7.5},
    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor='white',
    cbar_kws={'shrink': 0.6, 'label': 'Corrélation'},
)
ax5.set_title('Matrice de corrélation', fontsize=11, fontweight='bold')
ax5.tick_params(axis='x', rotation=45, labelsize=8)
ax5.tick_params(axis='y', rotation=0,  labelsize=8)

# ── Légende classes ──────────────────────────────────────────────────────────
handles = [mpatches.Patch(color=c, label=CLASS_LABELS[k]) for k, c in CLASS_COLORS.items()
           if any(get_class(t) == k for t in tickers)]
fig.legend(handles=handles, loc='lower center', ncol=len(handles),
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))

plt.suptitle(
    f'Portfolio Investment System — Profil : {SELECTED_PROFILE}  '
    f'(λ={profile["lambda_"]} · α_shrinkage={SHRINKAGE_ALPHA} · Ledoit-Wolf Σ)',
    fontsize=13, fontweight='bold', y=1.01
)
plt.savefig('portfolio_results.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("✅  Graphique sauvegardé → portfolio_results.png")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CELLULE 6 — Export CSV des résultats                                       │
# └─────────────────────────────────────────────────────────────────────────────┘

results_df = pd.DataFrame({
    'Ticker':             tickers,
    'Classe':             [CLASS_LABELS[get_class(t)] for t in tickers],
    'Poids (%)':          np.round(weights * 100, 2),
    'Rendement attendu (%)': np.round(mu * 100, 2),
    'MRC (%)':            np.round(MRC * 100, 4),
    'TRC (%)':            np.round(TRC * 100, 4),
    'RC (%)':             np.round(RC * 100, 2),
    'Budget cible (%)':   np.round(budgets_arr * 100, 2),
}).sort_values('Poids (%)', ascending=False).reset_index(drop=True)

print(results_df.to_string(index=False))
results_df.to_csv('portfolio_weights.csv', index=False)
print("\n✅  Résultats sauvegardés → portfolio_weights.csv")
