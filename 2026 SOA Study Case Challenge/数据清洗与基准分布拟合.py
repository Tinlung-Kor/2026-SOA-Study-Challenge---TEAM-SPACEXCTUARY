import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# ==========================================
# 1. Environment Configuration & Path Definitions
# ==========================================
warnings.filterwarnings('ignore')

# Visual style constants
COLOR_GRADIENT = ["#001F3F", "#0074D9", "#89CFF0"]
FIT_LINE_COLOR = "#FF4136"

# Source file mapping for different Lines of Business (LOB)
FILE_PATHS = {
    'EF': {'freq': r'/Users/lihonglin/Desktop/EF-freq.xlsx', 'sev': r'/Users/lihonglin/Desktop/EF-sev.xlsx'},
    'WC': {'freq': r'/Users/lihonglin/Desktop/WC-freq.xlsx', 'sev': r'/Users/lihonglin/Desktop/WC-sev.xlsx'},
    'Cargo': {'freq': r'/Users/lihonglin/Desktop/cargo-freq.xlsx', 'sev': r'/Users/lihonglin/Desktop/cargo-sev.xlsx'}
}

# Global container for storing actuarial metrics across all LOBs
all_results = []


# ==========================================
# 2. Data Processing: Differentiated Mapping Logic
# ==========================================
def get_actuarial_data(lob):
    try:
        # Load frequency and severity datasets
        df_f = pd.read_excel(FILE_PATHS[lob]['freq'], engine='openpyxl')
        df_s = pd.read_excel(FILE_PATHS[lob]['sev'], engine='openpyxl')

        # Standardize column names
        df_f.columns = [str(c).lower().strip() for c in df_f.columns]
        df_s.columns = [str(c).lower().strip() for c in df_s.columns]

        # Filter out negative values (data cleaning)
        df_f = df_f[df_f['claim_count'] >= 0]
        df_s = df_s[df_s['claim_amount'] >= 0]

        if lob == 'Cargo':
            # Logic for Cargo: Grouping by Risk Rating
            risk_col = [c for c in df_f.columns if 'risk' in c][0]

            def map_cargo(r):
                if r in [4, 5]: return 'High Risk (4-5)'
                if r == 3: return 'Medium Risk (3)'
                if r in [1, 2]: return 'Low Risk (1-2)'
                return None

            df_f['group'] = df_f[risk_col].apply(map_cargo)
            df_s['group'] = df_s[risk_col].apply(map_cargo)
            group_order = ['High Risk (4-5)', 'Medium Risk (3)', 'Low Risk (1-2)']
        else:
            # Logic for EF/WC: Grouping by System/Cluster names
            sys_col = [c for c in df_f.columns if 'system' in c or 'cluster' in c][0]

            def map_sys(s):
                s = str(s).upper()
                if 'ZETA' in s: return 'Zeta System'
                if 'EPSILON' in s: return 'Epsilon System'
                if 'HELIONIS' in s: return 'Helionis Cluster'
                return None

            df_f['group'] = df_f[sys_col].apply(map_sys)
            df_s['group'] = df_s[sys_col].apply(map_sys)
            group_order = ['Zeta System', 'Epsilon System', 'Helionis Cluster']

        return df_f, df_s, group_order
    except Exception as e:
        print(f"❌ Failed to load {lob}: {e}")
        return None, None, None


# ==========================================
# 3. Visualization & Metric Extraction Engine
# ==========================================
def run_calibration_report(lob):
    df_f, df_s, categories = get_actuarial_data(lob)
    if df_f is None: return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=100)
    fig.patch.set_facecolor('#FFFFFF')

    for i, cat in enumerate(categories):
        color = COLOR_GRADIENT[i]

        # --- Metric Initialization ---
        res = {"LOB": lob, "Category": cat}

        # --- Row 0: Frequency Analysis ---
        ax_f = axes[0, i]
        sub_f = df_f[df_f['group'] == cat]['claim_count'].dropna()

        if not sub_f.empty:
            counts = sub_f.value_counts().reindex(range(5), fill_value=0)
            ax_f.bar(counts.index, counts.values, color=color, alpha=0.8, edgecolor='black', width=0.6)
            ax_f.set_yscale('log')
            ax_f.set_ylim(1, counts.max() * 10)
            ax_f.set_xticks(range(5))
            ax_f.set_title(f"{cat}\nFrequency Distribution", fontsize=12, fontweight='bold', color=color)

            # Calculate and store frequency metrics (Mean and Variance)
            mu_f, var_f = sub_f.mean(), sub_f.var()
            # Distribution selection based on dispersion: Poisson (Var ≈ Mean) vs Neg-Binomial (Var > Mean)
            res.update({"E_N": round(mu_f, 4), "Var_N": round(var_f, 4),
                        "Freq_Dist": "Poisson" if var_f <= mu_f * 1.05 else "Neg-Binomial"})

            info_f = rf"$E[N]={mu_f:.3f}$" + "\n" + rf"$Var[N]={var_f:.3f}$"
            ax_f.text(0.95, 0.95, info_f, transform=ax_f.transAxes, va='top', ha='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

        # --- Row 1: Severity Analysis ---
        ax_s = axes[1, i]
        sub_s = df_s[df_s['group'] == cat]['claim_amount'].dropna()
        sub_s = sub_s[sub_s > 0]

        if len(sub_s) > 2:
            # Fit Log-Normal distribution
            shape, loc, scale = stats.lognorm.fit(sub_s, floc=0)
            mu_ln, sig_ln = np.log(scale), shape

            # Calculate theoretical expected severity E[X]
            exp_x = np.exp(mu_ln + (sig_ln ** 2) / 2)
            res.update({"Mu_ln": round(mu_ln, 4), "Sigma_ln": round(sig_ln, 4), "E_X": round(exp_x, 2)})

            # Plot histogram (capped at 95th percentile for better visibility)
            limit = np.percentile(sub_s, 95)
            sns.histplot(sub_s[sub_s < limit], bins=20, stat="density", ax=ax_s, color=color, alpha=0.3)

            # Plot the Log-Normal probability density function (PDF)
            x_line = np.linspace(0.1, limit, 200)
            ax_s.plot(x_line, stats.lognorm.pdf(x_line, shape, loc, scale), color=FIT_LINE_COLOR, lw=2.5)
            ax_s.set_title(f"{cat}\nSeverity (Log-Normal)", fontsize=12, fontweight='bold', color=color)

            info_s = rf"$\mu_{{ln}}={mu_ln:.2f}$" + "\n" + rf"$\sigma_{{ln}}={sig_ln:.2f}$"
            ax_s.text(0.95, 0.95, info_s, transform=ax_s.transAxes, va='top', ha='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

        all_results.append(res)

    plt.suptitle(f"Actuarial Baseline Calibration: {lob}", fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout(h_pad=3)
    plt.show()


# ==========================================
# 4. Execution & Export
# ==========================================
for lob_name in ['Cargo', 'EF', 'WC']:
    print(f"📑 Analyzing {lob_name}...")
    run_calibration_report(lob_name)

# Export results to CSV
df_final = pd.DataFrame(all_results)
export_path = r'/Users/lihonglin/Desktop/calibration_results.csv'
df_final.to_csv(export_path, index=False, encoding='utf-8-sig')

print(f"\n✅ Actuarial metrics successfully exported to: {export_path}")
print(df_final[['LOB', 'Category', 'E_N', 'Var_N', 'E_X']])  # Preview key metrics