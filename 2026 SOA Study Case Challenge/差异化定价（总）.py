import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 图表全局美化配置：深空宇宙主题 (Cosmic Blue Theme)
# ==========================================
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.sans-serif'] = ['Arial']
# 设置背景为极淡的蓝灰色，保持数据清晰，线条和字体使用深空蓝
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#F4F7FB",
    "grid.color": "#DCE4EE",
    "text.color": "#0B2545",
    "axes.labelcolor": "#0B2545",
    "xtick.color": "#0B2545",
    "ytick.color": "#0B2545"
})


def cosmic_pricing_engine():
    base_premiums = {'Cargo': 2463365.65, 'EF': 7058.84, 'WC': 83.90}

    assumptions = {
        'Helionis': {'Env_Gravity': 1.00, 'Eq_Age': 1.40, 'Eq_Maint': 1.00, 'Eq_Usage': 1.00, 'Occupational': 1.00},
        'Bayesian': {'Env_Gravity': 1.20, 'Eq_Age': 1.15, 'Eq_Maint': 0.80, 'Eq_Usage': 0.84, 'Occupational': 1.10},
        'Oryn_Delta': {'Env_Gravity': 1.60, 'Eq_Age': 0.85, 'Eq_Maint': 0.67, 'Eq_Usage': 0.79, 'Occupational': 1.50}
    }

    records = []
    for sys, factors in assumptions.items():
        ef_rel = factors['Env_Gravity'] * factors['Eq_Age'] * factors['Eq_Maint'] * factors['Eq_Usage']
        cargo_rel = factors['Env_Gravity']
        wc_rel = factors['Env_Gravity'] * factors['Occupational']

        records.append({
            'System': sys,
            'Env_Factor': factors['Env_Gravity'], 'Age_Factor': factors['Eq_Age'],
            'Maint_Factor': factors['Eq_Maint'], 'Usage_Factor': factors['Eq_Usage'],
            'Occ_Factor': factors['Occupational'],
            'EF_Relativity': ef_rel, 'Cargo_Relativity': cargo_rel, 'WC_Relativity': wc_rel,
            'EF_Premium': base_premiums['EF'] * ef_rel,
            'Cargo_Premium': base_premiums['Cargo'] * cargo_rel,
            'WC_Premium': base_premiums['WC'] * wc_rel
        })

    df_results = pd.DataFrame(records).set_index('System')

    # ==========================================
    # 可视化绘制
    # ==========================================
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#FFFFFF')  # 外框纯白

    # 颜色定义：深空蓝渐变
    color_deep = "#082567"  # 深邃星空 (用于重资产 Cargo)
    color_mid = "#3182CE"  # 蔚蓝 (用于 EF)
    color_light = "#90CDF4"  # 浅蓝/星光 (用于 WC)

    # --- 子图 1: 风险因子热力图 (蓝色渐变) ---
    ax1 = plt.subplot(2, 2, (1, 2))
    factor_cols = ['Env_Factor', 'Age_Factor', 'Maint_Factor', 'Usage_Factor', 'Occ_Factor']
    sns.heatmap(df_results[factor_cols].T, annot=True, cmap="Blues", center=1.0,
                fmt=".2f", linewidths=1.5, ax=ax1, cbar_kws={'label': 'Risk Multiplier'})
    ax1.set_title('Cosmic Risk Profiling Heatmap by Planetary System', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('')

    # --- 子图 2: 三大险种相对风险系数 ---
    ax2 = plt.subplot(2, 2, 3)
    rel_cols = ['EF_Relativity', 'Cargo_Relativity', 'WC_Relativity']
    df_results[rel_cols].plot(kind='bar', ax=ax2, color=[color_mid, color_deep, color_light], edgecolor='white')
    ax2.axhline(1.0, color='#E53E3E', linestyle='--', linewidth=2, label='Baseline (1.0)')
    ax2.set_title('Aggregate Risk Relativities', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Multiplier vs. Base')
    ax2.tick_params(axis='x', rotation=0)
    ax2.legend(['Baseline (1.0)', 'EF', 'Cargo', 'WC'])

    # --- 子图 3: 终极双坐标轴保费对比 (Premium Comparison) ---
    ax3 = plt.subplot(2, 2, 4)
    x = np.arange(len(df_results.index))
    width = 0.25  # 柱子宽度

    # 左坐标轴 (主轴): 绘制较小的 EF 和 WC 险种
    bar1 = ax3.bar(x - width, df_results['EF_Premium'], width, label='EF Premium', color=color_mid, edgecolor='white')
    bar2 = ax3.bar(x, df_results['WC_Premium'], width, label='WC Premium', color=color_light, edgecolor='white')
    ax3.set_ylabel('Premium: EF & WC (Đ)', color=color_mid, fontsize=12, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor=color_mid)

    # 右坐标轴 (副轴): 绘制巨额的 Cargo 险种
    ax3_twin = ax3.twinx()
    bar3 = ax3_twin.bar(x + width, df_results['Cargo_Premium'], width, label='Cargo Premium (Right Axis)',
                        color=color_deep, edgecolor='white')
    ax3_twin.set_ylabel('Premium: Cargo (Đ)', color=color_deep, fontsize=12, fontweight='bold')
    ax3_twin.tick_params(axis='y', labelcolor=color_deep)

    # 刻度与图例处理
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_results.index, fontweight='bold')
    ax3.set_title('Final Premium Comparison (Dual Axis)', fontsize=14, fontweight='bold')

    # 合并主副坐标轴的图例
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout(pad=3.0)
    plt.show()

    return df_results


# 运行代码
final_df = cosmic_pricing_engine()