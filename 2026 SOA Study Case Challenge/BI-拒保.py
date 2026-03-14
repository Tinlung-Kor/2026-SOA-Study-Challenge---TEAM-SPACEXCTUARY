import pandas as pd
import numpy as np
import scipy.stats as stats

# ---------------------------------------------------------
# 1. 数据加载与清洗
# ---------------------------------------------------------
# 确保读取的是 Severity (损失程度) 文件，因为拒保逻辑主要看损失的极端波动
sev_file = '/Users/lihonglin/Desktop/BI-sev.xlsx'

try:
    df = pd.read_excel(sev_file)
    df.columns = [c.strip().lower() for c in df.columns]  # 清洗列名

    # 确认核心列 claim_amount 是否存在
    if 'claim_amount' not in df.columns:
        raise KeyError(f"未找到赔付金额列。当前列名为: {list(df.columns)}")

    claims = df['claim_amount'].dropna()

    # ---------------------------------------------------------
    # 2. 关键精算指标计算
    # ---------------------------------------------------------
    mean_val = claims.mean()
    std_val = claims.std()
    cv = std_val / mean_val  # 变异系数 (衡量波动性的核心)
    skew = stats.skew(claims)  # 偏度 (衡量右偏/厚尾程度)
    kurt = stats.kurtosis(claims)  # 峰度 (衡量极端值出现的频率)

    # 风险价值 (Value at Risk) 与 尾部期望损失 (Tail VaR)
    p99 = np.percentile(claims, 99)
    p995 = np.percentile(claims, 99.5)
    tvar_99 = claims[claims >= p99].mean()  # 99% TVaR: 最差1%情况下的平均损失

    # ---------------------------------------------------------
    # 3. 输出深度精算分析报告
    # ---------------------------------------------------------
    print("=" * 70)
    print(f"{'BI 营业中断险 - 精算风险评估报告':^70}")
    print("=" * 70)
    print(f"{'指标名称 (Metrics)':<25} | {'数值 (Value)':<20} | {'精算学解读':<20}")
    print("-" * 70)
    print(f"{'平均索赔额 (Mean)':<25} | ${mean_val:,.2f} | 基础赔付成本")
    print(f"{'变异系数 (CV)':<25} | {cv:.4f} | 判定: {'不可保' if cv > 2 else '待定'}")
    print(f"{'偏度 (Skewness)':<25} | {skew:.4f} | 判定: {'极端右偏' if skew > 3 else '一般'}")
    print(f"{'99% VaR (极端损失)':<25} | ${p99:,.2f} | 百年一遇单笔损失")
    print(f"{'99.5% VaR (巨灾损失)':<25} | ${p995:,.2f} | 偿付能力极限测试")
    print(f"{'99% TVaR (尾部均值)':<25} | ${tvar_99:,.2f} | 极端事件平均杀伤力")
    print("-" * 70)

    # ---------------------------------------------------------
    # 4. 拒保论证逻辑 (Evidence for Refusal)
    # ---------------------------------------------------------
    print("\n[ 拒保结论支撑论证 (Key Findings for Refusal) ]")

    # 论据 1：大数定律失效
    if cv > 2.0:
        print(f"1. 波动性超限：CV ({cv:.2f}) 远超商险阈值。这意味着损失分布不服从正态分布，"
              f"大数定律无法通过增加保单数量来稀释风险。")

    # 论据 2：资本成本与保费倒挂
    capital_ratio = tvar_99 / mean_val
    print(f"2. 资本效率：99% TVaR 是平均值的 {capital_ratio:.1f} 倍。为承保该险种，"
          f"公司需计提极高的风险资本，导致保费定价将高到客户无法接受，产生‘逆向选择’。")

    # 论据 3：系统性风险相关性 (Systemic Risk)
    # 简单分析生产负荷与损失的关系
    if 'production_load' in df.columns:
        corr = df['production_load'].corr(df['claim_amount'])
        print(f"3. 关联性风险：生产负荷与损失的相关系数为 {corr:.2f}。一旦星系级停工，"
              f"全线保单将同时出险，不符合‘地理分散’的保险原理。")

    print("\n[ 最终精算判定 ]: 建议拒绝承保 (DECLINE UNDERWRITING)。")
    print("建议替代方案: 仅提供参数化保险 (Parametric Insurance) 或 高免赔额的巨灾垫付协议。")
    print("=" * 70)

except Exception as e:
    print(f"分析失败，请检查数据格式: {e}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. 精算指标配置
# ---------------------------------------------------------
mean_val = 4361165.54
p99_var = 35763084.22
p995_var = 48583094.39
tvar_99 = 55971175.01

# 读取数据
df = pd.read_excel('/Users/lihonglin/Desktop/BI-sev.xlsx')
df.columns = [c.strip().lower() for c in df.columns]
claims = df['claim_amount'].dropna()

# ---------------------------------------------------------
# 2. 设置“清新白底 + 蓝色渐变”绘图风格
# ---------------------------------------------------------
plt.rcParams['figure.facecolor'] = 'white'      # 背景白色
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#E0E0E0'          # 浅灰色网格
plt.rcParams['text.color'] = '#333333'          # 深灰色文字（易读）
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- 图表 A: 损失分布曲线 (KDE) ---
# 使用深蓝色绘制主要曲线
kde_plot = sns.kdeplot(claims, ax=ax1, color='#1864AB', linewidth=2.5, bw_adjust=1.5)
line = kde_plot.get_lines()[0]
x_kde, y_kde = line.get_xdata(), line.get_ydata()

# 填充背景（浅蓝色）
ax1.fill_between(x_kde, 0, y_kde, color='#D0EBFF', alpha=0.4)

# 突出显示 99% VaR 之后的极端风险区域 (使用深蓝色以示警示)
ax1.fill_between(x_kde, 0, y_kde, where=(x_kde >= p99_var),
                 color='#1971C2', alpha=0.7, label='Extreme Tail Risk (>99% VaR)')

# 标注线 - 使用不同深度的蓝色
ax1.axvline(mean_val, color='#4DABF7', linestyle='--', linewidth=2, label=f'Mean: Đ{mean_val/1e6:.1f}M')
ax1.axvline(p99_var, color='#1864AB', linestyle='-', linewidth=2, label=f'99% VaR: Đ{p99_var/1e6:.1f}M')

ax1.set_title("BI Severity Distribution: Quantile Analysis", fontsize=15, fontweight='bold', pad=20)
ax1.set_xlim(0, p995_var * 1.3)
ax1.legend(facecolor='white', edgecolor='#E0E0E0')
ax1.grid(True, linestyle=':', alpha=0.6)

# --- 图表 B: 风险指标对比 (蓝色梯度柱状图) ---
labels = ['Mean', '99% VaR', '99.5% VaR', '99% TVaR']
values = [mean_val, p99_var, p995_var, tvar_99]

# 蓝色渐变色阶（从浅到深：代表风险严重程度递增）
blue_gradient = ['#A5D8FF', '#4DABF7', '#1C7ED6', '#1864AB']

bars = ax2.bar(labels, values, color=blue_gradient, edgecolor='white', linewidth=1.5, width=0.6)
ax2.set_title("Capital Impact: Risk Multiples Comparison", fontsize=15, fontweight='bold', pad=20)
ax2.set_ylabel("Currency (Đ)", fontsize=12)

# 在柱状图上方标注数值 (深蓝色文本)
for bar in bars:
    height = bar.get_height()
    multiplier = height / mean_val
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1e6,
             f'Đ{height/1e6:.1f}M\n({multiplier:.1f}x)',
             ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1864AB')

ax2.grid(axis='y', linestyle=':', alpha=0.6)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
