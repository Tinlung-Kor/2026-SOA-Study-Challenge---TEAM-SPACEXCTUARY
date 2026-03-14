import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. 加载数据 (请确保文件名与你的路径一致)
df_freq = pd.read_excel('/Users/lihonglin/Desktop/cargo-freq.xlsx')
df_sev = pd.read_excel('/Users/lihonglin/Desktop/cargo-sev.xlsx')


def clean_cargo_data(df):
    """
    通用清洗函数：处理 ID 污染、转换数值、填充中位数
    """
    # 修正 cargo_type: 去除类似 _???123 的后缀
    df['cargo_type'] = df['cargo_type'].astype(str).str.split('_').str[0].str.strip()

    # 转换 route_risk 为数值并填充
    df['route_risk'] = pd.to_numeric(df['route_risk'], errors='coerce')
    df['route_risk'] = df['route_risk'].fillna(df['route_risk'].median())

    # 处理其他核心数值列
    numeric_cols = ['pilot_experience', 'vessel_age', 'debris_density', 'exposure', 'cargo_value']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    return df


# 执行清洗
df_f_clean = clean_cargo_data(df_freq)
df_s_clean = clean_cargo_data(df_sev)

# --- 1. 频率模型 (Frequency - Poisson) ---
# 使用 log(exposure) 作为 Offset 是精算建模的标准做法
df_f_clean['ln_exp'] = np.log(df_f_clean['exposure'].clip(lower=1e-6))

# Poisson 默认即为 Log Link，不需要显式声明，从而避开 FutureWarning
freq_mod = smf.glm(
    formula="claim_count ~ C(cargo_type) + route_risk + debris_density + pilot_experience",
    data=df_f_clean,
    family=sm.families.Poisson(),
    offset=df_f_clean['ln_exp']
).fit()

# --- 2. 案均模型 (Severity - Gamma) ---
# 仅筛选金额大于0的样本，并对货值做对数变换（平滑极值）
df_s_pos = df_s_clean[df_s_clean['claim_amount'] > 0].copy()
df_s_pos['ln_cargo_val'] = np.log(df_s_pos['cargo_value'].clip(lower=1))

# 【修正处】：使用大写的 Log() 避开 FutureWarning
sev_mod = smf.glm(
    formula="claim_amount ~ C(cargo_type) + route_risk + ln_cargo_val",
    data=df_s_pos,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()


# --- 3. 提取精算因子表 (Relativities) ---
def get_relativities(model, model_name):
    factors = np.exp(model.params)
    p_vals = model.pvalues
    result = pd.DataFrame({
        'Relativity (exp(coef))': factors,
        'P-Value': p_vals,
        'Significant_5%': p_vals < 0.05
    })
    result.index.name = f'{model_name}_Factor'
    return result


freq_factors = get_relativities(freq_mod, "Frequency")
sev_factors = get_relativities(sev_mod, "Severity")

# 打印结果
print("--- 频率模型因子 (Frequency Factors) ---")
print(freq_factors)
print("\n--- 案均模型因子 (Severity Factors) ---")
print(sev_factors)