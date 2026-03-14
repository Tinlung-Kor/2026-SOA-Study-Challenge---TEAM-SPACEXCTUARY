import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import re

# 1. 加载数据
df_wf = pd.read_excel('/Users/lihonglin/Desktop/WC-freq.xlsx')
df_ws = pd.read_excel('/Users/lihonglin/Desktop/WC-sev.xlsx')


def clean_wc_refined(df):
    # 【关键修订】使用正则清洗掉 _??? 后缀，统一职业和伤病分类
    def remove_suffix(text):
        if pd.isna(text): return "Unknown"
        return re.split(r'_\?', str(text))[0].strip()

    if 'occupation' in df.columns:
        df['occupation'] = df['occupation'].apply(remove_suffix)
    if 'injury_type' in df.columns:
        df['injury_type'] = df['injury_type'].apply(remove_suffix)

    # 数值型列清洗与中位数填充
    num_cols = ['experience_yrs', 'psych_stress_index', 'gravity_level',
                'safety_training_index', 'exposure', 'hours_per_week']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    return df


# 执行深度清洗
df_wf_clean = clean_wc_refined(df_wf)
df_ws_clean = clean_wc_refined(df_ws)

# --- 2. 频率模型 (Frequency - Poisson) ---
# Exposure 对数化作为 Offset
df_wf_clean['ln_exp'] = np.log(df_wf_clean['exposure'].clip(lower=1e-6))

# 建立频率模型：加入核心预测因子
freq_formula = "claim_count ~ C(occupation) + safety_training_index + psych_stress_index + gravity_level + experience_yrs"
wc_freq_mod = smf.glm(
    formula=freq_formula,
    data=df_wf_clean,
    family=sm.families.Poisson(),
    offset=df_wf_clean['ln_exp']
).fit()

# --- 3. 案均模型 (Severity - Gamma) ---
# 仅针对有赔付的记录
df_ws_pos = df_ws_clean[df_ws_clean['claim_amount'] > 0].copy()

# 建立案均模型：加入岗位、伤病类型及环境因素
sev_formula = "claim_amount ~ C(occupation) + C(injury_type) + gravity_level + experience_yrs"
wc_sev_mod = smf.glm(
    formula=sev_formula,
    data=df_ws_pos,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()


# --- 4. 提取精算因子表 (Relativities) ---
def get_clean_factors(model, name):
    df_res = pd.DataFrame({
        'Relativity (exp(coef))': np.exp(model.params),
        'P-Value': model.pvalues,
        'Significant_5%': model.pvalues < 0.05
    })
    df_res.index.name = f'WC_{name}_Variable'
    return df_res


# 输出结果
wc_f_factors = get_clean_factors(wc_freq_mod, "Freq")
wc_s_factors = get_clean_factors(wc_sev_mod, "Sev")

print("--- 修订后：WC 频率因子表 ---")
print(wc_f_factors)
print("\n--- 修订后：WC 案均因子表 ---")
print(wc_s_factors)