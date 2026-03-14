import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re


def calculate_equipment_relativities_v2(inv_path, sev_path):
    print("🚀 启动设备风险对齐引擎 V2...")

    # --- 1. 资产数据提取 ---
    df_inv = pd.read_excel(inv_path, skiprows=2)
    inv_core = df_inv.iloc[0:6, [0, 1, 2, 3]].copy()
    inv_core.columns = ['Equipment_Type', 'Helionis', 'Bayesian', 'Oryn_Delta']

    # 清理库存数值
    for col in ['Helionis', 'Bayesian', 'Oryn_Delta']:
        inv_core[col] = pd.to_numeric(inv_core[col], errors='coerce').fillna(0)
    inv_core['Total_Inventory'] = inv_core[['Helionis', 'Bayesian', 'Oryn_Delta']].sum(axis=1)

    # --- 2. 理赔数据提取 ---
    df_sev = pd.read_excel(sev_path)
    # 核心修正：确保列名正确，如果不叫 'equipment_type'，请根据实际修改
    sev_col = 'equipment_type' if 'equipment_type' in df_sev.columns else df_sev.columns[0]

    # --- 3. 强力对齐函数 ---
    def super_clean(text):
        if pd.isna(text): return ""
        # 转小写，只保留字母和数字，删掉所有空格和特殊字符
        text = re.sub(r'[^a-zA-Z0-9]', '', str(text).lower())
        # 处理常见的单复数或简称 (例如 bores -> bore)
        if text.endswith('s'): text = text[:-1]
        return text

    inv_core['match_key'] = inv_core['Equipment_Type'].apply(super_clean)
    df_sev['match_key'] = df_sev[sev_col].apply(super_clean)

    # 打印匹配预览，方便组长在线调试
    print(f"DEBUG: 库存表匹配键: {inv_core['match_key'].tolist()[:3]}")
    print(f"DEBUG: 理赔表匹配键: {df_sev['match_key'].unique().tolist()[:3]}")

    # --- 4. 汇总与合并 ---
    claim_stats = df_sev.groupby('match_key').agg(
        Claim_Count=('claim_amount', 'count'),
        Total_Loss=('claim_amount', 'sum')
    ).reset_index()

    pricing_df = pd.merge(inv_core, claim_stats, on='match_key', how='left').fillna(0)

    # 如果还是 0，尝试“包含匹配” (针对理赔表里名称更长的情况)
    if pricing_df['Claim_Count'].sum() == 0:
        print("⚠️ 警告：精确匹配失败，启动模糊包含算法...")
        for idx, row in pricing_df.iterrows():
            key = row['match_key']
            matches = df_sev[df_sev['match_key'].str.contains(key, na=False)]
            if not matches.empty:
                pricing_df.at[idx, 'Claim_Count'] = len(matches)
                pricing_df.at[idx, 'Total_Loss'] = matches['claim_amount'].sum()

    # --- 5. 精算计算 ---
    # 计算案均赔款
    pricing_df['Mean_Severity'] = np.where(pricing_df['Claim_Count'] > 0,
                                           pricing_df['Total_Loss'] / pricing_df['Claim_Count'], 0)
    # 计算频率 (理赔次数/库存总数)
    pricing_df['Frequency'] = pricing_df['Claim_Count'] / pricing_df['Total_Inventory'].replace(0, 1)
    # 计算纯保费
    pricing_df['Pure_Premium'] = pricing_df['Frequency'] * pricing_df['Mean_Severity']

    # 计算相对系数 (Relativity)
    # 选取全公司平均纯保费作为基准 (这样最稳健)
    base_premium = pricing_df['Pure_Premium'][pricing_df['Pure_Premium'] > 0].mean()
    if pd.isna(base_premium) or base_premium == 0: base_premium = 1

    pricing_df['Relativity_Multiplier'] = pricing_df['Pure_Premium'] / base_premium
    pricing_df = pricing_df.sort_values('Relativity_Multiplier', ascending=False)

    # --- 6. 可视化 ---
    if pricing_df['Relativity_Multiplier'].sum() > 0:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Relativity_Multiplier', y='Equipment_Type', data=pricing_df, hue='Equipment_Type',
                    palette='viridis', legend=False)
        plt.axvline(x=1.0, color='red', linestyle='--', label='Average Risk')
        for i, v in enumerate(pricing_df['Relativity_Multiplier']):
            plt.text(v, i, f" {v:.2f}x", va='center', fontweight='bold')
        plt.title('Equipment Risk Relativities (Corrected)')
        plt.show()
    else:
        print("❌ 最终匹配依然失败。建议手动检查 EF-sev.xlsx 的第一列内容。")

    return pricing_df


# 调用
inv_path = '/Users/lihonglin/Desktop/EF Risk Index.xlsx'
sev_path = '/Users/lihonglin/Desktop/EF-sev.xlsx'
final_table = calculate_equipment_relativities_v2(inv_path, sev_path)