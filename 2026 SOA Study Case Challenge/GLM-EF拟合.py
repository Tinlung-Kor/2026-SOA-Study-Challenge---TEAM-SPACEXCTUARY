import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
import re

warnings.filterwarnings('ignore')
#拟合频率模型
#注：目标变量是索赔次数。由于之前的基准测试显示数据存在过度离散（Var[N]>E[N]），
# 理想情况下应选择负二项分布或准泊松分布。在此基准代码中，我们采用 Poisson 配合对数连接函数，
# 并在后期通过偏差分析进行离散度调整。连接函数选用log-link函数进行拟合，确保预测频率为正，
# 并构建乘法费率结构，同时加入ln（exposure）作为偏移量作为风险暴露考量。
#拟合程度模型
#注：注：目标变量是单次索赔金额，仅针对发生过索赔的记录（claim_amount > 0 和 claim_count > 0）进行训练。
# 由于数据集整体呈现“长尾”且恒大于 0 的偏态损失特征，因此分布族选用Gamma分布。连接函数同样选用log-link，
# 用于构建乘法结构，并且便于限制极值。
# ==========================================
# 1. 数据加载与鲁棒清洗
# ==========================================
def load_and_prep_data(freq_path, sev_path):
    try:
        df_freq = pd.read_excel(freq_path) if freq_path.endswith('.xlsx') else pd.read_csv(freq_path)
        df_sev = pd.read_excel(sev_path) if sev_path.endswith('.xlsx') else pd.read_csv(sev_path)
    except Exception as e:
        print(f"❌ 数据读取失败: {e}")
        return None

    df_freq.columns = [str(c).lower().strip() for c in df_freq.columns]
    df_sev.columns = [str(c).lower().strip() for c in df_sev.columns]

    if 'solar_system' in df_freq.columns:
        def clean_sys(s):
            s = str(s).upper()
            if 'ZETA' in s: return 'Zeta'
            if 'EPSILON' in s: return 'Epsilon'
            if 'HELIONIS' in s: return 'Helionis Cluster'
            return 'Unknown'

        df_freq['solar_system'] = df_freq['solar_system'].apply(clean_sys)
        df_freq = df_freq[df_freq['solar_system'] != 'Unknown']

    # 【优化】：确保 exposure 不仅大于 0，且不至于小到触发数值下溢
    # 过滤掉暴露量极小的样本（例如小于 0.001 年的保单，通常为噪声）
    df_freq = df_freq[df_freq['exposure'] > 1e-4]

    sev_agg = df_sev.groupby('policy_id')['claim_amount'].sum().reset_index()
    df_model = pd.merge(df_freq, sev_agg, on='policy_id', how='left')
    df_model['claim_amount'] = df_model['claim_amount'].fillna(0)

    # 检查并处理 log_exposure
    df_model['log_exposure'] = np.log(df_model['exposure'])

    return df_model


# ==========================================
# 2. 增强版的频率模型 (解决拟合崩溃问题)
# ==========================================
def fit_frequency_model(df, formula_freq, dist='NegativeBinomial'):
    print("\n" + "=" * 80)
    print(f"🚀 TRAINING FREQUENCY MODEL ({dist.upper()})")
    print("=" * 80)

    # 数值预处理
    df['claim_count'] = df['claim_count'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['claim_count'] = df['claim_count'].round().astype(int)

    # 提取自变量进行缺失值和极值处理
    vars_in_formula = re.findall(r'\b\w+\b', formula_freq.split('~')[1])
    feature_cols = [v for v in vars_in_formula if v in df.columns]

    # 剔除无效数据
    df_clean = df.dropna(subset=['claim_count', 'log_exposure'] + feature_cols).copy()

    # 【新增】：对连续变量进行极值盖帽处理（防止系数爆炸导致 NaN）
    for col in feature_cols:
        if df_clean[col].dtype in [np.float64, np.int64]:
            upper_limit = df_clean[col].quantile(0.999)  # 剔除 0.1% 的极端离群值
            df_clean[col] = df_clean[col].clip(upper=upper_limit)

    # 选择分布族
    if dist == 'Poisson':
        family = sm.families.Poisson(link=sm.families.links.log())
    else:
        family = sm.families.NegativeBinomial(link=sm.families.links.log())

    try:
        # 【关键修复】：
        # 1. 使用 method='newton' 或 'bfgs' 代替默认的 IRLS，这通常能绕过“first guess nan”的问题
        # 2. 显式传递不可省略的 offset
        model_obj = smf.glm(
            formula=formula_freq,
            data=df_clean,
            family=family,
            offset=df_clean['log_exposure']
        )

        # 尝试使用 Newton 算法拟合，它在处理复杂边界时比默认方法更稳健
        freq_model = model_obj.fit(maxiter=100, method='newton')

        print(freq_model.summary())
        dispersion = freq_model.pearson_chi2 / freq_model.df_resid
        print(f"\n📊 Dispersion Indicator: {dispersion:.4f}")
        print("\n📈 Relativities (e^coef):")
        print(np.exp(freq_model.params))

        return freq_model

    except Exception as e:
        print(f"⚠️ Newton 算法拟合失败，正在尝试 BFGS 算法... 错误详情: {e}")
        try:
            # 二次尝试：BFGS 算法对初始值最不敏感
            freq_model = model_obj.fit(maxiter=100, method='bfgs')
            print("✅ BFGS 拟合成功。")
            print(freq_model.summary())
            return freq_model
        except Exception as e2:
            print(f"❌ 所有算法均拟合失败: {e2}")
            return None


# ==========================================
# 3. 程度模型拟合
# ==========================================
def fit_severity_model(df, formula_sev):
    print("\n" + "=" * 80)
    print("🚀 TRAINING SEVERITY MODEL (GAMMA)")
    print("=" * 80)

    df_sev_only = df[(df['claim_amount'] > 0) & (df['claim_count'] > 0)].copy()
    df_sev_only['avg_claim_amount'] = df_sev_only['claim_amount'] / df_sev_only['claim_count']

    vars_in_formula = re.findall(r'\b\w+\b', formula_sev.split('~')[1])
    actual_cols = [v for v in vars_in_formula if v in df_sev_only.columns]
    df_sev_clean = df_sev_only.dropna(subset=['avg_claim_amount'] + actual_cols).copy()

    try:
        sev_model = smf.glm(
            formula=formula_sev,
            data=df_sev_clean,
            family=sm.families.Gamma(link=sm.families.links.log())
        ).fit()  # Severity 通常用默认 IRLS 即可收敛
        print(sev_model.summary())
        print("\n📈 Relativities (e^coef):")
        print(np.exp(sev_model.params))
        return sev_model
    except Exception as e:
        print(f"❌ 程度模型拟合失败: {e}")
        return None


# ==========================================
# 4. 执行主程序
# ==========================================
if __name__ == "__main__":
    ef_freq_path = '/Users/lihonglin/Desktop/EF-freq.xlsx'
    ef_sev_path = '/Users/lihonglin/Desktop/EF-sev.xlsx'

    df_ef = load_and_prep_data(ef_freq_path, ef_sev_path)

    if df_ef is not None:
        ef_freq_formula = "claim_count ~ C(solar_system, Treatment('Helionis Cluster')) + equipment_age + maintenance_int + usage_int"
        ef_sev_formula = "avg_claim_amount ~ C(solar_system, Treatment('Helionis Cluster')) + equipment_age + usage_int"

        # 尝试 NegativeBinomial
        freq_mod = fit_frequency_model(df_ef, ef_freq_formula, dist='NegativeBinomial')

        # 如果拟合依然失败，可以降级尝试 Poisson
        if freq_mod is None:
            print("🔄 尝试降级为 Poisson 分布拟合...")
            freq_mod = fit_frequency_model(df_ef, ef_freq_formula, dist='Poisson')

        sev_mod = fit_severity_model(df_ef, ef_sev_formula)