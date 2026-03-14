import pandas as pd

# 1. 定义跨星系校准系数 (Final Calibration Factors)
galaxy_calib = {
    'Helionis': 1.0000,
    'Bayesia': 2.5309,
    'Oryn_Delta': 5.5489
}

# 2. 定义各险种在基准星系的风险单位纯保费 (Base Pure Premium)
# 数据来源：险种模型拟合报告与压力测试报告
premium_structure = {
    'WC - Drill Operator': 3711.0,
    'WC - Engineer': 3293.0,
    'WC - Administrator': 1000.0,
    'EF - Heavy Machinery': 2800.0,
    'EF - Precision Instruments': 1500.0,
    'CL - High Risk Cargo': 3180.0,
    'CL - Medium Risk Cargo': 2550.0,
    'CL - Low Risk Cargo': 1640.0
}

# 3. 循环计算所有组合
final_rates = {}
for system, factor in galaxy_calib.items():
    final_rates[system] = {risk: round(base * factor, 2)
                           for risk, base in premium_structure.items()}

# 4. 转化为 DataFrame 并输出
df_full_manual = pd.DataFrame(final_rates)
df_full_manual.index.name = "Risk Category / LOB"
print("--- 2026 Inter-System Comprehensive Rate Manual ---")
print(df_full_manual)