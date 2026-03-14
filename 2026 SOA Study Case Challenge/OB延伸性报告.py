import pandas as pd

# 1. 定义从 GLM 模型中提取的敏感度系数（Relativities）
# 依据《险种模型报告》拟合结果
sensitivities = {
    'solar_radiation': 1.15,  # 每增加0.1单位辐射，风险增加15%
    'gravity_level': 1.054,  # 每增加0.1G，风险增加5.4%
    'debris_density': 1.085  # 每增加0.01单位密度，风险增加8.5%
}

# 2. 定义各星系相对于基准（Helionis）的环境偏差
galaxy_env_delta = {
    'Helionis': {'rad': 0, 'grav': 0, 'debris': 0, 'uncertainty': 0.0},
    'Bayesia': {'rad': 0.3, 'grav': 0.1, 'debris': 0.05, 'uncertainty': 0.05},
    'Oryn_Delta': {'rad': 0.1, 'grav': 0.4, 'debris': 0.15, 'uncertainty': 0.15}  # 代理数据调整
}


def calculate_calibration_factor(name, deltas):
    # 计算环境诱发增量
    alpha_rad = sensitivities['solar_radiation'] ** (deltas['rad'] / 0.1)
    alpha_grav = sensitivities['gravity_level'] ** (deltas['grav'] / 0.1)
    alpha_debris = sensitivities['debris_density'] ** (deltas['debris'] / 0.01)

    # 综合环境系数
    alpha_s = alpha_rad * alpha_grav * alpha_debris

    # 叠加不确定性负载 (Uncertainty Loading)
    final_factor = alpha_s * (1 + deltas['uncertainty'])
    return round(final_factor, 4)


# 3. 生成校准清单
results = []
for galaxy, deltas in galaxy_env_delta.items():
    factor = calculate_calibration_factor(galaxy, deltas)
    results.append({'Solar System': galaxy, 'Final Calibration Factor': factor})

df_calibration = pd.DataFrame(results)
print("--- 跨星系定价校准结果 (Final Calibration Factors) ---")
print(df_calibration)

# 4. 导出为报告格式
# df_calibration.to_csv('System_Calibration_Factors.csv', index=False)