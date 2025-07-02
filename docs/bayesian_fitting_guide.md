# 贝叶斯拟合功能使用指南

## 概述

本项目新增了贝叶斯拟合功能，用于改进可转债数据分析中的多项式拟合效果。贝叶斯方法相比传统方法具有以下优势：

1. **更好的不确定性量化** - 提供预测区间和置信区间
2. **更鲁棒的参数估计** - 对异常值更不敏感
3. **自动模型选择** - 可以自动比较不同模型并选择最佳的
4. **防止过拟合** - 通过先验分布约束参数

## 依赖项

使用贝叶斯拟合功能需要安装以下依赖：

```bash
pip install pymc arviz
```

这些依赖已添加到 `requirements.txt` 中。

## 配置参数

### 基本配置结构

```toml
[config_name.fit_config]
enable_fitting = true
fit_type = "bayesian"  # 设置为贝叶斯拟合

[config_name.fit_config.bayesian]
method = "linear"           # 贝叶斯方法类型
prior_type = "normal"       # 先验分布类型
n_samples = 2000           # MCMC采样数量
n_chains = 2               # MCMC链数量
credible_interval = 0.95   # 置信区间
max_degree = 3             # 多项式最大次数（仅polynomial/auto方法）
```

### 参数详解

#### 贝叶斯方法类型 (`method`)

- `"linear"` - 贝叶斯线性回归
- `"polynomial"` - 贝叶斯多项式回归（自动选择最佳次数）
- `"auto"` - 自动模型选择（比较线性和多项式模型）

#### 先验分布类型 (`prior_type`)

- `"normal"` - 正态先验（默认，适用于大多数情况）
- `"laplace"` - 拉普拉斯先验（对异常值更鲁棒）
- `"horseshoe"` - Horseshoe先验（用于稀疏回归和特征选择）

#### MCMC参数

- `n_samples` - 每条链的采样数量（推荐 1000-5000）
- `n_chains` - 并行链数量（推荐 2-4）
- `credible_interval` - 置信区间水平（通常 0.90-0.95）

## 配置示例

### 1. 贝叶斯线性拟合

```toml
[bayesian_linear_example]
title = '贝叶斯线性拟合示例'
column = '转股溢价率(%)'
ctype = 'median'

[bayesian_linear_example.conditions]
main = ["\"债券类型\" = '可转债'", '"转换价值" > 90', '"转换价值" <= 120']

[bayesian_linear_example.fit_config]
enable_fitting = true
min_data_threshold = 50
fit_type = "bayesian"
historical_files_count = 15

[bayesian_linear_example.fit_config.bayesian]
method = "linear"
prior_type = "normal"
n_samples = 2000
n_chains = 2
credible_interval = 0.95

[bayesian_linear_example.fit_config.quality]
min_r_squared = 0.5
include_quality_info = true

[bayesian_linear_example.fit_config.predict_conditions]
predict_column = "转换价值"
predict_value = 100
target_column = "转股溢价率(%)"
```

### 2. 贝叶斯多项式拟合

```toml
[bayesian_polynomial_example]
title = '贝叶斯多项式拟合示例'
column = '转股溢价率(%)'
ctype = 'median'

[bayesian_polynomial_example.fit_config]
enable_fitting = true
fit_type = "bayesian"

[bayesian_polynomial_example.fit_config.bayesian]
method = "polynomial"
prior_type = "normal"
n_samples = 3000
n_chains = 2
max_degree = 4  # 最大4次多项式
credible_interval = 0.95
```

### 3. 自动模型选择

```toml
[bayesian_auto_example]
title = '贝叶斯自动模型选择'
column = '转股溢价率(%)'
ctype = 'median'

[bayesian_auto_example.fit_config]
enable_fitting = true
fit_type = "bayesian"

[bayesian_auto_example.fit_config.bayesian]
method = "auto"        # 自动选择最佳模型
prior_type = "horseshoe"  # 使用稀疏先验
n_samples = 2500
n_chains = 2
max_degree = 3
credible_interval = 0.95
```

### 4. 鲁棒性拟合（处理异常值）

```toml
[robust_bayesian_example]
title = '鲁棒贝叶斯拟合（抗异常值）'
column = '转股溢价率(%)'
ctype = 'median'

[robust_bayesian_example.fit_config]
enable_fitting = true
fit_type = "bayesian"

[robust_bayesian_example.fit_config.bayesian]
method = "linear"
prior_type = "laplace"  # 拉普拉斯先验对异常值更鲁棒
n_samples = 2000
n_chains = 2
credible_interval = 0.90  # 较低的置信区间
```

## 输出信息

当启用 `include_quality_info = true` 时，贝叶斯拟合会输出额外的信息：

```
拟合类型: bayesian_linear, R²: 0.854
预测当转换价值=100时，转股溢价率(%)=12.345
预测不确定性: ±1.234
置信区间: [10.123, 14.567]
```

这些信息包括：
- **预测不确定性** - 预测值的标准差
- **置信区间** - 基于设定的 `credible_interval` 计算的区间

## 与传统方法的比较

| 特性 | 传统方法 | 贝叶斯方法 |
|------|----------|------------|
| 参数估计 | 点估计 | 概率分布 |
| 不确定性 | 仅R² | 完整的不确定性量化 |
| 异常值敏感性 | 高 | 低（特别是laplace先验） |
| 过拟合风险 | 高 | 低（先验约束） |
| 模型选择 | 手动 | 自动（auto方法） |
| 计算时间 | 快 | 较慢（MCMC采样） |

## 最佳实践

1. **选择合适的方法**：
   - 简单关系：使用 `linear`
   - 复杂关系：使用 `polynomial` 或 `auto`
   - 不确定关系：使用 `auto`

2. **选择合适的先验**：
   - 默认情况：`normal`
   - 有异常值：`laplace`
   - 需要特征选择：`horseshoe`

3. **调整采样参数**：
   - 复杂模型：增加 `n_samples`
   - 提高精度：增加 `n_chains`
   - 计算资源有限：减少采样数量

4. **处理数据质量**：
   - 数据量少：降低 `credible_interval`
   - 噪声大：使用鲁棒先验
   - 关系复杂：允许更高的多项式次数

## 向后兼容性

贝叶斯拟合功能完全向后兼容，现有的传统拟合配置（`linear`, `polynomial`, `exponential` 等）继续正常工作，不受影响。

## 故障排除

1. **ImportError: No module named 'pymc'**
   - 解决：安装 PyMC：`pip install pymc arviz`

2. **采样收敛警告**
   - 解决：增加 `n_samples` 或 `n_chains`
   - 检查数据质量，移除明显异常值

3. **拟合失败**
   - 检查数据量是否足够（建议至少20个样本）
   - 尝试不同的先验类型
   - 降低 `min_r_squared` 阈值

4. **计算时间过长**
   - 减少 `n_samples`（最低500）
   - 减少 `max_degree`
   - 使用更简单的先验（`normal` 而非 `horseshoe`）

## 示例文件

完整的配置示例可以在 `config/demo/bayesian_demo.toml` 中找到。