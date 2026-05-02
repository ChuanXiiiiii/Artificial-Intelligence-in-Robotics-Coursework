# Task1 报告写作指南（执行复盘 + 叙事框架 + 必写清单）

> 适用对象：Task1 监督学习（HOG+SVM vs ResNet18，公平比较，三种子稳定性，含 ConvergenceWarning 对照）
> 更新时间：2026-04-14

---

## 1. 为了完成 Task1，你已经做了什么（可直接写进 Implementation）

### 1.1 已实现的端到端流水线
你已经完成了可复现的监督学习流程：

1. 数据与任务定义
   - 数据源：`dungeon_images_colour80/`
   - 标签：5 类（halfling / human / lizard / orc / wingedrat）
   - 任务：单标签多分类（species recognition）

2. 共享数据协议（两方法共用）
   - 脚本：`src/build_split_manifest.py`
   - 划分：train/val/test = 70/15/15
   - 稳健性策略：prefix + number bucket 分组（bucket-size=20）
   - 约束：`min-val-support=30`, `min-test-support=30`（每类最小样本支持）
   - 目标：避免同组泄漏，避免极端不平衡验证集

3. Method A（HOG + LinearSVC）
   - 脚本：`src/train_hog_svm.py`
   - 流程：灰度化 -> HOG 特征 -> StandardScaler -> LinearSVC
   - 调参：`C in {0.1, 1.0, 3.0}`（验证集选最优）
   - 调优版求解设置：`max_iter=30000`, `dual=False`, `tol=1e-4`

4. Method B（ResNet18 transfer learning）
   - 脚本：`src/train_resnet18.py`
   - 流程：ImageNet 预训练 -> 替换分类头 -> 冻结/解冻微调
   - 设备优先级：cuda > mps > cpu
   - class weighting：默认 `sqrt_balanced`

5. 公平比较与可视化
   - 脚本：`src/compare_results.py`
   - 指标：Accuracy、Macro-F1、混淆矩阵（raw + normalized）
   - 误差分析：每类指标 + 常见误分类对

6. 多种子稳定性
   - 脚本：`src/aggregate_multi_seed.py`
   - seeds：42, 123, 2026
   - 产物：多种子 all + summary 表，汇总图

7. 收敛警告对照分析（Method A）
   - 脚本：`src/make_hog_comparison.py`
   - 输出：`hog_convergence_before_after.csv` 与 summary

8. 报告素材同步
   - 脚本：`src/sync_report_assets.py`
   - 输出同步到：`docs/report/figures` 与 `docs/report/tables`

### 1.2 关键结果（可直接写 Results 主表）
来自 `outputs/tables/metrics_multi_seed_summary.csv`：

- HOG+SVM：
  - Accuracy: 0.9652 +- 0.0093
  - Macro-F1: 0.9662 +- 0.0077

- ResNet18：
  - Accuracy: 0.9961 +- 0.0046
  - Macro-F1: 0.9958 +- 0.0052

主结论：ResNet18 在当前数据和协议下显著优于 HOG+SVM，且跨种子稳定。

### 1.3 ConvergenceWarning 对照结论（Method A）
来自 `outputs/tables/hog_convergence_before_after_summary.csv`：

- before（max_iter=8000, dual=True）：
  - warning_count_total = 2
  - accuracy_mean = 0.9641
  - macro_f1_mean = 0.9643

- after（max_iter=30000, dual=False）：
  - warning_count_total = 2
  - accuracy_mean = 0.9652
  - macro_f1_mean = 0.9663

解释：警告没有在所有 seed 完全消失，但结论不发生反转，性能保持稳定并略有提升。可在报告中写：
在本任务与当前实验协议下，该 warning 不改变方法比较结论。

---

## 2. 报告中 Task1 应该讲一个怎样的故事

建议故事线（评分友好）：

1. 问题定义
   - 机器人视觉中的实体物种识别，目标是可靠分类而不是仅单次高分。

2. 方法设计
   - 选择传统强基线（HOG+SVM）与深度迁移学习（ResNet18）做互补比较。

3. 公平协议
   - 两方法同数据划分、同 seed 集、同评价指标、同测试集。
   - 划分引入 bucket + 最小样本约束，避免早期 split 偏斜。

4. 关键结果
   - 报告 3-seed mean/std，不依赖单 seed 偶然性。
   - ResNet18 全面领先。

5. 可信度讨论
   - 讨论 ConvergenceWarning：不掩盖，但用前后对照证明结论稳定。

6. 机器人语境落地
   - 结果可用于感知模块中的实体识别与策略条件输入。

---

## 3. 写报告时要注意什么（高分/避坑）

### 3.1 写作原则

- 先协议，后结果：先说明公平比较，再贴分数。
- 先定量，后图像：分数表是主证据，图是解释证据。
- 先主结论，后补充：主文写最终协议结果，warning 对照放 Results/Evaluation 小节或附录。
- 统一口径：结论只基于当前数据与协议，不做过度泛化。

### 3.2 常见扣分点

- 只报单 seed，不报 mean/std。
- 不说明数据划分策略，导致公平性质疑。
- 只放准确率，不报 Macro-F1（类别不平衡下会被质疑）。
- 有 warning 但完全不解释。
- 图表与表格路径不一致，难以复核。

---

## 4. 什么一定一定要记得写进去（硬性清单）

1. 数据与任务
   - 数据源、类别数、任务定义（5 类单标签分类）。

2. 预处理与划分
   - split 比例（70/15/15）
   - bucket 分组策略与最小 val/test 支持约束
   - 防泄漏说明

3. 方法与超参数
   - Method A: HOG 参数 + SVM C 搜索与 solver 设置
   - Method B: ResNet18 迁移学习策略 + class weighting

4. 公平比较协议
   - 同 split、同 seeds、同指标、同 test

5. 指标与结果
   - Accuracy、Macro-F1、混淆矩阵
   - 每类 precision/recall/F1
   - 3-seed mean/std

6. 误差分析
   - 高频误分类对
   - human 与 halfling 等易混类别解释

7. ConvergenceWarning 解释
   - 前后对照表
   - 为什么 warning 不改变最终结论（在本任务范围内）

8. 可复现清单
   - Python 版本（3.11）
   - 关键命令
   - 产物路径（outputs 与 docs/report）

---

## 5. 可直接复用的 Task1 一句话总论

在统一公平协议与三种子验证下，ResNet18 在 Accuracy 与 Macro-F1 上均显著优于 HOG+SVM；同时，Method A 的 ConvergenceWarning 经前后对照验证不改变最终比较结论，因此 Task1 的主结论具有稳定性与可复核性。
