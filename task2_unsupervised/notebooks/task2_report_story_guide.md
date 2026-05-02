# Task2 报告写作指南（执行复盘 + 叙事框架 + 必写清单）

> 适用对象：当前 Task2 无监督实验（K-Means vs GMM，公平双视角，三种子稳定性，含 bribe 消融）
> 更新时间：2026-04-14

---

## 1. 为了完成 Task2，你已经做了什么（可直接写进 Implementation）

### 1.1 已实现的实验流水线
你已经完成了一个可复现的端到端无监督流程：

1. 共享预处理（两种方法共用同一输入）
   - 脚本：src/preprocess_data.py
   - 输入：dungeon_sensorstats.csv
   - 处理：
     - 去除 species（防止训练时标签泄漏）
     - 缺失值填补（中位数/众数）
     - 连续特征标准化
     - 主实验默认不包含 bribe

2. Method A（K-Means）
   - 脚本：src/train_kmeans.py
   - 在同一搜索范围内扫描 k（2..8）
   - 输出每个 k 的指标、簇分配、簇中心、簇画像

3. Method B（GMM）
   - 脚本：src/train_gmm.py
   - 在同一搜索范围内扫描 components（2..8）
   - 输出每个 k 的指标、簇分配、均值、协方差对角、簇画像

4. 公平比较
   - 脚本：src/compare_methods.py
   - 双视角：
     - method_best：各方法用各自最优簇数
     - fixed_same_k：两方法都固定 k=5

5. 可视化
   - 脚本：src/visualize_embeddings.py
   - 输出 PCA + UMAP 并排可视化
   - 输出簇画像热图

6. 多种子稳定性汇总
   - 脚本：src/aggregate_multi_seed.py
   - seeds：42, 123, 2026
   - 输出 mean/std 汇总表与汇总图

### 1.2 已落地的风险控制
你不是只做了模型，还做了方法学管控：

- 风险1（公平性质疑）
  - 解决：双视角报告（method_best + fixed_same_k）

- 风险2（不可复核）
  - 解决：日志强制记录 6 项最小复核信息
    1) preprocessing_version
    2) search_space
    3) seed_set
    4) feature_columns
    5) software_versions
    6) selection_rule

- 风险3（UMAP图主导结论）
  - 解决：明确“图是解释证据，结论由三指标+多seed稳定性给出”

### 1.3 主实验关键结果（no-bribe）
来自 outputs/tables/metrics_multi_seed_summary.csv：

- fixed_same_k（k=5）
  - K-Means: silhouette 0.5578, DBI 0.7357, CHI 16845.7989
  - GMM:     silhouette 0.5571, DBI 0.7381, CHI 16726.0665

- method_best（两者都选到 k=6）
  - K-Means: silhouette 0.5671, DBI 0.7100, CHI 16317.8444
  - GMM:     silhouette 0.5663, DBI 0.7136, CHI 16193.0713

结论（主实验）：K-Means 在两种视角下都略优于 GMM，且更快。

### 1.4 消融结果（with-bribe）
来自 outputs_bribe/tables/metrics_multi_seed_summary.csv：

- fixed_same_k（k=5）
  - K-Means: silhouette 0.6142, DBI 0.6058, CHI 17686.6791
  - GMM:     silhouette 0.6141, DBI 0.6061, CHI 17656.9718

- method_best
  - K-Means: silhouette 0.6176, DBI 0.6156, CHI 18477.5715
  - GMM:     silhouette 0.6175, DBI 0.6154, CHI 18441.5968

结论（消融）：含 bribe 时分群指标显著“变好”，但可能是捷径特征带来的可分性放大，因此必须作为敏感性分析，不可替代主结论。

---

## 2. 报告中 Task2 应该讲一个怎样的故事

建议故事线（非常适合评分）：

### 故事主线

1. 问题定义
   - 目标不是监督分类，而是发现实体在传感器空间的自然分群结构。

2. 方法设计
   - 选择 K-Means 与 GMM 做互补比较：
     - K-Means：高效、可解释、强基线
     - GMM：概率软分配、可表达边界不确定性

3. 公平协议
   - 两方法同预处理、同搜索范围、同seed、同选模规则。
   - 用双视角避免“只比各自最优”的质疑。

4. 关键结果
   - task-aligned（k=5）与 method-best（k=6）都报告。
   - 三种子结果稳定，结论不依赖单次随机性。

5. 解释与风险
   - 为什么可以出现“5物种但k=6”：无监督最优结构不必等于标签数。
   - 为什么 no-bribe 为主、with-bribe 为辅：防止泄漏型特征夸大结论。

6. 机器人语境落地
   - 结果可用于实体行为分层、风险分层、边界样本识别。

---

## 3. 写报告时要注意什么（高分/避坑）

### 3.1 原则

- 原则A：先讲协议，再讲结果
  - 先交代公平性设计，否则结果会被质疑“挑数据/挑指标”。

- 原则B：先讲定量，再讲可视化
  - UMAP/PCA 是辅助解释，不是主证据。

- 原则C：主实验与消融严格分层
  - 正文主结论：no-bribe
  - 消融：appendix 或 sensitivity subsection

- 原则D：结果必须可复核
  - 表格、图、脚本路径、seed 一一对应。

### 3.2 常见扣分点

- 只报 method_best，不报 fixed_same_k
- 只放 UMAP 图，不给三指标表
- 不解释 k=6 与 5 物种的关系
- 把 with-bribe 的高分当作主结论
- 不给多seed稳定性（mean/std）

---

## 4. 什么一定一定要写进去（硬性清单）

下面每项建议在报告中显式出现：

1. 数据与特征
   - 数据来源
   - 去掉 species 参与训练
   - 主实验是否包含 bribe（答案：不包含）

2. 预处理细节
   - 缺失值策略
   - 标准化策略
   - 共享预处理给两方法

3. 方法与公平性
   - K-Means 与 GMM 的超参搜索范围一致
   - seeds = 42/123/2026
   - 选模规则（Silhouette desc, DBI asc, CHI desc）
   - 双视角报告（method_best + fixed_same_k）

4. 指标与结果
   - Silhouette、DBI、CHI 三指标
   - 每个视角的方法对比表
   - 多seed mean/std 汇总

5. 可视化
   - PCA + UMAP 并排图
   - PCA explained variance ratio
   - 簇画像热图

6. 关键解释
   - 为什么 5 物种可以出现 k=6
   - 为什么 with-bribe 仅作敏感性分析

7. 风险与局限
   - UMAP与定量可能不一致
   - 特征泄漏/捷径风险
   - 不同协方差假设对 GMM 的潜在影响

8. 可复现清单
   - 环境版本（Python + 包版本）
   - 脚本入口与命令
   - 产物路径

---

## 5. 推荐的报告段落结构（可直接套用）

### 5.1 Implementation
写“做了什么”：
- 数据管线
- 两种方法实现
- 输出产物结构

### 5.2 Design Choices
写“为什么这样做”：
- 选 K-Means + GMM 的互补性
- 双视角公平协议
- no-bribe 主实验 + with-bribe 消融

### 5.3 Results
写“发生了什么”：
- fixed_same_k 与 method_best 的结果表
- 多seed稳定性
- 可视化观察与簇解释

### 5.4 Evaluation
写“这意味着什么”：
- 对机器人任务的可用价值
- 局限性与风险
- 后续可改进方向

---

## 6. 一句话总论（可放在 Task2 小结末尾）

在严格公平协议与三种子验证下，K-Means 与 GMM 均能稳定发现实体分群结构，K-Means在当前数据上略占优；同时，with-bribe 消融显示某些特征可显著抬升内部指标，因此最终结论采用 no-bribe 主实验，with-bribe 仅作敏感性与鲁棒性补充。
