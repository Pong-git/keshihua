# VAST Challenge 2023 MC1 可视化课程设计

本项目用于完成 VAST Challenge 2023 Mini-Challenge 1 的课程设计，目标是围绕非法捕鱼调查任务，构建一个知识图谱可视分析系统，并配套完成报告与 PPT。

## 项目结构

```text
.
├─ MC1/                         # 原始数据与官方说明
│  └─ V1/MC1.json
├─ app.py                       # Streamlit 可视化入口
├─ requirements.txt             # 运行依赖
├─ src/
│  ├─ config.py                 # 路径、可疑实体、颜色等配置
│  ├─ data_loader.py            # 读取 JSON 并转换为节点表、边表
│  ├─ graph_analysis.py         # 邻居、子图、路径、共同邻居分析
│  ├─ risk_model.py             # FishHook 风格异常评分与候选实体推荐
│  └─ export_tables.py          # 导出课程报告可用的 CSV 表
├─ outputs/                     # 预处理结果、FishHook 排名和截图输出目录
├─ report/                      # 后续放课程设计报告
└─ slides/                      # 后续放课程展示 PPT
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 生成分析表

```bash
python -m src.export_tables
```

生成结果会保存到 `outputs/`：

- `nodes.csv`
- `edges.csv`
- `suspect_summary.csv`
- `company_risk_ranking.csv`：从 FishHook 综合排名中筛选出的 company 类型候选实体

## 启动可视化系统

```bash
streamlit run app.py
```

## 分析主线

课程报告建议围绕官方任务展开：

1. 对 4 个官方可疑实体进行上下文网络分析。
2. 比较它们的关系模式和风险证据。
3. 基于网络结构推荐 FishEye 继续调查的公司。
4. 反思知识图谱分析中的不确定性，例如抽取误差、匿名 ID、关系方向和边权重解释。

## FishHook 风格扩展

项目已参考论文 `FishHook: A Visual Analytics System for Tracing Suspicious Entities in the Fisheries Domain Using Knowledge Graphs` 增加了以下分析能力：

- **Ego Net**：以当前实体为中心展示一跳/二跳上下文网络，并高亮官方可疑实体。
- **从可疑实体扩展**：以官方 4 个可疑实体为种子，寻找 1-2 跳内的候选公司、船只、人物和组织。
- **FishHook 候选排名**：综合考虑可疑实体距离、弱连通分量船只比例、1-2 跳家族/政治关系、短环路数量、船只连接和 ownership/membership 关系。
- **平行坐标对比**：类似论文中的 Parallel Coordinates，用多个结构指标同时比较候选实体异常模式。
- **短环路检测**：在 ownership、membership、partnership 关系中寻找经过候选实体的短环路，用于辅助发现复杂控制结构或供应链闭环。

新增导出文件：

- `outputs/fishhook_anomaly_features.csv`
- `outputs/fishhook_candidate_ranking.csv`
- `outputs/suspect_expansion_candidates.csv`
