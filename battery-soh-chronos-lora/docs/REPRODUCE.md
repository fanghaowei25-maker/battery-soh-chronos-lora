# 代码使用与复现

这是基础 Chronos-2 + LoRA + SOH Head 主实验的代码整理包。源代码复制自 2026-09-25 参赛交付目录；训练实现保持原样。没有保存各折已训练最佳权重。本次检查为 Python 语法检查，未在 GPU 上重新训练。

## 文件用途

| 文件 | 用途 |
|---|---|
| code/backbone_adapt.py | LoRA 训练及按电芯留一评估主入口 |
| code/train_sit_soh.py | SOH 回归头、基础训练和指标 |
| code/diff_embed.py | 支持反向传播的 Chronos-2 特征提取 |
| code/prepare_sit_v2.py | 从原始 Data.zip 生成数据文件 |
| code/cache_sit_embeddings.py | 生成冻结模型电压特征缓存 |
| code/evaluate_protocol.py | 评分协议和区间指标 |
| code/analyze_adapt.py | 训练结果分析 |
| code/device_utils.py | 设备和模型路径选择 |
| code/check_env.py | 环境检查 |
| code/lowsoh_coverage.py | 低 SOH 样本覆盖检查 |
| code/verify_diff_embed.py | 检查两种嵌入路径的一致性 |
| recompute_aux_metric.py | 初始容量口径的辅助结果复算 |

## 环境与数据

先安装适合 GPU 的 CUDA 版 PyTorch，再安装 requirements.txt 中其余依赖。依赖版本尚未锁定，具体兼容性以环境检查为准。

在仓库根目录准备 data/ 和 cache/emb_V/。

```bash
python code/prepare_sit_v2.py --zip /path/to/Data.zip --out data/sit_v2_discharge.npz --length 1024
python code/cache_sit_embeddings.py --data data/sit_v2_discharge.npz --channels V --out cache/emb_V --device cuda
```

也可复制本地交付包的 NPZ、embeddings.npz 和缓存配置到上述对应目录。预训练模型会按 device_utils.py 的逻辑从本地 code/chronos-2/ 或官方模型源加载。预处理与模型版本须与原实验一致才能用于精确复核。

## 主实验

从仓库根目录执行：

```bash
python code/backbone_adapt.py --data data/sit_v2_discharge.npz --emb cache/emb_V --out results/reproduction --arms lora --lora-preset all --lora-r 8 --lora-alpha 16 --lr-back 1e-4 --lr-head 1e-3 --seeds 3 --epochs 20 --patience-adapt 3 --val-cells 3 --batch 8 --device cuda:0
```

该命令覆盖全部电芯，每折独立恢复初始权重，验证集选轮次。若只做冒烟运行，可另指定 --folds 1 --seeds 1 --epochs 1，并使用不同输出目录；其结果不等同于主实验。

## 辅助指标复算

确认 data/sit_v2_discharge.npz 和 results/ensemble_predictions/ 是同一历史数据版本，然后在仓库根目录执行：

```bash
python recompute_aux_metric.py
```

该脚本只做历史预测的初始容量换算和指定样本筛选，预期 17 电芯、11,302 循环；不会重新训练模型。结果口径说明见 README.md。

## 发布范围

本包未并入历史服务器启动脚本、云平台凭据、个人资料、参赛盖章证明及大体积数据/模型文件。代码许可和贡献者署名需由团队决定；数据及预训练模型遵循各自许可。
