# 数据获取

原始 SIT 数据来源：https://doi.org/10.25447/sit.32101523 。原数据说明见 SIT_SOURCE_README.md。

本代码包不包含原始 Data.zip、预处理 NPZ、嵌入缓存或预训练模型权重。可从数据来源获取原始文件后按 ../docs/REPRODUCE.md 生成预处理文件；也可使用你本地参赛交付包里的 data/sit_v2_discharge.npz 与 cache/embeddings.npz。

已有预测 CSV 与 NPZ 必须逐样本对应；重新预处理后若样本顺序或内容有变化，不得直接把历史预测 CSV 和新 NPZ 混用。
