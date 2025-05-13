# 项目概述
本项目演示如何使用Milvus向量数据库实现多向量查询（Multi-Vector Search）。

## 📊 数据集说明

### 数据集统计 (使用L₂距离或内积距离进行度量)

| 数据集名称 | 向量数据规模   | 向量数据维度 | 原始数据类型 | 关系属性 |
|:---------:|:----------:|:-------:|:----------:|:--------:|
| LoTTE (data) | 190,162   | 128 | 文本段（英文） | 文本段落ID (0~999)  |
| LoTTE (query) | 3,200   | 128 | 文本段（英文） | 查询向量ID (0~99)  |

**字段详细说明**：

- **向量数据规模**：查询向量的条目总数
- **向量数据维度**：每个查询向量的维度数
- **关系属性**：每个查询段落映射为多个向量

### 数据集原始文件格式

- **lotte-lifestyle-data-small.fivecs**：FileIO.py中的read_fivecs读取数据文件
  - 输出结果格式：[[vector_id, doc_id, embedding], ...]
  - 每个文本可能包含不同数量个向量
- **lotte-lifestyle-query-small.fivecs**：FileIO.py中的read_fivecs读取查询文件
  - 输出结果格式：[[vector_id, query_id, embedding], ...]
  - 每个Query包含32个向量
- **ground_truth.dat**：向量查询的精确结果文件（可读）
  - 第1行：向量查询数量m
  - 第2~m+1行：[doc_id, ...] # 每个查询向量的20NN

## Milvus向量数据库相关文件说明

### 代码结构
├── README.md
├── FileIO.py            # 读取原始数据文件
├── VdbConfig.py         # 配置文件
├── ListCollection.py    # 查询当前向量数据库中的数据集
├── DataLoader.py        # 加载数据到Milvus向量数据库中
└── MultiVectorSearch.py        # 使用Milvus向量数据库的实现多向量搜索

### VdbConfig.py
**功能**：统一配置数据集文件地址，主要包括：
>* **DATASET_NAME**：数据集（集合）名称
>* **DATASET_VECTOR_PATH**：原始向量数据的目录
>* **DATASET_ATTR_PATH**：原始向量数据所对应关系属性的目录
>* **SCHEMA_FIELD_CONFIG**：数据集的模式配置
>* **INDEX_PARAMS**：向量索引配置，其中FLAT向量索引用于计算Ground Truth
>* **QUERY_WORKLOAD**: 待测试向量查询的目录
>* **SEARCH_PARAMS**: 向量查询处理过程中的参数设置

**注意**：在``SCHEMA_FIELD_CONFIG``中，向量数据的``dim``属性需要根据数据集进行动态调整

### DataLoader.py
**功能**：将数据集加载到Milvus向量数据库，主要包括：
>* 检查并创建集合
>* 批量插入向量数据
>* 构建向量索引

**运行**：
```bash
python3 DataLoader.py
```

### MultiVectorSearch.py
**功能**：使用Milvus向量数据库实现多向量搜索
>* KNN查询
>* 混合查询
>* 召回率计算

**运行**：
```bash
python3 MultiVectorSearch.py
```
