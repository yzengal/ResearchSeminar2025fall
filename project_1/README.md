# 项目概述
本项目演示如何使用Qdrant和Milvus向量数据库，包含数据加载、查询测试和配置管理功能。

## 安装说明

### 安装Qdrant客户端
```bash
conda activate myenv
pip3 install qdrant-client -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**注意**：`myenv`是你自己的conda环境

## 📊 数据集说明

### 数据集统计 (使用L₂距离或内积距离进行度量)

| 数据集名称 | 数据规模   | 数据维度 | 原始数据类型 | 关系属性 |
|:---------:|:----------:|:-------:|:----------:|:--------:|
| WIT        | 8,807   | 2048 | 图像      | 图片尺寸（int类型）  |
| Youtube-audio   | 71,321 | 128  | 音频      | 类别标签（string类型） |
| Youtube-rgb    | 31,406 | 1024 | 视频      | 场景分类（string类型） |

**字段详细说明**：

- **数据规模**：向量条目总数
- **数据维度**：每个向量的维度数
- **关系属性**：关系属性均只包含1个特征

### 数据集原始文件格式

- **query.txt**：向量查询的文件
  - 第1行：向量查询数量m 向量数据维度d
  - 第2~m+1行：查询向量（d个浮点数） 关系属性过滤条件
- **vector_0.fivecs**：向量数据的文件（二进制格式）
- **meta_0.txt**：向量数据所对应关系属性的文件
  - 第1行：向量数量n 关系属性数量c（c均为1）
  - 第2行：关系属性的名称 关系属性的类型
  - 第3~n+2行：向量数据对应的关系属性

## Milvus向量数据库相关文件说明

### 代码结构
├── README.md
├── PlotFigure.py        # 画实验图脚本（optional）
├── VdbConfig.py         # 配置文件
├── ListCollection.py    # 查询当前向量数据库中的数据集
├── DataLoader.py        # 加载数据到Milvus向量数据库中
└── QueryProcessor.py    # 测试Milvus向量数据库的查询性能

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

### QueryProcessor.py
**功能**：测试Milvus向量数据库的查询性能
>* KNN查询
>* 混合查询
>* 召回率计算

**运行**：
```bash
python3 QueryProcessor.py
```

## Qdrant向量数据库相关文件说明 

### TestQdrant.py
**功能**：Qdrant向量数据库的基本操作，主要包括：
>* 数据集（集合）的创建与删除
>* 向量数据的装载
>* KNN查询
>* 混合查询
>* 查询结果打印

**运行**：
```bash
python3 TestQdrant.py
```