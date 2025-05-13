# 导入 Qdrant 客户端和相关模型类
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams  # 向量距离计算方式和向量参数配置
from qdrant_client.models import PointStruct  # 定义向量点的数据结构
from qdrant_client.models import Filter, FieldCondition, MatchValue  # 用于条件过滤

# 定义集合名称
collection_name = "test_collection"
# 初始化 Qdrant 客户端，连接到本地服务
client = QdrantClient(url="http://localhost:6333")

# ========== 集合管理 ==========
# 获取当前所有集合
existing_collections = client.get_collections()
# 提取所有集合名称
collection_names = [col.name for col in existing_collections.collections]

# 检查目标集合是否存在
if collection_name in collection_names:
    print(f"Collection '{collection_name}' exists. Deleting...")
    # 如果存在则删除集合
    client.delete_collection(collection_name)
    print(f"Collection '{collection_name}' deleted successfully.")
else:
    print(f"Collection '{collection_name}' does not exist.")

# ========== 创建新集合 ==========
# 创建新的向量集合
client.create_collection(
    collection_name=collection_name,  # 集合名称
    vectors_config=VectorParams(
        size=4,  # 向量维度
        distance=Distance.DOT  # 使用点积作为相似度计算方式
    ),
)

# ========== 插入数据 ==========
# 批量插入向量数据
operation_info = client.upsert(
    collection_name=collection_name,  # 目标集合
    wait=True,  # 等待操作完成
    points=[
        # 使用 PointStruct 定义每个向量点
        PointStruct(
            id=1,  # 点ID
            vector=[0.05, 0.61, 0.76, 0.74],  # 4维向量
            payload={"city": "Berlin"}  # 附加元数据
        ),
        PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
        PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
        PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
        PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
        PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
    ],
)

# 打印数据插入结果
print("\n" + "="*56)
print("Load dataset:")
print(operation_info)  # 显示操作状态信息

# ========== 向量搜索 ==========
# 执行K近邻搜索 (KNN)
search_result = client.query_points(
    collection_name=collection_name,  # 目标集合
    query=[0.2, 0.1, 0.9, 0.7],  # 查询向量
    with_payload=False,  # 不返回附加数据
    limit=3  # 返回前3个最相似结果
).points  # 获取结果点列表

print("\n" + "="*64)
print("KNN Search (k = 3):")
for point in search_result:
    print(f"ID: {point.id}")  # 打印向量ID


# ========== 混合搜索（带过滤条件） ==========
# 执行带过滤条件的混合搜索
search_result = client.query_points(
    collection_name=collection_name,
    query=[0.2, 0.1, 0.9, 0.7],  # 查询向量
    query_filter=Filter(  # 过滤条件
        must=[  # 必须满足的条件
            FieldCondition(
                key="city",  # 过滤字段
                match=MatchValue(value="London")  # 值匹配条件
            )
        ]
    ),
    with_payload=True,  # 返回附加数据
    limit=3,  # 返回结果数量
).points

print("\n" + "="*64)
print("Hybrid Search (Filter: \"city = London\"):")
for point in search_result:
    print(f"ID: {point.id}, City: {point.payload['city']}")  # 打印向量ID和关系属性
