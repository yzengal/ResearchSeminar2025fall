from pymilvus import MilvusClient, DataType
from FileIO import read_fivecs
from VdbConfig import vdb_config
from pymilvus import (
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusClient
)
from typing import Dict, List, Optional, Any
from tqdm import tqdm

class DataLoader:
    def __init__(self, milvus_client: MilvusClient):
        """
        初始化 DataLoader 类
        
        Args:
            milvus_client (MilvusClient): Milvus 客户端实例
        """
        self.client = milvus_client

    def read_data(self, vector_file_path: str) -> List[Dict[str, Any]]:
        """
        读取向量数据文件（支持 .fivecs）
        
        Args:
            vector_file_path (str): 向量数据文件路径
            
        Returns:
            list: 读取的数据，三元组字典组成的列表 (Vector_ID, Embedding, Doc_ID)
        """
        if not vector_file_path.endswith('.fivecs'):
            raise ValueError("向量数据仅支持 .fivecs 文件")

        multivector_data_list = read_fivecs(vector_file_path)
        vector_data_list, attr_data_list = [], []
        for vector_id, doc_id, embedding in multivector_data_list:
            vector_data_list.append([vector_id, embedding])
            attr_data_list.append(doc_id)
        del multivector_data_list
            
        data_list = []
        for i in range(len(vector_data_list)):
            element = {
                "id": vector_data_list[i][0],
                "vector": vector_data_list[i][1],
                "doc": int(attr_data_list[i])
            }
            data_list.append(element)        

        return data_list

    def create_schema(
        self,
        collection_name: str,
        fields_config: List[Dict],
        description: str = "",
        **kwargs
    ) -> CollectionSchema:
        """
        创建 Milvus 集合的 Schema
        
        Args:
            collection_name (str): 集合名称
            fields_config (List[Dict]): 字段配置列表，例如:
                [
                    {"name": "id", "type": DataType.INT64, "is_primary": True},
                    {"name": "embedding", "type": DataType.FLOAT_VECTOR, "dim": 128}
                ]
            description (str): 集合描述
            
        Returns:
            CollectionSchema: 创建的 Schema
        """
        fields = []
        for config in fields_config:
            if "dim" in config:
                field = FieldSchema(
                    name=config["name"],
                    dtype=config["dtype"],
                    is_primary=config.get("is_primary", False),
                    dim=config["dim"],
                    description=config.get("description", ""),
                )
            else:
                field = FieldSchema(
                    name=config["name"],
                    dtype=config["dtype"],
                    is_primary=config.get("is_primary", False),
                    description=config.get("description", ""),
                )
            print(field.to_dict())
            fields.append(field)
        
        schema = CollectionSchema(
            fields=fields,
            description=description,
            **kwargs
        )
        
        # 创建集合
        if utility.has_collection(collection_name, using=self.client._using):
            print(f"集合 {collection_name} 已存在，正在删除...")
            utility.drop_collection(collection_name, using=self.client._using)
            print("删除完成")
        else:
            print(f"集合 {collection_name} 不存在，正在创建...")
        
        collection = Collection(
            name=collection_name,
            schema=schema,
            using=self.client._using
        )
        print(f"集合 {collection_name} 创建成功")
        
        return schema


    def load_data(
        self,
        collection_name: str,
        data_list: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> None:
        """
        将数据加载到 Milvus 集合中
        
        Args:
            collection_name (str): 目标集合名称
            data_list (List[Dict[str, Any]]): 要插入的数据
            batch_size (int): 批量插入的大小
        """
        collection = Collection(collection_name, using=self.client._using)

        # 分批插入
        total_size = len(data_list)
        with tqdm(total=total_size, desc="插入数据进度") as pbar:
            for sid in range(0, total_size, batch_size):
                eid = min(sid+batch_size, total_size)
                batch_data = data_list[sid:eid]
                collection.insert(batch_data)
                pbar.update(eid - sid)  # 更新已插入的数据条数

        collection.flush()

    def create_index(
        self,
        collection_name: str,
        index_params: Dict[str, Any]
    ) -> None:
        """
        将数据加载到 Milvus 集合中
        
        Args:
            collection_name (str): 目标集合名称
            index_params (Dict[str, Any]): 索引配置的参数
        """
        collection = Collection(collection_name, using=self.client._using)
        
        index_field_name = index_params["field_name"]
        collection.create_index(field_name=index_field_name, index_params=index_params)
        print(f"{collection_name} 向量索引创建完成")
        
        collection.flush()


    def is_loaded(self, collection_name: str) -> bool:
        """
        检查集合是否已加载到内存
        
        Args:
            collection_name (str): 集合名称
            
        Returns:
            bool: 是否已加载
        """
        if not utility.has_collection(collection_name, using=self.client._using):
            return False
        
        collection = Collection(collection_name, using=self.client._using)
        return collection.get_load_state() == utility.LoadState.Loaded


    def ensure_loaded(self, collection_name: str, timeout: int = 300) -> bool:
        """
        确保集合已加载，若未加载则自动加载
        
        Args:
            collection_name (str): 集合名称
            timeout (int): 超时时间（秒）
            
        Returns:
            bool: 是否加载成功
        """
        if self.is_loaded(collection_name):
            return True
        
        print(f"开始加载集合 {collection_name}...")
        collection = Collection(collection_name, using=self.client._using)
        collection.load()
        
        # 等待加载完成
        import time
        start = time.time()
        while time.time() - start < timeout:
            if self.is_loaded(collection_name):
                print("集合加载完成")
                return True
            time.sleep(1)
        
        print("加载超时")
        return False

if __name__ == "__main__":
    # 初始化Milvus客户端
    milvus_client_uri = vdb_config.VDB_URI
    client = MilvusClient(uri = milvus_client_uri)
    data_loader = DataLoader(client)
    
    dataset_num = len(vdb_config.DATASET_NAME)
    for i in range(dataset_num):
        dataset_name = vdb_config.DATASET_NAME[i]
        vector_file_path = vdb_config.DATASET_VECTOR_PATH[i]
        data_list = data_loader.read_data(vector_file_path)
        # print(data_list[0])

        schema_field_config = vdb_config.SCHEMA_FIELD_CONFIG[i]
        data_loader.create_schema(dataset_name, schema_field_config)

        data_loader.load_data(dataset_name, data_list)

        index_params = vdb_config.INDEX_PARAMS[i]
        # print(index_params)
        data_loader.create_index(dataset_name, index_params)
