from pymilvus import MilvusClient, DataType

# VdbConfig.py
class VdbConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init_config()
        return cls._instance

    def init_config(self):
        # 修改为你自己的前缀
        YOUR_PREFIX = "ZYX"             
        # 修改为不同的数据集名称 ["WIT", "Youtube_audio", "Youtube_rgb"]
        dataset_name = "Youtube_rgb"            
        # 在 L2 和 IP 之间切换
        DISTANCE_TYPE = "L2"            
        # 根据数据集名称确定数据集维度
        if dataset_name == "Youtube_audio":
            dataset_dim = 128
        elif dataset_name == "Youtube_rgb":
            dataset_dim = 1024
        elif dataset_name == "WIT":
            dataset_dim = 2048
        else:
            # 若数据集名称未知，抛出异常
            raise ValueError("Unknown dataset")
        
        self.VDB_URI = "http://localhost:50055"
        self.DATASET_NAME = [
            f"{YOUR_PREFIX}_EXACT_{dataset_name}",
            f"{YOUR_PREFIX}_APPROX_{dataset_name}",
        ]
        self.DATASET_VECTOR_PATH = [
            f"/home/dataset/Seminar2025Fall/{dataset_name}/vector_0.fivecs",
            f"/home/dataset/Seminar2025Fall/{dataset_name}/vector_0.fivecs",
        ]
        self.DATASET_ATTR_PATH = [
            f"/home/dataset/Seminar2025Fall/{dataset_name}/meta_0.txt",
            f"/home/dataset/Seminar2025Fall/{dataset_name}/meta_0.txt",
        ]
        if dataset_name == "WIT":
            self.SCHEMA_FIELD_CONFIG = [
                [
                    {"name": "id", "dtype": DataType.INT64, "is_primary": True, "description": "primary key"},
                    {"name": "vector", "dtype": DataType.FLOAT_VECTOR, "dim": dataset_dim, "description": "vector"},
                    {"name": "size", "dtype": DataType.INT64, "description": "image size"},
                ],
                [
                    {"name": "id", "dtype": DataType.INT64, "is_primary": True, "description": "primary key"},
                    {"name": "vector", "dtype": DataType.FLOAT_VECTOR, "dim": dataset_dim, "description": "vector"},
                    {"name": "size", "dtype": DataType.INT64, "description": "image size"},
                ],
            ]
        else:
            self.SCHEMA_FIELD_CONFIG = [
                [
                    {"name": "id", "dtype": DataType.INT64, "is_primary": True, "description": "primary key"},
                    {"name": "vector", "dtype": DataType.FLOAT_VECTOR, "dim": dataset_dim, "description": "vector"},
                    {"name": "label", "dtype": DataType.VARCHAR, "max_length": 50, "description": "YouTube category"},
                ],
                [
                    {"name": "id", "dtype": DataType.INT64, "is_primary": True, "description": "primary key"},
                    {"name": "vector", "dtype": DataType.FLOAT_VECTOR, "dim": dataset_dim, "description": "vector"},
                    {"name": "label", "dtype": DataType.VARCHAR, "max_length": 50, "description": "YouTube category"},
                ],
            ]
        self.INDEX_PARAMS = [
            {
                "field_name": "vector",
                "metric_type": DISTANCE_TYPE,
                "index_type": "FLAT",
                "index_name": "flat_index",
            },
            {
                "field_name": "vector",
                "metric_type": DISTANCE_TYPE,
                "index_type": "HNSW",
                "index_name": "hnsw_index",
                "params": {"M": 32, "efConstruction": 512},
            },
        ]
        self.QUERY_WORKLOAD = [
            {"collection_name": f"{YOUR_PREFIX}_EXACT_{dataset_name}", "query_file_path": f"/home/dataset/Seminar2025Fall/{dataset_name}/query.txt"},
            {"collection_name": f"{YOUR_PREFIX}_APPROX_{dataset_name}", "query_file_path": f"/home/dataset/Seminar2025Fall/{dataset_name}/query.txt"},
        ]
        self.SEARCH_PARAMS = [
            {"metric_type": DISTANCE_TYPE},
            {"metric_type": DISTANCE_TYPE, "params": {"ef": 32}},
        ]

# 单例模式保证全局唯一
vdb_config = VdbConfig()