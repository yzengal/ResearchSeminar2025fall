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
        YOUR_PREFIX = "ZYX"             # change into your own prefix
        dataset_name = "LoTTE"          # change into different dataset names
        DISTANCE_TYPE = "IP"            # change between L2 and IP
        dataset_dim = 128
        self.VDB_URI = "http://localhost:50055"
        self.DATASET_NAME = [
            f"{YOUR_PREFIX}_EXACT_{dataset_name}",
            f"{YOUR_PREFIX}_APPROX_{dataset_name}",
        ]
        self.DATASET_VECTOR_PATH = [
            f"./lotte-lifestyle-data-small.fivecs",
            f"./lotte-lifestyle-data-small.fivecs",
        ]
        self.SCHEMA_FIELD_CONFIG = [
            [
                {"name": "id", "dtype": DataType.INT64, "is_primary": True, "description": "primary key"},
                {"name": "vector", "dtype": DataType.FLOAT_VECTOR, "dim": dataset_dim, "description": "vector"},
                {"name": "doc", "dtype": DataType.INT64, "description": "Doc id"},
            ],
            [
                {"name": "id", "dtype": DataType.INT64, "is_primary": True, "description": "primary key"},
                {"name": "vector", "dtype": DataType.FLOAT_VECTOR, "dim": dataset_dim, "description": "vector"},
                {"name": "doc", "dtype": DataType.INT64, "description": "Doc id"},
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
            {"collection_name": f"{YOUR_PREFIX}_EXACT_{dataset_name}", "query_file_path": f"./lotte-lifestyle-query-small.fivecs"},
            {"collection_name": f"{YOUR_PREFIX}_APPROX_{dataset_name}", "query_file_path": f"./lotte-lifestyle-query-small.fivecs"},
        ]
        self.SEARCH_PARAMS = [
            {"metric_type": DISTANCE_TYPE},
            {"metric_type": DISTANCE_TYPE, "params": {"ef": 32}},
        ]

# 单例模式保证全局唯一
vdb_config = VdbConfig()