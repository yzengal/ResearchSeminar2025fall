from pymilvus import MilvusClient, Collection
from VdbConfig import vdb_config

def vdb_list_collections(client):
    # 获取所有集合列表
    collections = client.list_collections()
    collections = filter(lambda x:x.startswith("ZYX"), collections)
    
    # 打印所有集合名称
    print(f"Current collections in vector database system ({milvus_client_uri}):")
    for idx,collection_name in enumerate(collections):
        collection = Collection(collection_name, using=client._using)
        cnt = collection.num_entities  
        print(f"Collection #{idx}: {collection_name} with {cnt} entities")

if __name__ == "__main__":
    # 初始化Milvus客户端
    milvus_client_uri = vdb_config.VDB_URI
    client = MilvusClient(
        uri = milvus_client_uri
    )
    vdb_list_collections(client)

