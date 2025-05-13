from pymilvus import MilvusClient
from FileIO import read_query, dump2json
import time, sys
from pymilvus import connections, Collection, utility
from VdbConfig import vdb_config


class QueryProcessor:
    def __init__(self, milvus_client: MilvusClient):
        """
        初始化 QueryProcessor 类
        
        Args:
            milvus_client (MilvusClient): Milvus 客户端实例
        """
        self.client = milvus_client
        

    def knn_search(self, collection_name, search_field_name, query_vector, top_k, search_params):
        """
        KNN查询
        :param collection_name: 集合名称
        :param search_field_name: 带搜索的字段
        :param query_vector: 查询向量 (list/np.array)
        :param top_k: 返回最相似的 k 个结果
        :param search_params: 搜索参数 (可选)
        :return: (结果列表, 耗时(毫秒))
        """
        collection = Collection(collection_name, using=self.client._using)
        collection.load()
        # print(f"top = {top_k}")

        # 执行搜索
        start_time = time.time()
        result_list = collection.search(
            data=[query_vector],
            anns_field=search_field_name,
            param=search_params,
            limit=top_k,
            output_fields=["id"],
        )
        result_list = result_list[0]
        latency = (time.time() - start_time) * 1000.0
        for result in result_list:
            print(f"result = {{ {result} }}")
        
        return result_list, latency
    

    def hybrid_search(self, collection_name, search_field_name, query_vector, filter_expr, top_k, search_params):
        """
        混合查询
        :param collection_name: 集合名称
        :param search_field_name: 带搜索的字段
        :param query_vector: 查询向量 (list/np.array)
        :param filter_expr: 关系型属性过滤条件
        :param top_k: 返回最相似的 k 个结果
        :param search_params: 搜索参数 (可选)
        :return: (结果列表, 耗时(毫秒))
        """
        collection = Collection(collection_name, using=self.client._using)
        collection.load()
        # print(f"top = {top_k}")

        # 执行搜索
        start_time = time.time()
        result_list = collection.search(
            data=[query_vector],
            anns_field=search_field_name,
            param=search_params,
            expr=filter_expr,
            limit=top_k,
            output_fields=["id"],
        )
        result_list = result_list[0]
        latency = (time.time() - start_time) * 1000.0
        print(f"filter condition: ${filter_expr}")
        for result in result_list:
            print(f"result = {{ {result} }}")
        
        return result_list, latency


    def calculate_recall(self, true_list, result_list):
        """计算召回率"""
        true_ids, result_ids = [], []
        for entity_dict in true_list:
            true_ids.append(entity_dict.id)
        for entity_dict in result_list:
            result_ids.append(entity_dict.id)
        true_set = set(true_ids)
        if len(true_set)==0:
            return 0.0
        result_set = set(result_ids)
        intersection = true_set.intersection(result_set)
        return len(intersection)*1.0 / len(true_set)
    

    def search_performance(self, result_list, truth_list):
        """
        打印查询处理性能（包括查询时间与召回率）
        :param result_list: [[result_ids, time]]
        :param truth_list: [[truth_ids, time]]
        """   
        query_time_list, query_recall_list = [], []     
        avg_query_time,avg_query_recall = 0.0, 0.0
        for query_id in range(len(result_list)):
            result_ids = result_list[query_id][0]
            truth_ids = truth_list[query_id][0]
            query_time = result_list[query_id][1]
            query_recall = self.calculate_recall(truth_ids, result_ids)

            avg_query_time += query_time
            avg_query_recall += query_recall

            query_time_list.append(query_time)
            query_recall_list.append(query_recall)

        if len(result_list) > 0:
            avg_query_time /= len(result_list)
            avg_query_recall /= len(result_list)

        print(f"(Average) search time: {avg_query_time:.3f} ms, result recall: {avg_query_recall*100:.1f}%")
        return query_time_list, query_recall_list


def DumpResult(filename, query_time_list, query_recall_list):
    with open(filename, "w") as fout:
        fout.write(f"{query_time_list}")
        fout.write("\n")
        fout.write(f"{query_recall_list}")
        fout.write("\n")


if __name__ == "__main__":
    # 初始化Milvus客户端
    milvus_client_uri = vdb_config.VDB_URI
    client = MilvusClient(uri = milvus_client_uri)
    query_processor = QueryProcessor(client)
    top_k = 1
    
    ## 测试KNN查询
    print("Test KNN Search")
    result_list = []
    truth_list = []
    for idx,query_dict in enumerate(vdb_config.QUERY_WORKLOAD):
        collection_name = query_dict["collection_name"]
        query_file_path = query_dict["query_file_path"]
        query_vector_list, attr_filter_list = read_query(query_file_path)

        search_params = vdb_config.SEARCH_PARAMS[idx]
        print(f"search_params = {search_params}")

        for i in range(len(query_vector_list)):
            query_vector, attr_filter = query_vector_list[i], attr_filter_list[i]
            search_field_name = "vector"
            result = query_processor.knn_search(collection_name, search_field_name, query_vector, top_k, search_params)
            if idx==0:
                truth_list.append(result)
            else:
                result_list.append(result)

    print("Search performance of Index FLAT:")
    flat_query_time_list, flat_query_recall_list = query_processor.search_performance(truth_list, truth_list)
    DumpResult("flat.log", flat_query_time_list, flat_query_recall_list)

    print("="*64)
    print("Search performance of Index HNSW:")
    hnsw_query_time_list, hnsw_query_recall_list = query_processor.search_performance(result_list, truth_list)
    DumpResult("hnsw.log", hnsw_query_time_list, hnsw_query_recall_list)
    sys.exit(0)

    ## 测试混合查询
    print("Test Hybrid Search")
    result_list = []
    truth_list = []
    for idx,query_dict in enumerate(vdb_config.QUERY_WORKLOAD):
        collection_name = query_dict["collection_name"]
        query_file_path = query_dict["query_file_path"]
        query_vector_list, attr_filter_list = read_query(query_file_path)

        search_params = vdb_config.SEARCH_PARAMS[idx]
        print(f"search_params = {search_params}")

        for i in range(len(query_vector_list)):
            query_vector, attr_filter = query_vector_list[i], attr_filter_list[i]
            search_field_name = "vector"
            result = query_processor.hybrid_search(collection_name, search_field_name, query_vector, attr_filter, top_k, search_params)
            if idx==0:
                truth_list.append(result)
            else:
                result_list.append(result)

    print("Search performance of Index FLAT:")
    flat_query_time_list, flat_query_recall_list = query_processor.search_performance(truth_list, truth_list)
    DumpResult("flat.log", flat_query_time_list, flat_query_recall_list)

    print("="*64)
    print("Search performance of Index HNSW:")
    hnsw_query_time_list, hnsw_query_recall_list = query_processor.search_performance(result_list, truth_list)
    DumpResult("hnsw.log", hnsw_query_time_list, hnsw_query_recall_list)