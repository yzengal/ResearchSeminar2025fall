import numpy as np
from pymilvus import MilvusClient
from FileIO import read_fivecs
import time, sys
from pymilvus import connections, Collection, utility
from VdbConfig import vdb_config
from tqdm import tqdm

class QueryProcessor:
    def __init__(self, milvus_client: MilvusClient):
        """
        初始化 QueryProcessor 类
        
        Args:
            milvus_client (MilvusClient): Milvus 客户端实例
        """
        self.client = milvus_client
    

    @staticmethod
    def _has_nested_list(lst):
        if not isinstance(lst, list):
            return False
        for element in lst:
            if not isinstance(element, list):
                return False
        return True

    def _hybrid_search(self, collection, search_field_name, query_vectors, filter_expr, top_k, search_params):
        """
        混合查询
        :param collection_name: 向量数据库
        :param search_field_name: 带搜索的字段
        :param query_vector: 查询向量 (list/np.array)
        :param filter_expr: 关系型属性过滤条件
        :param top_k: 返回最相似的 k 个结果
        :param search_params: 搜索参数 (可选)
        :return: (结果列表, 耗时(毫秒))
        """
        if self._has_nested_list(query_vectors):
            query_vector_list = query_vectors
        else:
            query_vector_list = [query_vectors]

        result_list = collection.search(
            data=query_vector_list,
            anns_field=search_field_name,
            param=search_params,
            expr=filter_expr,
            limit=top_k,
            output_fields=["doc"],
        )
        # print(result_list)

        ret_list = []
        for hits in result_list:
            for hit in hits:
                # print(f"ID: {hit.id}")  # 获取实体的 ID
                # print(f"Distance/Score: {hit.distance}")  # 获取距离或相似度分数
                # print(f"Doc: {hit.entity.get('doc')}")  # 获取输出字段 "doc" 的值
                ret_list.append(hit.distance)

        return sum(ret_list)


    def _scan_all_doc(self, collection, doc_list, query_vectors, top_k, search_params):         
        search_field_name = "vector"
        maxsim_list = []
        for doc_id in doc_list:
            filter_expr = f"doc == {doc_id}"
            similarity_score = self._hybrid_search(
                                    collection=collection, 
                                    search_field_name=search_field_name, 
                                    query_vectors=query_vectors, 
                                    filter_expr=filter_expr, 
                                    top_k=1, 
                                    search_params=search_params)
            maxsim_list.append(similarity_score)
        
        similarity_list = np.array(maxsim_list)
        sorted_indices = np.argsort(similarity_list)[::-1]
        return sorted_indices[:top_k]
        

    def _process_vectors(self, vector_file_path):
        data_list = read_fivecs(vector_file_path)
        query_id = -1
        queries = []
        tmpList = []
        for vector_id, doc_id, embedding in data_list:
            if doc_id == query_id:
                tmpList.append(embedding)
            else:
                if query_id >= 0:
                    queries.append(tmpList)
                tmpList = [embedding]
                query_id = doc_id
        if len(tmpList) > 0:
            queries.append(tmpList)
        return queries

    @staticmethod
    def _calculate_maxsim_score(q_i, d_k):
        ret = 0
        for q_ij in q_i:
            max_score = np.max(np.dot(q_ij, d_k.T))
            ret += max_score
        return ret

    def _multi_vector_search_byNumpy(self, 
                                    vector_file_path: str, 
                                    query_file_path: str, 
                                    top_k: int, 
                                    search_params: dict):

        queries = self._process_vectors(query_file_path)
        vectors = self._process_vectors(vector_file_path)

        result_list = []
        num_queries = len(queries)
        num_docs = len(vectors)

        for i in range(num_queries):
            start_time = time.time()
            query = np.array(queries[i], dtype=np.float32)

            scores = []
            for j in range(num_docs):
                vector = np.array(vectors[j], dtype=np.float32)

                score_ij = self._calculate_maxsim_score(query, vector)
                scores.append((j, score_ij))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            top_k_docs = [doc_idx for doc_idx, _ in scores[:top_k]]
            assert len(top_k_docs) == top_k
            result_list.append(top_k_docs)

            latency = (time.time() - start_time) * 1000.0
            print(f"Latency: {latency} ms")

        return result_list



    def _multi_vector_search_byDB(self, 
                                    collection_name: str, 
                                    query_file_path: str, 
                                    top_k: int, 
                                    search_params: dict):

        queries = self._process_queries(query_file_path)

        collection = Collection(collection_name, using=self.client._using)
        collection.load()
        results = collection.query(expr="id >= 0", output_fields=["doc"])
        doc_list = [item['doc'] for item in results]
        doc_list = list(set(doc_list))
        doc_list.sort()

        result = []
        for query_vector in tqdm(queries, total=len(queries), desc="Multi-vector search"):
            start_time = time.time()
            answer_idx_list = self._scan_all_doc(collection=collection, doc_list=doc_list, query_vectors=query_vector, top_k=top_k, search_params=search_params)
            answer_doc_id = [doc_list[idx] for idx in answer_idx_list]
            latency = (time.time() - start_time) * 1000.0
            print(f"Latency: {latency} ms")
            result.append(answer_doc_id)
        return result


    def multi_vector_search(self, 
                            collection_name: str, 
                            query_file_path: str, 
                            top_k: int, 
                            search_params: dict,
                            mode: str="ByDB"):
        if mode not in ["ByDB", "ByNumpy"]:
             raise ValueError("mode must be one of 'ByDB' or 'ByNumpy'")

        if mode == "ByDB":
            return self._multi_vector_search_byDB(collection_name, query_file_path, top_k, search_params)
        else:
            return self._multi_vector_search_byNumpy(collection_name, query_file_path, top_k, search_params)


if __name__ == "__main__":
    milvus_client_uri = vdb_config.VDB_URI
    client = MilvusClient(uri = milvus_client_uri)
    query_processor = QueryProcessor(client)
    top_k = 20 
    mode = "ByNumpy"

    for idx,query_dict in enumerate(vdb_config.QUERY_WORKLOAD):
        collection_name = query_dict["collection_name"]
        if "EXACT" not in collection_name:
            continue
        if mode=="ByNumpy":
            collection_name = vdb_config.DATASET_VECTOR_PATH[idx]
        query_file_path = query_dict["query_file_path"]
        search_params = vdb_config.SEARCH_PARAMS[idx]
        print(f"search_params = {search_params}")
    
        result = query_processor.multi_vector_search(
                    collection_name=collection_name, 
                    query_file_path=query_file_path, 
                    top_k=top_k, 
                    search_params=search_params,
                    mode=mode)
        
        with open(f"ground_truth_{mode}.txt", "w") as fout:
            fout.write(f"{len(result)}\n")
            for answer_doc_list in result:
                line = " ".join(map(str, answer_doc_list))
                fout.write(line)
                fout.write("\n")
