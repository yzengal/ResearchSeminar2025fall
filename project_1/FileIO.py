import numpy as np
import struct


class VectorDataType:
    def __init__(self, vid, data):
        self.vid = vid
        self.data = data


def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def read_fvecs(filename, start_idx=0, chunk_size=None):
    """ Read *.fvecs file that contains float32 vectors
    Args:
        :param filename (str): path to *.fvecs file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    idx = start_idx
    vectors = []
    with open(filename, "rb") as f:
        # 跳过前面的向量，直到到达 start_idx
        for _ in range(start_idx):
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # 文件结束
            dim = struct.unpack('i', dim_bytes)[0]
            f.seek(dim * 4, 1)  # 跳过 dim 个浮点数

        # 读取指定数量的向量或所有向量
        if chunk_size is None:
            while True:
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break  # 文件结束
                dim = struct.unpack('i', dim_bytes)[0]
                vec_bytes = f.read(dim * 4)
                if len(vec_bytes) != dim * 4:
                    raise RuntimeError("Error reading file")
                vec = list(struct.unpack(f'{dim}f', vec_bytes))
                # vectors.append({"vector": vec})
                vectors.append({"id": idx, "vector": vec})
                idx += 1
        else:
            for _ in range(chunk_size):
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break  # 文件结束
                dim = struct.unpack('i', dim_bytes)[0]
                vec_bytes = f.read(dim * 4)
                if len(vec_bytes) != dim * 4:
                    raise RuntimeError("Error reading file")
                vec = list(struct.unpack(f'{dim}f', vec_bytes))
                # vectors.append({"vector": vec})
                vectors.append({"id": idx, "vector": vec})
                idx += 1

    return vectors


def read_fivecs(file_name, data_list):
    with open(file_name, 'rb') as file:
        # Read the number of vectors and dimension
        nvecs, = struct.unpack('i', file.read(4))
        dim, = struct.unpack('i', file.read(4))
        print(f"Read data: size = {nvecs}, dimension = {dim}")

        data_list.clear()

        for _ in range(nvecs):
            # Read the vector ID
            vid, = struct.unpack('i', file.read(4))
            # Read the vector data
            vec = struct.unpack(f'{dim}f', file.read(dim * 4))
            if len(vec) != dim:
                raise RuntimeError("Error reading file")

            data_list.append(VectorDataType(vid, vec))


def read_meta(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().splitlines()
        content = content[2:]
    return content


def read_query(file_path):
    data_list = []
    meta_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().splitlines()
        nvecs, dim = map(int, content[0].split())
        for i in range(1, nvecs + 1):
            line = content[i].split()
            vecs = list(map(float, line[:dim]))
            meta_str = " and ".join(line[dim:])
            meta_str = meta_str.replace("<=", "<")
            meta_str = meta_str.replace("=", "==")
            meta_str = meta_str.replace("<", "<=")
            data_list.append(vecs)
            meta_list.append(meta_str)
    return data_list, meta_list


import json


def dump2json(my_list, file_name):
    with open(file_name, 'w') as file:
        json.dump(my_list, file)
