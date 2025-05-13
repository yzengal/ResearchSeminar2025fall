import numpy as np
import struct
from tqdm import tqdm
import os, sys
from io import BytesIO

def write_fivecs(file_name, data_list, chunk_size=4096):
    # First pass: validate data and collect metadata
    total_docs = len(data_list)
    total_vectors = sum(len(embeddings) for embeddings in data_list)
    dim = len(data_list[0][0]) if total_vectors > 0 else 0
    
    # Use buffer for batch writing (adjust chunk_size as needed)
    buffer = BytesIO()
    
    with open(file_name, 'wb') as file:
        # Write header
        file.write(struct.pack('<3q', total_vectors, total_docs, dim))
        
        vector_id = 0
        current_chunk_count = 0
        
        for doc_id, embeddings in tqdm(enumerate(data_list), 
                                        total=len(data_list),
                                        desc="Writing vector data file"):
            for embedding in embeddings:
                # Write to buffer
                buffer.write(struct.pack('<2q', vector_id, doc_id))
                buffer.write(struct.pack(f'<{dim}d', *embedding))
                
                vector_id += 1
                current_chunk_count += 1
                
                # Flush buffer when chunk is full
                if current_chunk_count >= chunk_size:
                    file.write(buffer.getvalue())
                    buffer.seek(0)
                    buffer.truncate()
                    current_chunk_count = 0
        
        # Write remaining data in buffer
        if current_chunk_count > 0:
            file.write(buffer.getvalue())


def read_fivecs(file_name, chunk_size=4096):  
    with open(file_name, 'rb') as file:
        # Read header
        header = file.read(24)
        total_vectors, total_docs, dim = struct.unpack('<3q', header)
        print(f"#(vectors) = {total_vectors}, #(docs) = {total_docs}, #(dim) = {dim}")
        
        # Pre-allocate memory for all vectors
        data = []
        
        # Read data in chunks
        for start_vector_id in tqdm(range(0, total_vectors, chunk_size), 
                                    total=total_vectors,
                                    desc="Reading vector data file"):
            # Calculate bytes to read in this chunk
            vectors_in_chunk = min(total_vectors-start_vector_id, chunk_size)
            bytes_to_read = vectors_in_chunk * (16 + dim * 8)
            
            # Read raw chunk
            chunk = file.read(bytes_to_read)
            if len(chunk) != bytes_to_read:
                raise EOFError("Unexpected end of file")
            
            # Process chunk
            offset = 0
            for _ in range(vectors_in_chunk):
                # Unpack vector_id & doc_id
                vector_id, doc_id = struct.unpack_from('<2q', chunk, offset)
                offset += 16
                
                # Unpack embedding
                embedding = list(struct.unpack_from(f'<{dim}d', chunk, offset))
                offset += dim * 8
                
                # Store data
                data.append([vector_id, doc_id, embedding])

    return data