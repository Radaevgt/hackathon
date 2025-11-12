"""
Dense Retrieval с GPU acceleration для encoding
"""

from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
from pathlib import Path
from loguru import logger

from src.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """
    Dense retrieval с GPU для encoding, CPU для FAISS
    
    ВАЖНО: На Windows FAISS-GPU недоступен через pip,
    поэтому используем гибридный подход:
    - GPU для encoding (10x ускорение)
    - CPU для FAISS index (поиск и так быстрый)
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        use_gpu: bool = False,
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ):
        super().__init__(name="DenseRetriever")
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        # Проверка CUDA
        self.device = "cpu"
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            logger.info(f"🚀 GPU encoding will be ~10x faster than CPU")
        elif use_gpu:
            logger.warning("⚠️ GPU requested but CUDA not available, using CPU")
        else:
            logger.info("Using CPU for encoding")
        
        # Загрузка модели
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self.index = None
        self.num_documents = 0
        
        logger.info(f"Model loaded: embedding_dim={self.dimension}, device={self.device}")
    
    def build_index(
        self,
        documents: List[str],
        index_type: str = "Flat",
        nlist: int = 100
    ):
        """Построение FAISS индекса с GPU encoding"""
        logger.info(f"Building dense index for {len(documents)} documents...")
        logger.info(f"Index type: {index_type}, Device: {self.device}")
        
        self.num_documents = len(documents)
        
        # 🚀 Кодирование на GPU (если доступен)
        import time
        start_time = time.time()
        
        embeddings = self._encode_batch(documents, desc="Encoding documents")
        
        encoding_time = time.time() - start_time
        logger.info(f"⚡ Encoding completed in {encoding_time:.1f}s ({len(documents)/encoding_time:.1f} docs/s)")
        
        # Построение FAISS индекса (всегда CPU)
        if index_type == "Flat":
            self.index = self._build_flat_index(embeddings)
        elif index_type == "IVFFlat":
            self.index = self._build_ivf_flat_index(embeddings, nlist)
        elif index_type == "IVFPQ":
            self.index = self._build_ivf_pq_index(embeddings, nlist)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.is_built = True
        total_time = time.time() - start_time
        logger.info(f"✅ Dense index built in {total_time:.1f}s: {self.index.ntotal} vectors")
    
    def _build_flat_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Flat индекс (exact search) - всегда CPU"""
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)
        return index
    
    def _build_ivf_flat_index(self, embeddings: np.ndarray, nlist: int) -> faiss.Index:
        """IVF Flat индекс - CPU"""
        quantizer = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        logger.info(f"Training IVF index with {nlist} clusters...")
        index.train(embeddings)
        index.add(embeddings)
        
        return index
    
    def _build_ivf_pq_index(self, embeddings: np.ndarray, nlist: int, m: int = 96) -> faiss.Index:
        """IVF PQ индекс - CPU, compressed"""
        quantizer = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, 8)
        
        logger.info(f"Training IVF-PQ index with {nlist} clusters and {m} subquantizers...")
        index.train(embeddings)
        index.add(embeddings)
        
        return index
    
    def search(self, query: str, k: int = 5, nprobe: int = 10) -> List[Tuple[int, float]]:
        """Поиск с GPU encoding запроса"""
        self._check_built()
        
        # E5 требует префикс для запросов
        query_text = f"query: {query}"
        
        # 🚀 Encoding на GPU
        query_emb = self._encode_batch([query_text], desc=None)[0:1]
        
        # Поиск в CPU индексе (быстро)
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        
        distances, indices = self.index.search(query_emb, k)
        
        results = [
            (int(idx), float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx >= 0
        ]
        
        return results
    
    def _encode_batch(
        self,
        texts: List[str],
        desc: Optional[str] = None
    ) -> np.ndarray:
        """Пакетное кодирование на GPU/CPU"""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=desc is not None,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            device=self.device  # Использует self.device (cuda или cpu)
        )
        
        return embeddings.astype('float32')
    
    def save_index(self, path: str):
        """Сохранение индекса"""
        self._check_built()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path))
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Загрузка индекса"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        
        self.index = faiss.read_index(str(path))
        self.num_documents = self.index.ntotal
        self.is_built = True
        
        logger.info(f"Index loaded from {path}: {self.num_documents} vectors")