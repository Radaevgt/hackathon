"""
Dense Retrieval с использованием E5 embeddings
"""

from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
from loguru import logger

from src.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """
    Dense retrieval используя multilingual-e5-large embeddings
    
    Преимущества:
    - Semantic search (понимает смысл, не только keywords)
    - Обучен на контрастивном learning
    - Multilingual (включая русский)
    
    FAISS индекс типы:
    - Flat: Exact search, медленный для N>10K
    - IVFFlat: Approximate search с кластерами
    - IVFPQ: Compressed с Product Quantization
    
    Args:
        model_name: Название модели из HuggingFace
        use_gpu: Использовать GPU для FAISS (если доступен)
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
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        # Загрузка модели
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self.index = None
        self.num_documents = 0
        
        logger.info(f"Model loaded: embedding_dim={self.dimension}")
    
    def build_index(
        self,
        documents: List[str],
        index_type: str = "Flat",
        nlist: int = 100
    ):
        """
        Построение FAISS индекса
        
        Args:
            documents: Список текстов документов
            index_type: Тип индекса (Flat, IVFFlat, IVFPQ)
            nlist: Число кластеров для IVF индексов
        """
        logger.info(f"Building dense index for {len(documents)} documents...")
        logger.info(f"Index type: {index_type}, nlist: {nlist}")
        
        self.num_documents = len(documents)
        
        # Кодирование документов
        embeddings = self._encode_batch(documents, desc="Encoding documents")
        
        # Выбор типа индекса
        if index_type == "Flat":
            self.index = self._build_flat_index(embeddings)
        elif index_type == "IVFFlat":
            self.index = self._build_ivf_flat_index(embeddings, nlist)
        elif index_type == "IVFPQ":
            self.index = self._build_ivf_pq_index(embeddings, nlist)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.is_built = True
        logger.info(f"Dense index built: {self.index.ntotal} vectors")
    
    def _build_flat_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Построение Flat индекса (exact search)"""
        index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine for normalized)
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Using GPU for FAISS")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        index.add(embeddings)
        return index
    
    def _build_ivf_flat_index(self, embeddings: np.ndarray, nlist: int) -> faiss.Index:
        """Построение IVF Flat индекса (approximate search)"""
        quantizer = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        # Обучение кластеров
        logger.info(f"Training IVF index with {nlist} clusters...")
        index.train(embeddings)
        
        # GPU (опционально)
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        index.add(embeddings)
        return index
    
    def _build_ivf_pq_index(self, embeddings: np.ndarray, nlist: int, m: int = 96) -> faiss.Index:
        """Построение IVF PQ индекса (compressed)"""
        quantizer = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, 8)
        
        logger.info(f"Training IVF-PQ index with {nlist} clusters and {m} subquantizers...")
        index.train(embeddings)
        index.add(embeddings)
        
        return index
    
    def search(self, query: str, k: int = 5, nprobe: int = 10) -> List[Tuple[int, float]]:
        """
        Поиск топ-k документов
        
        Args:
            query: Текст запроса
            k: Число результатов
            nprobe: Число проверяемых кластеров (для IVF индексов)
            
        Returns:
            List[(doc_idx, similarity_score)]
        """
        self._check_built()
        
        # ВАЖНО: E5 требует префикс "query:" для запросов!
        query_text = f"query: {query}"
        
        # Кодирование запроса
        query_emb = self._encode_batch([query_text], desc=None)[0:1]
        
        # Настройка nprobe для IVF
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        
        # Поиск
        distances, indices = self.index.search(query_emb, k)
        
        # Формирование результатов
        results = [
            (int(idx), float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx >= 0  # FAISS возвращает -1 для пустых слотов
        ]
        
        return results
    
    def _encode_batch(
        self,
        texts: List[str],
        desc: Optional[str] = None
    ) -> np.ndarray:
        """
        Пакетное кодирование текстов
        
        Args:
            texts: Список текстов
            desc: Описание для progress bar
            
        Returns:
            Массив embeddings shape (N, dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=desc is not None,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        return embeddings.astype('float32')  # FAISS требует float32
    
    def save_index(self, path: str):
        """
        Сохранение FAISS индекса
        
        Args:
            path: Путь для сохранения
        """
        self._check_built()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Если GPU индекс, переносим на CPU перед сохранением
        if isinstance(self.index, faiss.GpuIndex):
            index_cpu = faiss.index_gpu_to_cpu(self.index)
        else:
            index_cpu = self.index
        
        faiss.write_index(index_cpu, str(path))
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """
        Загрузка FAISS индекса
        
        Args:
            path: Путь к сохраненному индексу
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        
        self.index = faiss.read_index(str(path))
        self.num_documents = self.index.ntotal
        
        # GPU (опционально)
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.is_built = True
        logger.info(f"Index loaded from {path}: {self.num_documents} vectors")
