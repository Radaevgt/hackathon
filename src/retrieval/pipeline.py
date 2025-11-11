"""
Главный Retrieval Pipeline - объединение всех компонентов
"""

from typing import List, Dict, Optional, Tuple
from loguru import logger

from src.retrieval.base import BaseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.fusion.rrf import ReciprocalRankFusion, WeightedFusion


class RetrievalPipeline:
    """
    Главный pipeline для retrieval
    
    Архитектура:
    1. Multiple retrievers (BM25, Dense)
    2. Fusion strategy (RRF)
    3. Optional reranking
    
    Strategy Pattern для гибкой конфигурации
    
    Args:
        config: Словарь конфигурации
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.retrievers: List[BaseRetriever] = []
        self.fusion = None
        self.reranker = None
        
        self._init_retrievers()
        self._init_fusion()
        
        logger.info(f"RetrievalPipeline initialized with {len(self.retrievers)} retrievers")
    
    def _init_retrievers(self):
        """Инициализация retriever'ов из конфигурации"""
        retrieval_config = self.config.get('retrieval', {})
        
        # BM25
        if retrieval_config.get('use_bm25', True):
            bm25_config = retrieval_config.get('bm25', {})
            bm25 = BM25Retriever(
                k1=bm25_config.get('k1', 1.5),
                b=bm25_config.get('b', 0.75)
            )
            self.retrievers.append(bm25)
            logger.info("Added BM25Retriever")
        
        # Dense
        if retrieval_config.get('use_dense', True):
            dense_config = retrieval_config.get('dense', {})
            dense = DenseRetriever(
                model_name=dense_config.get('model_name', 'intfloat/multilingual-e5-large'),
                use_gpu=dense_config.get('use_gpu', False),
                batch_size=dense_config.get('batch_size', 32),
                normalize_embeddings=dense_config.get('normalize_embeddings', True)
            )
            self.retrievers.append(dense)
            logger.info("Added DenseRetriever")
    
    def _init_fusion(self):
        """Инициализация fusion стратегии"""
        retrieval_config = self.config.get('retrieval', {})
        fusion_type = retrieval_config.get('fusion_type', 'rrf')
        
        if fusion_type == 'rrf':
            rrf_k = retrieval_config.get('rrf_k', 60)
            self.fusion = ReciprocalRankFusion(k=rrf_k)
            logger.info(f"Using RRF fusion with k={rrf_k}")
        
        elif fusion_type == 'weighted':
            alpha = retrieval_config.get('alpha', 0.3)
            self.fusion = WeightedFusion(alpha=alpha)
            logger.info(f"Using Weighted fusion with alpha={alpha}")
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def build_indices(self, documents: List[str]):
        """
        Построение индексов для всех retriever'ов
        
        Args:
            documents: Список текстов документов
        """
        logger.info(f"Building indices for {len(documents)} documents...")
        
        for retriever in self.retrievers:
            logger.info(f"Building index for {retriever.name}...")
            
            if isinstance(retriever, DenseRetriever):
                # Для Dense retriever передаем дополнительные параметры
                dense_config = self.config.get('retrieval', {}).get('dense', {})
                retriever.build_index(
                    documents,
                    index_type=dense_config.get('index_type', 'Flat'),
                    nlist=dense_config.get('nlist', 100)
                )
            else:
                retriever.build_index(documents)
        
        logger.info("All indices built successfully")
    
    def search(self, query: str, k: int = 5) -> List[int]:
        """
        Поиск топ-k документов для запроса
        
        Args:
            query: Текст запроса
            k: Число результатов
            
        Returns:
            Список индексов топ-k документов
        """
        retrieval_config = self.config.get('retrieval', {})
        
        # Stage 1: Получение результатов от каждого retriever'а
        all_rankings = []
        
        for retriever in self.retrievers:
            if isinstance(retriever, BM25Retriever):
                top_k = retrieval_config.get('bm25', {}).get('top_k', 100)
                ranking = retriever.search(query, k=top_k)
                all_rankings.append(ranking)
            
            elif isinstance(retriever, DenseRetriever):
                top_k = retrieval_config.get('dense', {}).get('top_k', 100)
                nprobe = retrieval_config.get('dense', {}).get('nprobe', 10)
                
                # Dense возвращает (idx, score)
                results = retriever.search(query, k=top_k, nprobe=nprobe)
                ranking = [idx for idx, _ in results]
                all_rankings.append(ranking)
        
        # Stage 2: Fusion
        if len(all_rankings) == 1:
            # Только один retriever
            fused_ranking = all_rankings[0]
        else:
            fused_ranking = self.fusion.fuse(all_rankings)
        
        # Stage 3: Reranking (опционально)
        if retrieval_config.get('use_reranking', False):
            logger.warning("Reranking not implemented yet")
            # TODO: Implement reranking
        
        # Возвращаем топ-k
        return fused_ranking[:k]
    
    def batch_search(self, queries: List[str], k: int = 5) -> List[List[int]]:
        """
        Пакетный поиск для нескольких запросов
        
        Args:
            queries: Список запросов
            k: Число результатов для каждого запроса
            
        Returns:
            Список результатов для каждого запроса
        """
        results = []
        
        from tqdm import tqdm
        for query in tqdm(queries, desc="Searching"):
            result = self.search(query, k=k)
            results.append(result)
        
        return results
    
    def save_indices(self, output_dir: str):
        """
        Сохранение индексов всех retriever'ов
        
        Args:
            output_dir: Директория для сохранения
        """
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, retriever in enumerate(self.retrievers):
            index_path = output_path / f"{retriever.name}_index_{i}.faiss"
            retriever.save_index(str(index_path))
        
        logger.info(f"Indices saved to {output_dir}")
    
    def load_indices(self, input_dir: str):
        """
        Загрузка индексов для всех retriever'ов
        
        Args:
            input_dir: Директория с сохраненными индексами
        """
        from pathlib import Path
        input_path = Path(input_dir)
        
        for i, retriever in enumerate(self.retrievers):
            index_path = input_path / f"{retriever.name}_index_{i}.faiss"
            if index_path.exists():
                retriever.load_index(str(index_path))
            else:
                logger.warning(f"Index not found: {index_path}")
        
        logger.info(f"Indices loaded from {input_dir}")
    
    def __repr__(self) -> str:
        return (
            f"RetrievalPipeline("
            f"retrievers={[r.name for r in self.retrievers]}, "
            f"fusion={self.fusion.__class__.__name__})"
        )
