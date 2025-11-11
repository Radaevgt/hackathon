"""
BM25 Sparse Retrieval
"""

from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from loguru import logger

try:
    from razdel import tokenize
    RAZDEL_AVAILABLE = True
except ImportError:
    RAZDEL_AVAILABLE = False
    logger.warning("razdel not available, using simple tokenization")

from src.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """
    BM25 (Best Match 25) sparse retrieval
    
    Преимущества:
    - Быстрый (нет neural inference)
    - Хорошо работает на exact match
    - Interpretable scores
    
    Best practices:
    - Русская токенизация через razdel
    - Lowercase normalization
    - Без удаления стоп-слов (вредит для queries)
    
    Формула BM25:
        BM25(Q, D) = Σ IDF(t) · [f(t,D) · (k₁+1)] / [f(t,D) + k₁ · (1 - b + b·|D|/avgDL)]
    
    Args:
        k1: Saturation parameter (default: 1.5)
        b: Length normalization (default: 0.75)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__(name="BM25Retriever")
        
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.tokenized_corpus = None
        
        logger.info(f"BM25Retriever initialized: k1={k1}, b={b}")
    
    def build_index(self, documents: List[str]):
        """
        Построение BM25 индекса
        
        Args:
            documents: Список текстов документов
        """
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        
        # Токенизация корпуса
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        
        # Построение BM25 индекса
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )
        
        self.is_built = True
        logger.info(f"BM25 index built successfully")
    
    def search(self, query: str, k: int = 5) -> List[int]:
        """
        Поиск топ-k документов для запроса
        
        Args:
            query: Текст запроса
            k: Число результатов
            
        Returns:
            Список индексов документов (отсортированы по убыванию релевантности)
        """
        self._check_built()
        
        # Токенизация запроса
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            logger.warning("Empty query after tokenization")
            return list(range(min(k, len(self.tokenized_corpus))))
        
        # Получение BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Топ-k индексов
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        return top_k_indices.tolist()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Токенизация текста для BM25
        
        Args:
            text: Исходный текст
            
        Returns:
            Список токенов
        """
        if not text:
            return []
        
        # Lowercase
        text = text.lower()
        
        # Токенизация
        if RAZDEL_AVAILABLE:
            # Используем razdel для русского языка
            tokens = [t.text for t in tokenize(text)]
        else:
            # Fallback: простая токенизация по пробелам
            tokens = text.split()
        
        # Фильтрация
        tokens = [
            t for t in tokens
            if len(t) > 2 and t.isalnum()  # Длина > 2, буквы/цифры
        ]
        
        return tokens
    
    def get_scores(self, query: str) -> np.ndarray:
        """
        Получение BM25 scores для всех документов
        
        Args:
            query: Текст запроса
            
        Returns:
            Массив scores для каждого документа
        """
        self._check_built()
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        return scores
