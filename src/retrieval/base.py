"""
Базовый класс для всех retriever'ов
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from loguru import logger


class BaseRetriever(ABC):
    """
    Абстрактный базовый класс для всех retriever'ов
    
    Все retriever'ы (BM25, Dense, Hybrid) должны наследоваться от этого класса
    и реализовывать методы build_index и search
    """
    
    def __init__(self, name: str = "BaseRetriever"):
        """
        Args:
            name: Название retriever'а для логирования
        """
        self.name = name
        self.is_built = False
        logger.info(f"{self.name} initialized")
    
    @abstractmethod
    def build_index(self, documents: List[str]):
        """
        Построение индекса (offline операция)
        
        Args:
            documents: Список текстов документов
            
        Raises:
            NotImplementedError: Должен быть реализован в подклассе
        """
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[int]:
        """
        Поиск топ-k документов (online операция)
        
        Args:
            query: Текст запроса
            k: Число возвращаемых результатов
            
        Returns:
            Список индексов документов, отсортированных по релевантности
            
        Raises:
            NotImplementedError: Должен быть реализован в подклассе
        """
        pass
    
    def _check_built(self):
        """Проверка что индекс построен"""
        if not self.is_built:
            raise RuntimeError(f"{self.name}: Index not built. Call build_index() first.")
    
    def save_index(self, path: str):
        """
        Сохранение индекса на диск (опционально)
        
        Args:
            path: Путь для сохранения
        """
        logger.warning(f"{self.name}: save_index() not implemented")
    
    def load_index(self, path: str):
        """
        Загрузка индекса с диска (опционально)
        
        Args:
            path: Путь к сохраненному индексу
        """
        logger.warning(f"{self.name}: load_index() not implemented")
    
    def __repr__(self) -> str:
        return f"{self.name}(is_built={self.is_built})"
