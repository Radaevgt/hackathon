"""
Alpha RAG System - Интеллектуальная система поиска документов
Команда: Neuro Bureau
Хакатон: Альфа-Будущее 2025
"""

__version__ = "1.0.0"
__author__ = "Neuro Bureau"

from src.preprocessing.document_processor import DocumentProcessor
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.pipeline import RetrievalPipeline
from src.fusion.rrf import ReciprocalRankFusion
from src.evaluation.metrics import RetrievalMetrics

__all__ = [
    "DocumentProcessor",
    "BM25Retriever",
    "DenseRetriever",
    "RetrievalPipeline",
    "ReciprocalRankFusion",
    "RetrievalMetrics",
]
