"""
Reciprocal Rank Fusion (RRF) - метод из Yaoshi-RAG
"""

from typing import List, Dict, Tuple
from collections import defaultdict
from loguru import logger


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) для объединения результатов нескольких retriever'ов
    
    Алгоритм из Yaoshi-RAG (показал лучшие результаты чем weighted sum):
        RRF(d) = Σ_{r∈rankings} 1 / (k + rank_r(d))
    
    Где:
    - k = 60 (константа из литературы)
    - rank_r(d) - позиция документа d в ранжировании r
    
    Преимущества:
    - Не требует калибровки весов
    - Робастен к различиям в scale scores
    - Работает лучше на практике
    
    Args:
        k: Константа для сглаживания (default: 60)
    """
    
    def __init__(self, k: int = 60):
        self.k = k
        logger.info(f"ReciprocalRankFusion initialized with k={k}")
    
    def fuse(self, rankings: List[List[int]]) -> List[int]:
        """
        Объединение нескольких ранжирований
        
        Args:
            rankings: Список ранжирований, каждое - список doc_id
                     [[doc1, doc2, ...], [doc5, doc1, ...], ...]
        
        Returns:
            Объединенное ранжирование (список doc_id по убыванию RRF score)
            
        Examples:
            >>> fusion = ReciprocalRankFusion(k=60)
            >>> bm25_ranking = [0, 1, 2, 3, 4]
            >>> dense_ranking = [2, 0, 5, 1, 6]
            >>> fused = fusion.fuse([bm25_ranking, dense_ranking])
            >>> print(fused[:5])  # топ-5 после fusion
        """
        if not rankings:
            logger.warning("Empty rankings provided to RRF")
            return []
        
        # Подсчет RRF scores для каждого документа
        scores = defaultdict(float)
        
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                # RRF формула: 1 / (k + rank)
                scores[doc_id] += 1.0 / (self.k + rank)
        
        # Сортировка по убыванию RRF score
        fused_ranking = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        result = [doc_id for doc_id, _ in fused_ranking]
        
        logger.debug(f"Fused {len(rankings)} rankings -> {len(result)} unique documents")
        
        return result
    
    def fuse_with_scores(
        self,
        rankings: List[List[Tuple[int, float]]]
    ) -> List[Tuple[int, float]]:
        """
        Объединение ранжирований с сохранением scores
        
        Args:
            rankings: Список ранжирований с scores
                     [[(doc_id, score), ...], ...]
        
        Returns:
            List[(doc_id, rrf_score)]
        """
        if not rankings:
            return []
        
        # Извлекаем только doc_id для RRF
        doc_rankings = [[doc_id for doc_id, _ in ranking] for ranking in rankings]
        
        # Применяем RRF
        scores = defaultdict(float)
        for ranking in doc_rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                scores[doc_id] += 1.0 / (self.k + rank)
        
        # Сортировка с scores
        fused = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return fused


class WeightedFusion:
    """
    Weighted linear combination (альтернатива RRF)
    
    Score(d) = α * score_1(d) + (1-α) * score_2(d)
    
    Требует нормализации scores!
    
    Args:
        alpha: Вес первого retriever'а (0 < alpha < 1)
    """
    
    def __init__(self, alpha: float = 0.3):
        if not 0 < alpha < 1:
            raise ValueError(f"Alpha must be in (0, 1), got {alpha}")
        
        self.alpha = alpha
        logger.info(f"WeightedFusion initialized with alpha={alpha}")
    
    def fuse(
        self,
        ranking1: List[Tuple[int, float]],
        ranking2: List[Tuple[int, float]]
    ) -> List[int]:
        """
        Объединение двух ранжирований с весами
        
        Args:
            ranking1: Первое ранжирование [(doc_id, score), ...]
            ranking2: Второе ранжирование [(doc_id, score), ...]
        
        Returns:
            Объединенное ранжирование
        """
        # Извлекаем scores
        scores1 = {doc_id: score for doc_id, score in ranking1}
        scores2 = {doc_id: score for doc_id, score in ranking2}
        
        # Нормализация (min-max)
        scores1_norm = self._normalize_scores(scores1)
        scores2_norm = self._normalize_scores(scores2)
        
        # Weighted combination
        all_docs = set(scores1_norm.keys()) | set(scores2_norm.keys())
        
        final_scores = {}
        for doc_id in all_docs:
            s1 = scores1_norm.get(doc_id, 0.0)
            s2 = scores2_norm.get(doc_id, 0.0)
            final_scores[doc_id] = self.alpha * s1 + (1 - self.alpha) * s2
        
        # Сортировка
        fused = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc_id for doc_id, _ in fused]
    
    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Min-max нормализация scores"""
        if not scores:
            return {}
        
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {k: 1.0 for k in scores.keys()}
        
        normalized = {
            doc_id: (score - min_val) / (max_val - min_val)
            for doc_id, score in scores.items()
        }
        
        return normalized


class DistributionBasedFusion:
    """
    Distribution-based Fusion (DBFusion)
    
    Использует CDF (cumulative distribution function) для нормализации
    Более робастный чем min-max
    """
    
    def __init__(self):
        logger.info("DistributionBasedFusion initialized")
    
    def fuse(
        self,
        ranking1: List[Tuple[int, float]],
        ranking2: List[Tuple[int, float]]
    ) -> List[int]:
        """
        Объединение через rank-based normalization
        
        Args:
            ranking1: Первое ранжирование
            ranking2: Второе ранжирование
        
        Returns:
            Объединенное ранжирование
        """
        # Преобразуем в rank-based scores
        rank_scores1 = self._ranking_to_scores(ranking1)
        rank_scores2 = self._ranking_to_scores(ranking2)
        
        # Объединяем через geometric mean
        all_docs = set(rank_scores1.keys()) | set(rank_scores2.keys())
        
        final_scores = {}
        for doc_id in all_docs:
            s1 = rank_scores1.get(doc_id, 0.0)
            s2 = rank_scores2.get(doc_id, 0.0)
            
            # Geometric mean
            if s1 > 0 and s2 > 0:
                final_scores[doc_id] = (s1 * s2) ** 0.5
            else:
                final_scores[doc_id] = max(s1, s2)
        
        # Сортировка
        fused = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc_id for doc_id, _ in fused]
    
    def _ranking_to_scores(
        self,
        ranking: List[Tuple[int, float]]
    ) -> Dict[int, float]:
        """
        Преобразование ранжирования в rank-based scores
        
        Score = (N - rank + 1) / N
        """
        n = len(ranking)
        scores = {}
        
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] = (n - rank + 1) / n
        
        return scores
