"""
Метрики для оценки качества retrieval
"""

from typing import List, Set, Dict
import numpy as np
from loguru import logger


class RetrievalMetrics:
    """
    Метрики для оценки качества retrieval системы
    
    Поддерживаемые метрики:
    - Hit@k: Есть ли релевантный документ в топ-k
    - MRR (Mean Reciprocal Rank): Средний обратный ранг первого релевантного
    - Precision@k: Точность в топ-k
    - Recall@k: Полнота в топ-k
    - NDCG@k: Normalized Discounted Cumulative Gain
    """
    
    @staticmethod
    def hit_at_k(
        predictions: List[List[int]],
        ground_truth: List[Set[int]],
        k: int
    ) -> float:
        """
        Hit@k метрика (основная метрика хакатона)
        
        Hit@k = (1/|Q|) × Σ 𝟙[G_q ∩ R_q^k ≠ ∅]
        
        Args:
            predictions: Предсказания для каждого запроса [[doc_id, ...], ...]
            ground_truth: Ground truth для каждого запроса [{doc_id, ...}, ...]
            k: Число топовых результатов для проверки
            
        Returns:
            Hit@k score ∈ [0, 1]
            
        Examples:
            >>> predictions = [[0, 1, 2], [3, 4, 5]]
            >>> ground_truth = [{0, 10}, {3, 11}]
            >>> hit_at_k(predictions, ground_truth, k=3)
            1.0  # Оба запроса имеют релевантный документ в топ-3
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Length mismatch: predictions={len(predictions)}, "
                f"ground_truth={len(ground_truth)}"
            )
        
        hits = 0
        total = 0
        
        for pred, truth in zip(predictions, ground_truth):
            # Пропускаем запросы без ground truth
            if not truth:
                continue
            
            # Берем топ-k предсказаний
            top_k_pred = set(pred[:k])
            
            # Проверяем пересечение
            if len(top_k_pred & truth) > 0:
                hits += 1
            
            total += 1
        
        if total == 0:
            logger.warning("No valid queries with ground truth")
            return 0.0
        
        score = hits / total
        logger.debug(f"Hit@{k}: {score:.4f} ({hits}/{total})")
        
        return score
    
    @staticmethod
    def mrr(
        predictions: List[List[int]],
        ground_truth: List[Set[int]]
    ) -> float:
        """
        Mean Reciprocal Rank
        
        MRR = (1/|Q|) × Σ (1 / rank_first_relevant)
        
        Args:
            predictions: Предсказания
            ground_truth: Ground truth
            
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for pred, truth in zip(predictions, ground_truth):
            if not truth:
                continue
            
            # Найти ранг первого релевантного документа
            for rank, doc_id in enumerate(pred, start=1):
                if doc_id in truth:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                # Релевантный не найден
                reciprocal_ranks.append(0.0)
        
        if not reciprocal_ranks:
            return 0.0
        
        score = np.mean(reciprocal_ranks)
        logger.debug(f"MRR: {score:.4f}")
        
        return score
    
    @staticmethod
    def precision_at_k(
        predictions: List[List[int]],
        ground_truth: List[Set[int]],
        k: int
    ) -> float:
        """
        Precision@k
        
        Precision@k = (1/|Q|) × Σ (|pred_k ∩ truth| / k)
        
        Args:
            predictions: Предсказания
            ground_truth: Ground truth
            k: Число топовых результатов
            
        Returns:
            Precision@k score
        """
        precisions = []
        
        for pred, truth in zip(predictions, ground_truth):
            if not truth:
                continue
            
            top_k = set(pred[:k])
            precision = len(top_k & truth) / k
            precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        score = np.mean(precisions)
        logger.debug(f"Precision@{k}: {score:.4f}")
        
        return score
    
    @staticmethod
    def recall_at_k(
        predictions: List[List[int]],
        ground_truth: List[Set[int]],
        k: int
    ) -> float:
        """
        Recall@k
        
        Recall@k = (1/|Q|) × Σ (|pred_k ∩ truth| / |truth|)
        
        Args:
            predictions: Предсказания
            ground_truth: Ground truth
            k: Число топовых результатов
            
        Returns:
            Recall@k score
        """
        recalls = []
        
        for pred, truth in zip(predictions, ground_truth):
            if not truth:
                continue
            
            top_k = set(pred[:k])
            recall = len(top_k & truth) / len(truth)
            recalls.append(recall)
        
        if not recalls:
            return 0.0
        
        score = np.mean(recalls)
        logger.debug(f"Recall@{k}: {score:.4f}")
        
        return score
    
    @staticmethod
    def evaluate_all(
        predictions: List[List[int]],
        ground_truth: List[Set[int]],
        k_values: List[int] = [5, 10]
    ) -> Dict[str, float]:
        """
        Вычисление всех метрик
        
        Args:
            predictions: Предсказания
            ground_truth: Ground truth
            k_values: Список значений k для метрик
            
        Returns:
            Словарь с метриками
            
        Examples:
            >>> results = RetrievalMetrics.evaluate_all(preds, gt, k_values=[5, 10])
            >>> print(results)
            {'hit@5': 0.83, 'hit@10': 0.91, 'mrr': 0.75, ...}
        """
        metrics = RetrievalMetrics()
        results = {}
        
        # Hit@k для каждого k
        for k in k_values:
            results[f'hit@{k}'] = metrics.hit_at_k(predictions, ground_truth, k)
            results[f'precision@{k}'] = metrics.precision_at_k(predictions, ground_truth, k)
            results[f'recall@{k}'] = metrics.recall_at_k(predictions, ground_truth, k)
        
        # MRR
        results['mrr'] = metrics.mrr(predictions, ground_truth)
        
        logger.info(f"Evaluation results: {results}")
        
        return results


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Форматирование метрик для вывода
    
    Args:
        metrics: Словарь метрик
        
    Returns:
        Отформатированная строка
    """
    lines = ["=" * 50]
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 50)
    
    for metric_name, value in sorted(metrics.items()):
        lines.append(f"{metric_name:.<30} {value:.4f}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)
