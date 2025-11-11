"""
Утилиты для работы с данными
"""

from typing import List, Dict, Any
import pandas as pd
from pathlib import Path
from loguru import logger


def load_csv_safe(path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Безопасная загрузка CSV с обработкой ошибок
    
    Args:
        path: Путь к CSV файлу
        encoding: Кодировка файла
        
    Returns:
        DataFrame
    """
    try:
        df = pd.read_csv(path, encoding=encoding)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed, trying cp1251...")
        df = pd.read_csv(path, encoding='cp1251')
        logger.info(f"Loaded {len(df)} rows from {path} (cp1251)")
        return df
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        raise


def save_submission(
    q_ids: List[int],
    predictions: List[List[int]],
    output_path: str
):
    """
    Сохранение submission файла в правильном формате
    
    Args:
        q_ids: Список ID запросов
        predictions: Список предсказаний для каждого запроса
        output_path: Путь для сохранения
    """
    data = []
    
    for q_id, pred in zip(q_ids, predictions):
        # Форматируем как список в строку
        web_ids_str = str(pred)
        data.append({
            'q_id': q_id,
            'web_id': web_ids_str
        })
    
    df = pd.DataFrame(data)
    
    # Создаем директорию если не существует
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем
    df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Format: q_id, [web_id_1, web_id_2, web_id_3, web_id_4, web_id_5]")


def validate_submission(submission_path: str) -> bool:
    """
    Валидация submission файла
    
    Args:
        submission_path: Путь к submission файлу
        
    Returns:
        True если валиден, False иначе
    """
    try:
        df = pd.read_csv(submission_path)
        
        # Проверка колонок
        required_cols = ['q_id', 'web_id']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing columns. Required: {required_cols}")
            return False
        
        # Проверка что web_id это списки
        sample_web_id = df['web_id'].iloc[0]
        if not (sample_web_id.startswith('[') and sample_web_id.endswith(']')):
            logger.error("web_id must be formatted as lists: [id1, id2, ...]")
            return False
        
        logger.info(f"✅ Submission is valid: {len(df)} predictions")
        return True
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False


def merge_submissions(
    submission_paths: List[str],
    output_path: str,
    weights: List[float] = None
):
    """
    Объединение нескольких submission файлов (ensemble)
    
    Args:
        submission_paths: Пути к submission файлам
        output_path: Путь для сохранения объединенного submission
        weights: Веса для каждого submission (опционально)
    """
    if weights is None:
        weights = [1.0] * len(submission_paths)
    
    if len(weights) != len(submission_paths):
        raise ValueError("Number of weights must match number of submissions")
    
    # TODO: Implement ensemble logic
    logger.warning("Ensemble not implemented yet")
    pass
