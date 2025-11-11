"""
Скрипт для оценки качества на validation set
"""

import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.preprocessing.document_processor import DocumentProcessor
from src.retrieval.pipeline import RetrievalPipeline
from src.evaluation.metrics import RetrievalMetrics, format_metrics
from loguru import logger


def load_ground_truth(path: str) -> dict:
    """
    Загрузка ground truth
    
    Формат CSV: q_id, relevant_web_ids
    где relevant_web_ids это список через запятую
    """
    df = pd.read_csv(path)
    
    ground_truth = {}
    for _, row in df.iterrows():
        q_id = row['q_id']
        relevant_ids = set(map(int, str(row['relevant_web_ids']).split(',')))
        ground_truth[q_id] = relevant_ids
    
    return ground_truth


def main():
    """Оценка качества на validation set"""
    
    setup_logger(log_level="INFO", log_file="logs/evaluate.log")
    
    logger.info("=" * 60)
    logger.info("EVALUATION ON VALIDATION SET")
    logger.info("=" * 60)
    
    # Загрузка конфигурации
    config = Config("config/base.yaml")
    
    # Загрузка документов
    doc_processor = DocumentProcessor(config['preprocessing'])
    documents = doc_processor.load_documents(config['data']['websites_path'])
    texts = doc_processor.get_texts_for_indexing(documents, use_enhanced=True)
    web_ids = [doc.web_id for doc in documents]
    
    # Создание mapping web_id -> index
    web_id_to_idx = {web_id: idx for idx, web_id in enumerate(web_ids)}
    
    # Pipeline
    pipeline = RetrievalPipeline(config.config)
    
    # Загрузка индексов
    indices_dir = config['output']['indices_dir']
    if Path(indices_dir).exists():
        pipeline.load_indices(indices_dir)
    else:
        logger.info("Building indices...")
        pipeline.build_indices(texts)
    
    # Загрузка validation queries
    val_questions = pd.read_csv("data/raw/val_questions.csv")
    
    # Загрузка ground truth
    ground_truth_dict = load_ground_truth("data/raw/val_ground_truth.csv")
    
    # Retrieval
    logger.info(f"Running retrieval for {len(val_questions)} validation queries...")
    
    predictions = []
    ground_truth = []
    
    for _, row in val_questions.iterrows():
        q_id = row['q_id']
        query = row['query']
        
        # Поиск
        top_k_indices = pipeline.search(query, k=10)
        
        # Конвертация индексов в web_ids
        predicted_web_ids = [web_ids[idx] for idx in top_k_indices]
        predictions.append(predicted_web_ids)
        
        # Ground truth для этого запроса
        gt = ground_truth_dict.get(q_id, set())
        ground_truth.append(gt)
    
    # Оценка
    logger.info("\nCalculating metrics...")
    metrics = RetrievalMetrics.evaluate_all(
        predictions,
        ground_truth,
        k_values=[5, 10]
    )
    
    # Вывод результатов
    print(format_metrics(metrics))
    
    # Сохранение результатов
    results_df = pd.DataFrame([metrics])
    results_df.to_csv("data/evaluation_results.csv", index=False)
    logger.info("Results saved to data/evaluation_results.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error during evaluation: {e}")
        sys.exit(1)
