"""
Скрипт для запуска retrieval и создания submission файла
"""

import sys
from pathlib import Path
import pandas as pd

# Добавляем корневую директорию в PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.preprocessing.document_processor import DocumentProcessor, QueryProcessor
from src.retrieval.pipeline import RetrievalPipeline
from loguru import logger


def load_questions(path: str) -> pd.DataFrame:
    """Загрузка вопросов из CSV"""
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} questions from {path}")
    return df


def save_submission(predictions: list, web_ids: list, output_path: str):
    """
    Сохранение submission файла
    
    Args:
        predictions: Список предсказаний [[q_id, [web_id, ...]], ...]
        web_ids: Список всех web_id документов
        output_path: Путь для сохранения
    """
    # Преобразуем индексы в web_id
    submission_data = []
    
    for q_id, doc_indices in predictions:
        # Конвертируем индексы в web_id
        predicted_web_ids = [web_ids[idx] for idx in doc_indices]
        
        submission_data.append({
            'q_id': q_id,
            'web_id': predicted_web_ids
        })
    
    # Создаем DataFrame
    df = pd.DataFrame(submission_data)
    
    # Сохраняем
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Format: q_id, [web_id_1, web_id_2, web_id_3, web_id_4, web_id_5]")


def main():
    """Главная функция для retrieval"""
    
    # Настройка логирования
    setup_logger(log_level="INFO", log_file="logs/run_retrieval.log")
    
    logger.info("=" * 60)
    logger.info("RUNNING RETRIEVAL")
    logger.info("=" * 60)
    
    # Загрузка конфигурации
    config = Config("config/base.yaml")
    
    # Загрузка документов
    logger.info("Loading documents...")
    doc_processor = DocumentProcessor(config['preprocessing'])
    documents = doc_processor.load_documents(config['data']['websites_path'])
    
    # Извлечение web_ids для submission
    web_ids = [doc.web_id for doc in documents]
    
    # Извлечение текстов
    texts = doc_processor.get_texts_for_indexing(documents, use_enhanced=True)
    
    # Инициализация pipeline
    logger.info("Initializing retrieval pipeline...")
    pipeline = RetrievalPipeline(config.config)
    
    # Проверка сохраненных индексов
    indices_dir = config['output']['indices_dir']
    if Path(indices_dir).exists():
        logger.info(f"Loading pre-built indices from {indices_dir}...")
        try:
            pipeline.load_indices(indices_dir)
        except Exception as e:
            logger.warning(f"Could not load indices: {e}")
            logger.info("Building new indices...")
            pipeline.build_indices(texts)
    else:
        logger.info("Building indices...")
        pipeline.build_indices(texts)
    
    # Загрузка вопросов
    logger.info("Loading questions...")
    questions_df = load_questions(config['data']['questions_path'])
    
    # Обработка запросов
    query_processor = QueryProcessor(config['preprocessing'])
    processed_queries = query_processor.process_batch(questions_df['query'].tolist())
    
    # Retrieval
    logger.info(f"Running retrieval for {len(processed_queries)} queries...")
    predictions = []
    
    from tqdm import tqdm
    for idx, (q_id, query) in enumerate(tqdm(
        zip(questions_df['q_id'], processed_queries),
        total=len(processed_queries),
        desc="Retrieving"
    )):
        # Поиск топ-5
        top_k_indices = pipeline.search(query, k=5)
        predictions.append((q_id, top_k_indices))
    
    # Сохранение submission
    output_path = Path(config['data']['output_path']) / "submission.csv"
    save_submission(predictions, web_ids, str(output_path))
    
    logger.info("=" * 60)
    logger.info("RETRIEVAL COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    # Вывод примеров
    logger.info("\nSample predictions:")
    for i in range(min(3, len(predictions))):
        q_id, indices = predictions[i]
        query = processed_queries[i]
        logger.info(f"\nQuery {q_id}: '{query}'")
        logger.info(f"Top-5 web_ids: {[web_ids[idx] for idx in indices]}")
        logger.info("Documents:")
        for rank, idx in enumerate(indices, 1):
            doc = documents[idx]
            logger.info(f"  {rank}. {doc.title}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error during retrieval: {e}")
        sys.exit(1)
