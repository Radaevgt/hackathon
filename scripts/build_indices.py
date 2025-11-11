"""
Скрипт для построения индексов (offline операция)
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.preprocessing.document_processor import DocumentProcessor
from src.retrieval.pipeline import RetrievalPipeline
from loguru import logger


def main():
    """Главная функция для построения индексов"""
    
    # Настройка логирования
    setup_logger(log_level="INFO", log_file="logs/build_indices.log")
    
    logger.info("=" * 60)
    logger.info("BUILDING INDICES")
    logger.info("=" * 60)
    
    # Загрузка конфигурации
    config = Config("config/base.yaml")
    logger.info(f"Loaded config: {config.config_path}")
    
    # Загрузка и обработка документов
    logger.info("Loading documents...")
    doc_processor = DocumentProcessor(config['preprocessing'])
    documents = doc_processor.load_documents(config['data']['websites_path'])
    
    # Извлечение текстов для индексации
    texts = doc_processor.get_texts_for_indexing(
        documents,
        use_enhanced=config.get('preprocessing.use_metadata_enhancement', True)
    )
    
    logger.info(f"Prepared {len(texts)} texts for indexing")
    
    # Построение индексов
    logger.info("Building retrieval pipeline...")
    pipeline = RetrievalPipeline(config.config)
    
    logger.info("Building indices (this may take several minutes)...")
    pipeline.build_indices(texts)
    
    # Сохранение индексов
    indices_dir = config['output']['indices_dir']
    logger.info(f"Saving indices to {indices_dir}...")
    pipeline.save_indices(indices_dir)
    
    logger.info("=" * 60)
    logger.info("INDICES BUILT SUCCESSFULLY!")
    logger.info("=" * 60)
    
    # Тестовый поиск
    logger.info("\nTesting retrieval with sample query...")
    test_query = "как открыть счёт"
    results = pipeline.search(test_query, k=5)
    
    logger.info(f"Query: '{test_query}'")
    logger.info(f"Top-5 results: {results}")
    logger.info(f"Top-5 documents:")
    for i, idx in enumerate(results, 1):
        doc = documents[idx]
        logger.info(f"{i}. web_id={doc.web_id}, title='{doc.title}'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error building indices: {e}")
        sys.exit(1)
