"""
Ручная проверка качества retrieval
"""

import sys
from pathlib import Path
import pandas as pd
import random

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.document_processor import DocumentProcessor
from src.retrieval.pipeline import RetrievalPipeline
from src.utils.config import Config


def manual_check(n_samples: int = 20):
    """
    Ручная проверка качества на случайных запросах
    """
    print("=" * 80)
    print("MANUAL QUALITY CHECK")
    print("=" * 80)
    
    # Загрузка
    config = Config("config/base.yaml")
    
    doc_processor = DocumentProcessor(config['preprocessing'])
    documents = doc_processor.load_documents(config['data']['websites_path'])
    texts = doc_processor.get_texts_for_indexing(documents, use_enhanced=True)
    web_ids = [doc.web_id for doc in documents]
    
    # Pipeline
    pipeline = RetrievalPipeline(config.config)
    pipeline.load_indices("data/indices")
    
    # Загрузка запросов
    questions_df = pd.read_csv(config['data']['questions_path'])
    
    # Случайная выборка
    samples = questions_df.sample(n=n_samples, random_state=42)
    
    print(f"\nChecking {n_samples} random queries...\n")
    
    good = 0
    bad = 0
    
    for idx, row in samples.iterrows():
        query = row['query']
        
        # Retrieval
        top_k_indices = pipeline.search(query, k=5)
        
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"{'='*80}")
        
        print("\nTop-5 results:")
        for rank, doc_idx in enumerate(top_k_indices, 1):
            doc = documents[doc_idx]
            print(f"\n{rank}. [web_id={doc.web_id}]")
            print(f"   Title: {doc.title}")
            print(f"   Text preview: {doc.text[:150]}...")
        
        # Ручная оценка
        print("\n" + "-" * 80)
        rating = input("Quality? (g=good, b=bad, s=skip): ").strip().lower()
        
        if rating == 'g':
            good += 1
        elif rating == 'b':
            bad += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTS: Good={good}, Bad={bad}")
    if good + bad > 0:
        print(f"Estimated quality: {good/(good+bad)*100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    manual_check(n_samples=10)  # Проверь 10 запросов вручную"""
