# 🏦 Alpha RAG System - Intelligent Document Retrieval

Интеллектуальная система поиска релевантных документов для Альфа-Банка на основе RAG (Retrieval-Augmented Generation).

**Команда:** Neuro Bureau  
**Хакатон:** Альфа-Будущее 2025  
**Цель:** Hit@5 ≥ 0.83

---

## 📋 Описание проекта

RAG-система для поиска релевантных банковских документов по пользовательским запросам. Система использует гибридный подход, объединяющий:

- **BM25** (sparse retrieval) - быстрый лексический поиск
- **E5 Embeddings** (dense retrieval) - семантический поиск
- **Reciprocal Rank Fusion** - интеллектуальное объединение результатов

### Ключевые особенности

✅ Hybrid retrieval (BM25 + Dense)  
✅ Metadata enhancement (из Yaoshi-RAG подхода)  
✅ Reciprocal Rank Fusion  
✅ Русскоязычная обработка текста  
✅ Модульная архитектура  
✅ Быстрый inference (<500ms)  

---

## 🚀 Быстрый старт

### Требования

- Python 3.10+
- 8GB RAM (минимум)
- ~5GB свободного места на диске

### Установка

1. **Клонирование репозитория**
```powershell
git clone <repository-url>
cd alpha-rag-system
```

2. **Создание виртуального окружения**
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Если возникает ошибка ExecutionPolicy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

3. **Установка зависимостей**
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Подготовка данных**

Поместите файлы данных в директорию `data/raw/`:
- `websites.csv` - корпус документов
- `questions.csv` - запросы для поиска

---

## 📖 Использование

### Шаг 1: Построение индексов (offline)
```powershell
python scripts/build_indices.py
```

Этот скрипт:
- Загружает и обрабатывает документы
- Строит BM25 и Dense индексы
- Сохраняет индексы в `data/indices/`

**Время выполнения:** ~5-10 минут (зависит от размера корпуса)

### Шаг 2: Запуск retrieval (online)
```powershell
python scripts/run_retrieval.py
```

Этот скрипт:
- Загружает предобработанные индексы
- Обрабатывает запросы из `questions.csv`
- Выполняет поиск топ-5 документов для каждого запроса
- Сохраняет результаты в `data/submissions/submission.csv`

**Время выполнения:** ~2-5 минут (для 1000 запросов)

### Шаг 3: Проверка результатов
```powershell
# Просмотр submission файла
Get-Content data/submissions/submission.csv | Select-Object -First 10
```

Формат submission:
```
q_id,[web_id_1,web_id_2,web_id_3,web_id_4,web_id_5]
1,[42,17,93,5,28]
2,[156,89,234,12,67]
...
```

---

## 🏗️ Архитектура
```
┌─────────────────────────────────────────────────────────┐
│                   RETRIEVAL PIPELINE                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User Query                                             │
│      │                                                  │
│      ├──────────┬────────────┐                          │
│      │          │            │                          │
│  ┌───▼───┐  ┌───▼────┐  ┌───▼────┐                      │
│  │ BM25  │  │ Dense  │  │ Other  │                      │
│  │(100)  │  │ E5(100)│  │        │                      │
│  └───┬───┘  └───┬────┘  └───┬────┘                      │
│      │          │            │                          │
│      └──────────┴────────────┘                          │
│                 │                                       │
│          ┌──────▼──────┐                                │
│          │ RRF Fusion  │                                │
│          │   (k=60)    │                                │
│          └──────┬──────┘                                │
│                 │                                       │
│          ┌──────▼──────┐                                │
│          │   Top-5     │                                │
│          │  Results    │                                │
│          └─────────────┘                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Компоненты системы

#### 1. **Preprocessing** (`src/preprocessing/`)
- `TextCleaner` - очистка и нормализация текста
- `DocumentProcessor` - загрузка и обработка документов
- `MetadataEnhancer` - обогащение метаданными (title, URL section)

#### 2. **Retrieval** (`src/retrieval/`)
- `BM25Retriever` - традиционный sparse retrieval
- `DenseRetriever` - neural embeddings (E5)
- `RetrievalPipeline` - объединение всех компонентов

#### 3. **Fusion** (`src/fusion/`)
- `ReciprocalRankFusion` - RRF алгоритм (основной)
- `WeightedFusion` - взвешенная комбинация (альтернатива)

#### 4. **Evaluation** (`src/evaluation/`)
- `RetrievalMetrics` - метрики качества (Hit@k, MRR, Precision, Recall)

---

## ⚙️ Конфигурация

Основная конфигурация находится в `config/base.yaml`:
```yaml
retrieval:
  # BM25 параметры
  use_bm25: true
  bm25:
    k1: 1.5          # saturation parameter
    b: 0.75          # length normalization
    top_k: 100
  
  # Dense retrieval параметры
  use_dense: true
  dense:
    model_name: "intfloat/multilingual-e5-large"
    use_gpu: false
    top_k: 100
  
  # Fusion параметры
  fusion_type: "rrf"  # rrf | weighted
  rrf_k: 60
```

### Настройка параметров

Для экспериментов создайте новый конфиг в `config/experiments/`:
```powershell
Copy-Item config/base.yaml config/experiments/exp001.yaml
# Отредактируйте exp001.yaml
```

---

## 📊 Метрики и оценка

### Основная метрика: Hit@5

**Hit@5** = доля запросов, для которых хотя бы один релевантный документ попал в топ-5 результатов.
```
Hit@5 = (Число запросов с релевантным в топ-5) / (Общее число запросов)
```

**Целевое значение:** ≥ 0.83

### Дополнительные метрики

- **MRR** (Mean Reciprocal Rank) - средний обратный ранг первого релевантного
- **Precision@5** - точность в топ-5
- **Recall@5** - полнота в топ-5

### Запуск evaluation (если есть ground truth)
```python
from src.evaluation.metrics import RetrievalMetrics

# Загрузка предсказаний и ground truth
predictions = [...]  # [[doc_id, ...], ...]
ground_truth = [...]  # [{doc_id, ...}, ...]

# Оценка
metrics = RetrievalMetrics.evaluate_all(predictions, ground_truth, k_values=[5, 10])
print(metrics)
# {'hit@5': 0.83, 'mrr': 0.75, ...}
```

---

## 🧪 Тестирование

### Запуск unit tests
```powershell
# Все тесты
pytest tests/ -v

# Конкретный модуль
pytest tests/test_retrieval.py -v

# С покрытием кода
pytest tests/ --cov=src --cov-report=html
```

### Создание новых тестов

Тесты находятся в `tests/`:
- `test_preprocessing.py` - тесты preprocessing
- `test_retrieval.py` - тесты retrieval
- `test_fusion.py` - тесты fusion
- `test_integration.py` - интеграционные тесты

---

## 🔧 Troubleshooting

### Проблема: "Module not found"
```powershell
# Убедитесь что установлены все зависимости
pip install -r requirements.txt

# Проверьте PYTHONPATH
$env:PYTHONPATH = "."
```

### Проблема: "Out of memory"
```yaml
# В config/base.yaml уменьшите batch_size
dense:
  batch_size: 16  # было 32
```

### Проблема: Медленный inference
```yaml
# Используйте меньшую модель
dense:
  model_name: "intfloat/multilingual-e5-base"  # вместо large

# Или уменьшите top_k
bm25:
  top_k: 50  # было 100
dense:
  top_k: 50
```

### Проблема: FAISS ошибки на Windows
```powershell
# Переустановите faiss-cpu
pip uninstall faiss-cpu
pip install faiss-cpu==1.7.4
```

---

## 📁 Структура проекта
```
alpha-rag-system/
│
├── config/                      # Конфигурации
│   ├── base.yaml               # Основная конфигурация
│   └── experiments/            # Экспериментальные конфиги
│
├── data/
│   ├── raw/                    # Исходные данные
│   │   ├── websites.csv
│   │   └── questions.csv
│   ├── processed/              # Обработанные данные
│   ├── indices/                # FAISS индексы
│   └── submissions/            # Файлы сабмитов
│
├── src/                         # Исходный код
│   ├── preprocessing/          # Обработка текста
│   │   ├── text_cleaner.py
│   │   └── document_processor.py
│   ├── retrieval/              # Retrieval компоненты
│   │   ├── base.py
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   └── pipeline.py
│   ├── fusion/                 # Fusion стратегии
│   │   └── rrf.py
│   ├── evaluation/             # Метрики
│   │   └── metrics.py
│   └── utils/                  # Утилиты
│       ├── config.py
│       └── logger.py
│
├── scripts/                     # Скрипты запуска
│   ├── build_indices.py        # Построение индексов
│   └── run_retrieval.py        # Запуск retrieval
│
├── tests/                       # Тесты
│   ├── test_preprocessing.py
│   ├── test_retrieval.py
│   └── test_integration.py
│
├── logs/                        # Логи (создается автоматически)
├── notebooks/                   # Jupyter notebooks для анализа
├── requirements.txt             # Зависимости
├── setup.py                     # Setup script
├── .gitignore
└── README.md
```

---

## 🎯 Roadmap

### День 1: MVP ✅
- [x] Базовая структура проекта
- [x] Preprocessing pipeline
- [x] BM25 retriever
- [x] Dense retriever (E5)
- [x] RRF fusion
- [x] Baseline submission

### День 2: Optimization 🔄
- [ ] Hyperparameter tuning
- [ ] Query expansion
- [ ] Metadata enhancement optimization
- [ ] Performance profiling

### День 3: Advanced 🚀
- [ ] Cross-encoder reranking
- [ ] Ensemble methods
- [ ] Error analysis
- [ ] Final optimization

---

## 📚 Референсы

### Научные статьи
- **Yaoshi-RAG** (2025) - Hybrid retrieval + KG reasoning
- **ColBERT** (Khattab & Zaharia, 2020) - Late interaction
- **SPLADE** (Formal et al., 2021) - Learned sparse retrieval
- **E5** (Wang et al., 2022) - Text embeddings by weakly-supervised contrastive pre-training

### Библиотеки
- [Sentence-Transformers](https://www.sbert.net/) - Semantic search
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [rank_bm25](https://github.com/dorianbrown/rank_bm25) - BM25 implementation

---

## 👥 Команда

**Neuro Bureau**
- Разработка RAG-системы
- Оптимизация retrieval pipeline
- Адаптация Yaoshi-RAG подхода

---

## 📄 Лицензия

MIT License

---

## 🙏 Благодарности

- Альфа-Банк за организацию хакатона
- Yaoshi-RAG авторы за инсайты в hybrid retrieval
- HuggingFace за pre-trained модели

---

**Версия:** 1.0.0  
**Дата:** Ноябрь 2025  
**Статус:** В разработке 🚧
