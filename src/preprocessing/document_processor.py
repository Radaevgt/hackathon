"""
Модуль для обработки документов
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
import re
from loguru import logger
from src.preprocessing.text_cleaner import TextCleaner, RussianTextNormalizer


@dataclass
class Document:
    """
    Структура документа
    
    Attributes:
        web_id: Уникальный идентификатор документа
        text: Основной текст документа
        title: Заголовок
        url: URL страницы
        kind: Тип контента
        enhanced_text: Обогащенный текст (с метаданными)
    """
    web_id: int
    text: str
    title: str
    url: str
    kind: str
    enhanced_text: Optional[str] = None
    
    def __repr__(self) -> str:
        return f"Document(web_id={self.web_id}, title='{self.title[:50]}...', text_len={len(self.text)})"


class MetadataEnhancer:
    """
    Обогащение документов метаданными (из Yaoshi-RAG подхода)
    
    Добавляет структурированную информацию для улучшения поиска:
    - Title (наиболее важная часть)
    - Section (извлекается из URL)
    - Content type (kind)
    """
    
    # Маппинг URL паттернов на разделы сайта
    URL_SECTION_MAPPING = {
        r'credit|kredit|loan': 'Кредиты',
        r'card|karta|debet': 'Карты',
        r'deposit|vklad|wklad': 'Вклады',
        r'mortgage|ipoteka': 'Ипотека',
        r'insurance|strakhovan': 'Страхование',
        r'invest|investicii': 'Инвестиции',
        r'transfer|perevod': 'Переводы',
        r'account|schet|счёт': 'Счета',
        r'business|biznes': 'Бизнес',
        r'mobile|app|prilozhenie': 'Мобильный банк',
    }
    
    def enhance(self, row: pd.Series) -> str:
        """
        Обогащение документа метаданными
        
        Args:
            row: Строка DataFrame с полями документа
            
        Returns:
            Обогащенный текст документа
            
        Examples:
            >>> enhancer = MetadataEnhancer()
            >>> row = pd.Series({'title': 'Кредит', 'url': '...', 'kind': 'product', 'text': '...'})
            >>> enhanced = enhancer.enhance(row)
        """
        components = []
        
        # 1. Title (самая важная часть - query-document overlap)
        if pd.notna(row.get('title')) and row['title'].strip():
            components.append(f"Заголовок: {row['title']}")
        
        # 2. Section из URL
        section = self._extract_section_from_url(row.get('url', ''))
        if section:
            components.append(f"Раздел: {section}")
        
        # 3. Тип контента
        if pd.notna(row.get('kind')) and row['kind'].strip():
            kind_readable = self._format_kind(row['kind'])
            components.append(f"Тип: {kind_readable}")
        
        # 4. Основной текст
        if pd.notna(row.get('text')) and row['text'].strip():
            components.append(row['text'])
        
        return "\n".join(components)
    
    def _extract_section_from_url(self, url: str) -> Optional[str]:
        """
        Извлечение раздела сайта из URL
        
        Args:
            url: URL страницы
            
        Returns:
            Название раздела или None
        """
        if not url:
            return None
        
        url_lower = url.lower()
        
        for pattern, section in self.URL_SECTION_MAPPING.items():
            if re.search(pattern, url_lower):
                return section
        
        return None
    
    def _format_kind(self, kind: str) -> str:
        """Форматирование типа контента для читаемости"""
        kind_mapping = {
            'product': 'Продукт',
            'faq': 'Вопрос-ответ',
            'article': 'Статья',
            'terms': 'Условия',
            'guide': 'Руководство',
        }
        return kind_mapping.get(kind.lower(), kind)


class DocumentProcessor:
    """
    Центральный класс для обработки документов
    
    Responsibilities:
    - Загрузка и валидация данных
    - Cleaning & normalization
    - Metadata enhancement
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Словарь конфигурации с параметрами обработки
        """
        self.config = config
        self.cleaner = TextCleaner()
        self.enhancer = MetadataEnhancer()
        self.normalizer = RussianTextNormalizer()
        
        logger.info("DocumentProcessor initialized")
    
    def load_documents(self, path: str) -> List[Document]:
        """
        Загрузка документов из CSV файла
        
        Args:
            path: Путь к CSV файлу с документами
            
        Returns:
            Список объектов Document
            
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если отсутствуют обязательные колонки
        """
        logger.info(f"Loading documents from {path}")
        
        # Загрузка CSV
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
        
        # Валидация колонок
        required_columns = ['web_id', 'text', 'title', 'url', 'kind']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Loaded {len(df)} documents")
        
        # Обработка документов
        documents = []
        for idx, row in df.iterrows():
            try:
                doc = self._process_document(row)
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Error processing document {row.get('web_id', idx)}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(documents)} documents")
        
        # Статистика
        self._log_statistics(documents)
        
        return documents
    
    def _process_document(self, row: pd.Series) -> Document:
        """
        Обработка одного документа
        
        Args:
            row: Строка DataFrame
            
        Returns:
            Объект Document
        """
        # Cleaning основного текста
        clean_text = self.cleaner.clean(row['text']) if pd.notna(row['text']) else ""
        
        # Metadata enhancement (если включено в конфиге)
        if self.config.get('use_metadata_enhancement', True):
            enhanced_text = self.enhancer.enhance(row)
        else:
            enhanced_text = clean_text
        
        # Создание документа
        doc = Document(
            web_id=int(row['web_id']),
            text=clean_text,
            title=str(row['title']) if pd.notna(row['title']) else "",
            url=str(row['url']) if pd.notna(row['url']) else "",
            kind=str(row['kind']) if pd.notna(row['kind']) else "",
            enhanced_text=enhanced_text
        )
        
        return doc
    
    def get_texts_for_indexing(self, documents: List[Document], use_enhanced: bool = True) -> List[str]:
        """
        Извлечение текстов для индексации
        
        Args:
            documents: Список документов
            use_enhanced: Использовать enhanced_text или обычный text
            
        Returns:
            Список текстов для индексации
        """
        if use_enhanced:
            texts = [doc.enhanced_text for doc in documents]
        else:
            texts = [doc.text for doc in documents]
        
        logger.info(f"Extracted {len(texts)} texts for indexing (use_enhanced={use_enhanced})")
        return texts
    
    def _log_statistics(self, documents: List[Document]):
        """Вывод статистики по документам"""
        if not documents:
            return
        
        text_lengths = [len(doc.text.split()) for doc in documents]
        
        logger.info(f"Document statistics:")
        logger.info(f"  Total: {len(documents)}")
        logger.info(f"  Avg text length: {sum(text_lengths) / len(text_lengths):.0f} words")
        logger.info(f"  Min/Max length: {min(text_lengths)}/{max(text_lengths)} words")
        
        # Статистика по типам
        kinds = {}
        for doc in documents:
            kinds[doc.kind] = kinds.get(doc.kind, 0) + 1
        
        logger.info(f"  Document types: {kinds}")


class QueryProcessor:
    """
    Обработка пользовательских запросов
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Словарь конфигурации
        """
        self.config = config
        self.cleaner = TextCleaner()
        logger.info("QueryProcessor initialized")
    
    def process(self, query: str) -> str:
        """
        Базовая обработка запроса
        
        Args:
            query: Исходный запрос пользователя
            
        Returns:
            Обработанный запрос
        """
        # Базовая очистка
        cleaned = self.cleaner.clean(query)
        
        return cleaned
    
    def process_batch(self, queries: List[str]) -> List[str]:
        """
        Пакетная обработка запросов
        
        Args:
            queries: Список запросов
            
        Returns:
            Список обработанных запросов
        """
        return [self.process(q) for q in queries]
