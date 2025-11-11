"""
Модуль для очистки и нормализации текста
"""

import re
import unicodedata
import html
from typing import Optional
from loguru import logger


class TextCleaner:
    """
    Класс для очистки текста документов и запросов
    
    Принципы:
    - Минимальная очистка (не агрессивная)
    - Сохранение доменной терминологии
    - Unicode нормализация
    - Различная обработка для BM25 и Dense моделей
    """
    
    def __init__(self, lowercase: bool = False, remove_punctuation: bool = False):
        """
        Args:
            lowercase: Приводить к lowercase (для BM25, не для neural)
            remove_punctuation: Удалять пунктуацию
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        logger.debug(f"TextCleaner initialized: lowercase={lowercase}, remove_punct={remove_punctuation}")
    
    def clean(self, text: str) -> str:
        """
        Базовая очистка текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный текст
            
        Examples:
            >>> cleaner = TextCleaner()
            >>> cleaner.clean("  Привет,   мир!  ")
            'Привет, мир!'
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. Unicode нормализация (NFC)
        text = unicodedata.normalize('NFC', text)
        
        # 2. HTML entities
        text = html.unescape(text)
        
        # 3. Удаление контрольных символов
        text = self._remove_control_chars(text)
        
        # 4. Нормализация пробелов
        text = re.sub(r'\s+', ' ', text)
        
        # 5. Удаление пунктуации (опционально)
        if self.remove_punctuation:
            text = self._remove_punctuation(text)
        
        # 6. Lowercase (опционально)
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def _remove_control_chars(self, text: str) -> str:
        """Удаление контрольных символов кроме \n и \t"""
        return ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
    
    def _remove_punctuation(self, text: str) -> str:
        """Удаление пунктуации (сохраняем буквы, цифры, пробелы)"""
        return re.sub(r'[^\w\s]', ' ', text)
    
    def clean_for_bm25(self, text: str) -> str:
        """
        Специальная очистка для BM25 (более агрессивная)
        
        Args:
            text: Исходный текст
            
        Returns:
            Текст готовый для BM25 токенизации
        """
        text = self.clean(text)
        text = text.lower()
        # Удаляем URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Удаляем email
        text = re.sub(r'\S+@\S+', '', text)
        # Удаляем числа (опционально)
        # text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def clean_for_dense(self, text: str) -> str:
        """
        Минимальная очистка для Dense моделей (сохраняем больше контекста)
        
        Args:
            text: Исходный текст
            
        Returns:
            Текст готовый для neural embeddings
        """
        text = self.clean(text)
        # НЕ делаем lowercase - neural модели учитывают case
        # НЕ удаляем пунктуацию - она важна для контекста
        
        return text


class RussianTextNormalizer:
    """
    Дополнительная нормализация для русского текста
    """
    
    # Словарь замен для нормализации
    REPLACEMENTS = {
        'ё': 'е',  # Опционально: унификация ё -> е
    }
    
    @staticmethod
    def normalize(text: str, replace_yo: bool = False) -> str:
        """
        Нормализация русского текста
        
        Args:
            text: Исходный текст
            replace_yo: Заменять ё на е
            
        Returns:
            Нормализованный текст
        """
        if replace_yo:
            text = text.replace('ё', 'е').replace('Ё', 'Е')
        
        return text
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Простое определение языка текста
        
        Returns:
            'ru', 'en' или 'unknown'
        """
        russian_chars = len(re.findall(r'[а-яА-ЯёЁ]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = russian_chars + english_chars
        if total_chars == 0:
            return 'unknown'
        
        if russian_chars / total_chars > 0.5:
            return 'ru'
        elif english_chars / total_chars > 0.5:
            return 'en'
        else:
            return 'unknown'
