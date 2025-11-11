"""
Модуль для работы с конфигурацией
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class Config:
    """
    Класс для загрузки и управления конфигурацией проекта
    
    Examples:
        >>> config = Config("config/base.yaml")
        >>> model_name = config.get("retrieval.dense.model_name")
        >>> use_reranking = config.get("retrieval.use_reranking", default=False)
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Путь к YAML файлу конфигурации
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.config = self._load_config()
        logger.info(f"Configuration loaded from {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка YAML конфигурации"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Валидация обязательных полей
        self._validate_config(config)
        
        return config
    
    def _validate_config(self, config: dict):
        """Проверка наличия обязательных полей"""
        required_keys = ['data', 'preprocessing', 'retrieval']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        logger.debug("Config validation passed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получение значения по ключу с поддержкой вложенности через точку
        
        Args:
            key: Ключ в формате "section.subsection.parameter"
            default: Значение по умолчанию если ключ не найден
            
        Returns:
            Значение конфигурации или default
            
        Examples:
            >>> config.get("retrieval.bm25.k1")
            1.5
            >>> config.get("nonexistent.key", default=42)
            42
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Key '{key}' not found, returning default: {default}")
            return default
    
    def set(self, key: str, value: Any):
        """
        Установка значения по ключу
        
        Args:
            key: Ключ в формате "section.subsection.parameter"
            value: Новое значение
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Set {key} = {value}")
    
    def save(self, output_path: Optional[str] = None):
        """
        Сохранение конфигурации в YAML файл
        
        Args:
            output_path: Путь для сохранения (если None, перезаписывает исходный)
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
        
        logger.info(f"Configuration saved to {save_path}")
    
    def __getitem__(self, key: str) -> Any:
        """Доступ через квадратные скобки"""
        return self.config[key]
    
    def __repr__(self) -> str:
        return f"Config(path={self.config_path}, keys={list(self.config.keys())})"


def load_config(config_path: str = "config/base.yaml") -> Config:
    """
    Удобная функция для загрузки конфигурации
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Объект Config
    """
    return Config(config_path)
