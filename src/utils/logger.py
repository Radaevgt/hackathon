"""
Модуль настройки логирования
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "10 days"
):
    """
    Настройка логгера для проекта
    
    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_file: Путь к файлу для логов (если None, только в stdout)
        rotation: Параметр ротации файлов логов
        retention: Время хранения старых логов
        
    Examples:
        >>> setup_logger("DEBUG", "logs/app.log")
        >>> logger.info("Application started")
    """
    # Удаляем дефолтный handler
    logger.remove()
    
    # Формат логов
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Console handler
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    # File handler (опционально)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
        
        logger.info(f"Logging to file: {log_file}")
    
    logger.info(f"Logger initialized with level: {log_level}")
    return logger


# Инициализация дефолтного логгера
setup_logger()
