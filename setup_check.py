"""
Скрипт быстрой установки и проверки окружения
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Проверка версии Python"""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required, but found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_requirements():
    """Установка зависимостей"""
    print("\nInstalling requirements...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        print("✅ Requirements installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False


def create_directories():
    """Создание необходимых директорий"""
    print("\nCreating directories...")
    
    dirs = [
        "data/raw",
        "data/processed",
        "data/indices",
        "data/submissions",
        "logs",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")
    return True


def check_data_files():
    """Проверка наличия файлов данных"""
    print("\nChecking data files...")
    
    required_files = [
        "data/raw/websites.csv",
        "data/raw/questions.csv"
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print(f"⚠️  Missing data files:")
        for f in missing:
            print(f"   - {f}")
        print("\nPlease place data files in data/raw/ directory")
        return False
    
    print("✅ Data files found")
    return True


def test_imports():
    """Тестирование импортов"""
    print("\nTesting imports...")
    
    required_modules = [
        'torch',
        'transformers',
        'sentence_transformers',
        'faiss',
        'rank_bm25',
        'pandas',
        'numpy',
        'sklearn',
        'razdel',
        'yaml',
        'loguru',
    ]
    
    failed = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            failed.append(module)
            print(f"  ❌ {module}")
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    
    print("\n✅ All imports successful")
    return True


def main():
    """Главная функция установки"""
    print("=" * 60)
    print("ALPHA RAG SYSTEM - SETUP")
    print("=" * 60)
    
    steps = [
        ("Python version", check_python_version),
        ("Install requirements", install_requirements),
        ("Create directories", create_directories),
        ("Check data files", check_data_files),
        ("Test imports", test_imports),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Place data files in data/raw/")
    print("2. Run: python scripts/build_indices.py")
    print("3. Run: python scripts/run_retrieval.py")
    print("\nFor help: python scripts/build_indices.py --help")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
