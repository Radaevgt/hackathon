"""
Тест GPU для E5 модели
"""
import torch
from sentence_transformers import SentenceTransformer
import time
import numpy as np

print("=" * 60)
print("  GPU SPEED TEST - E5 MODEL")
print("=" * 60)

# Проверка CUDA
print(f"\n1. CUDA Status:")
print(f"   Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  CUDA not available - will use CPU only")

# Загрузка модели
print(f"\n2. Loading E5 model...")
model = SentenceTransformer('intfloat/multilingual-e5-large')
print(f"   ✅ Model loaded")

# Тестовые данные (100 документов)
test_texts = [
    "Как открыть счёт в банке",
    "Условия кредита для малого бизнеса",
    "Дебетовая карта с кэшбэком"
] * 34  # ~100 текстов

print(f"\n3. Speed Test ({len(test_texts)} texts):")

# CPU
print(f"\n   CPU Encoding...")
model = model.to('cpu')
start = time.time()
embeddings_cpu = model.encode(
    test_texts, 
    batch_size=32,
    show_progress_bar=False,
    convert_to_numpy=True
)
cpu_time = time.time() - start
print(f"   CPU Time: {cpu_time:.2f} sec ({len(test_texts)/cpu_time:.1f} texts/sec)")

# GPU (если доступен)
if torch.cuda.is_available():
    print(f"\n   GPU Encoding...")
    model = model.to('cuda')
    
    # Warmup
    _ = model.encode(["warmup"], show_progress_bar=False)
    torch.cuda.synchronize()
    
    start = time.time()
    embeddings_gpu = model.encode(
        test_texts,
        batch_size=64,  # Больше batch для GPU
        show_progress_bar=False,
        convert_to_numpy=True
    )
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"   GPU Time: {gpu_time:.2f} sec ({len(test_texts)/gpu_time:.1f} texts/sec)")
    print(f"\n   🚀 SPEEDUP: {cpu_time/gpu_time:.1f}x faster!")
    
    # Проверка что результаты одинаковые
    diff = np.abs(embeddings_cpu - embeddings_gpu).mean()
    print(f"   Difference: {diff:.6f} (should be ~0)")
else:
    print(f"\n   ⚠️  GPU not available, skipping GPU test")

print("\n" + "=" * 60)
print("  Для нашего проекта (1938 docs + 6977 queries):")
if torch.cuda.is_available():
    estimated_gpu = (1938 + 6977) / (len(test_texts)/gpu_time) / 60
    estimated_cpu = (1938 + 6977) / (len(test_texts)/cpu_time) / 60
    print(f"  GPU: ~{estimated_gpu:.1f} минут")
    print(f"  CPU: ~{estimated_cpu:.1f} минут")
    print(f"  Экономия: ~{estimated_cpu - estimated_gpu:.1f} минут")
else:
    print(f"  ⚠️  GPU недоступен")
print("=" * 60)
