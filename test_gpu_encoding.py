"""
Тест GPU encoding vs CPU
"""
import torch
from sentence_transformers import SentenceTransformer
import time

print("=" * 60)
print("GPU ENCODING TEST")
print("=" * 60)

# Проверка CUDA
print(f"\n🔹 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔹 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔹 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Тестовые данные
texts = ["Это тестовый документ о банковских услугах"] * 100

model_name = "intfloat/multilingual-e5-large"

# CPU encoding
print(f"\n⏱️ CPU encoding...")
model_cpu = SentenceTransformer(model_name, device='cpu')
start = time.time()
embeddings_cpu = model_cpu.encode(texts, batch_size=32, show_progress_bar=False)
cpu_time = time.time() - start
print(f"   {cpu_time:.2f}s ({len(texts)/cpu_time:.1f} docs/s)")

# GPU encoding
if torch.cuda.is_available():
    print(f"\n⚡ GPU encoding...")
    model_gpu = SentenceTransformer(model_name, device='cuda')
    
    # Warmup
    _ = model_gpu.encode(texts[:10], batch_size=32, show_progress_bar=False)
    
    start = time.time()
    embeddings_gpu = model_gpu.encode(texts, batch_size=64, show_progress_bar=False)
    gpu_time = time.time() - start
    print(f"   {gpu_time:.2f}s ({len(texts)/gpu_time:.1f} docs/s)")
    
    speedup = cpu_time / gpu_time
    print(f"\n🚀 Speedup: {speedup:.1f}x faster with GPU")
    
    print(f"\n📊 For 2000 documents:")
    print(f"   CPU: ~{(2000/len(texts)*cpu_time)/60:.1f} minutes")
    print(f"   GPU: ~{(2000/len(texts)*gpu_time)/60:.1f} minutes")

print("\n" + "=" * 60)