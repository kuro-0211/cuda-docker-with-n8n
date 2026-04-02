"""
WSLg + CUDA 시연 스크립트
WSL2 터미널에서 직접 실행: python3 cuda_wslg_demo.py

사전 설치:
  sudo apt install -y python3-tk
  pip install torch matplotlib

실행하면 matplotlib GUI 창이 Windows 바탕화면에 WSLg를 통해 뜹니다.
"""
import torch
import matplotlib
matplotlib.use('TkAgg')  # WSLg GUI backend
import matplotlib.pyplot as plt
import time
import sys


def main():
    print("=" * 60)
    print("  WSLg + CUDA 시연")
    print("=" * 60)

    # CUDA 확인
    if not torch.cuda.is_available():
        print("ERROR: CUDA를 사용할 수 없습니다!")
        print("torch.cuda.is_available() == False")
        print("NVIDIA 드라이버와 CUDA Toolkit 설치를 확인하세요.")
        sys.exit(1)

    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    print(f"  CUDA version    : {torch.version.cuda}")
    print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    print(f"  GPU count       : {torch.cuda.device_count()}")
    print("=" * 60)
    print()

    # CPU vs GPU 행렬곱 벤치마크
    sizes = [100, 500, 1000, 2000, 3000, 4000]
    cpu_times = []
    gpu_times = []

    print("Running CPU vs GPU matrix multiplication benchmark...")
    print("-" * 60)

    for n in sizes:
        # CPU 벤치마크
        a_cpu = torch.randn(n, n)
        b_cpu = torch.randn(n, n)
        start = time.time()
        _ = torch.mm(a_cpu, b_cpu)
        cpu_times.append(time.time() - start)

        # GPU 벤치마크
        a_gpu = torch.randn(n, n, device='cuda')
        b_gpu = torch.randn(n, n, device='cuda')
        torch.cuda.synchronize()
        start = time.time()
        _ = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_times.append(time.time() - start)

        speedup = cpu_times[-1] / gpu_times[-1] if gpu_times[-1] > 0 else float('inf')
        print(
            f"  Matrix {n:>5}x{n:<5} | "
            f"CPU: {cpu_times[-1]:.4f}s | "
            f"GPU: {gpu_times[-1]:.6f}s | "
            f"Speedup: {speedup:.1f}x"
        )

    print("-" * 60)
    print("Benchmark complete! Opening WSLg GUI window...")
    print()

    # WSLg GUI 창으로 결과 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 왼쪽: 시간 비교 (로그 스케일)
    ax1.plot(sizes, cpu_times, 'o-', label='CPU', linewidth=2, color='#e74c3c', markersize=8)
    ax1.plot(sizes, gpu_times, 's-', label='GPU (CUDA)', linewidth=2, color='#2ecc71', markersize=8)
    ax1.set_xlabel('Matrix Size (N×N)', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('CPU vs GPU: Matrix Multiplication', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 오른쪽: 속도 배수
    speedups = [c / g if g > 0 else 0 for c, g in zip(cpu_times, gpu_times)]
    bars = ax2.bar(
        [str(s) for s in sizes], speedups,
        color=['#3498db', '#2980b9', '#2471a3', '#1f639a', '#1a5276', '#154360']
    )
    ax2.set_xlabel('Matrix Size (N×N)', fontsize=12)
    ax2.set_ylabel('Speedup (x times faster)', fontsize=12)
    ax2.set_title('GPU Speedup over CPU', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 바 위에 숫자 표시
    for bar, sp in zip(bars, speedups):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{sp:.1f}x', ha='center', fontsize=10, fontweight='bold'
        )

    fig.suptitle(
        f'WSLg CUDA Demo — {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}',
        fontsize=11, color='gray', y=0.02
    )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
