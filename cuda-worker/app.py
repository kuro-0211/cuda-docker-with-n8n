from flask import Flask, request, jsonify
import cupy as cp
import time
import threading

app = Flask(__name__)

status = {
    "running": False,
    "progress": 0,
    "result": None
}


def particle_simulation(n_particles, n_steps):
    """CuPy N-body 파티클 시뮬레이션 — GPU에 O(N^2) 부하"""
    global status
    status["running"] = True
    status["progress"] = 0
    status["result"] = None

    try:
        # 파티클 초기화 (위치, 속도, 질량)
        pos = cp.random.randn(n_particles, 3).astype(cp.float32)
        vel = cp.random.randn(n_particles, 3).astype(cp.float32) * 0.01
        mass = cp.random.uniform(0.5, 2.0, (n_particles, 1)).astype(cp.float32)

        dt = 0.001
        softening = 0.1
        stats = []

        for step in range(n_steps):
            # N-body 중력 계산 (O(N^2) — GPU에 부하 확실히 걸림)
            diff = pos[:, cp.newaxis, :] - pos[cp.newaxis, :, :]   # (N, N, 3)
            dist_sq = cp.sum(diff ** 2, axis=2) + softening ** 2    # (N, N)
            inv_dist3 = dist_sq ** (-1.5)                           # (N, N)

            # 중력 가속도
            acc = -cp.sum(
                diff * (mass[cp.newaxis, :, :] * inv_dist3[:, :, cp.newaxis]),
                axis=1
            )

            # Verlet 적분
            vel += acc * dt
            pos += vel * dt

            # 주기적으로 통계 수집
            if step % 10 == 0:
                ke = float(
                    (0.5 * cp.sum(mass * cp.sum(vel ** 2, axis=1, keepdims=True))).get()
                )
                stats.append({
                    "step": int(step),
                    "kinetic_energy": ke,
                })
                status["progress"] = int((step / n_steps) * 100)

            cp.cuda.Stream.null.synchronize()

        # GPU 정보 수집
        gpu_props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = gpu_props["name"]
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()

        status["result"] = {
            "n_particles": n_particles,
            "n_steps": n_steps,
            "final_stats": stats[-1] if stats else {},
            "gpu_name": gpu_name,
            "gpu_memory_used_mb": int(cp.get_default_memory_pool().used_bytes() / 1024 / 1024),
            "cuda_version": str(cp.cuda.runtime.runtimeGetVersion()),
        }
    except Exception as e:
        status["result"] = {"error": str(e)}
    finally:
        cp.get_default_memory_pool().free_all_blocks()
        status["running"] = False
        status["progress"] = 100


@app.route('/health', methods=['GET'])
def health():
    """GPU 상태 확인 엔드포인트"""
    try:
        mem = cp.cuda.runtime.memGetInfo()
        gpu_props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = gpu_props["name"]
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        return jsonify({
            "status": "ok",
            "cuda_available": True,
            "gpu": gpu_name,
            "gpu_memory_free_mb": int(mem[0] / 1024 / 1024),
            "gpu_memory_total_mb": int(mem[1] / 1024 / 1024),
            "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/simulate', methods=['POST'])
def simulate():
    """파티클 시뮬레이션 시작 (비동기)"""
    data = request.get_json() or {}
    n_particles = data.get('n_particles', 4096)
    n_steps = data.get('n_steps', 200)

    if status["running"]:
        return jsonify({"error": "Simulation already running"}), 409

    thread = threading.Thread(
        target=particle_simulation,
        args=(n_particles, n_steps)
    )
    thread.start()

    return jsonify({
        "message": "Simulation started",
        "n_particles": n_particles,
        "n_steps": n_steps
    })


@app.route('/status', methods=['GET'])
def get_status():
    """현재 시뮬레이션 상태 조회"""
    return jsonify(status)


if __name__ == '__main__':
    print("=== CuPy CUDA Worker ===")
    print(f"CuPy version: {cp.__version__}")
    print(f"CUDA Runtime: {cp.cuda.runtime.runtimeGetVersion()}")
    try:
        gpu_props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = gpu_props["name"]
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        print(f"GPU: {gpu_name}")
    except Exception as e:
        print(f"GPU detection error: {e}")
    print("========================")
    app.run(host='0.0.0.0', port=5000)
