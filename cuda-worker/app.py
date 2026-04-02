from flask import Flask, request, jsonify
import cupy as cp
import time
import threading
import subprocess
import json as json_module
import pymysql

app = Flask(__name__)

MYSQL_CONFIG = {
    "host": "mysql",
    "port": 3306,
    "user": "n8n",
    "password": "n8n1234",
    "database": "gpu_monitoring",
    "charset": "utf8mb4",
}

def get_db():
    return pymysql.connect(**MYSQL_CONFIG)

status = {
    "running": False,
    "progress": 0,
    "result": None
}

stop_flag = threading.Event()


def particle_simulation(n_particles, n_steps):
    """CuPy N-body 파티클 시뮬레이션 — GPU에 O(N^2) 부하"""
    global status
    stop_flag.clear()
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
            # 종료 요청 확인
            if stop_flag.is_set():
                status["result"] = {
                    "message": "Simulation stopped by user",
                    "stopped_at_step": step,
                    "n_particles": n_particles,
                    "n_steps": n_steps,
                }
                break

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


@app.route('/stop', methods=['POST'])
def stop():
    """실행 중인 시뮬레이션 종료"""
    if not status["running"]:
        return jsonify({"message": "No simulation running"}), 400

    stop_flag.set()
    return jsonify({"message": "Stop signal sent, simulation will terminate shortly"})


@app.route('/metrics', methods=['GET'])
def metrics():
    """GPU 메트릭 수집 (n8n → MySQL 저장용)"""
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        parts = [p.strip() for p in result.stdout.strip().split(',')]

        gpu_name = parts[0]
        gpu_util = float(parts[1])
        mem_used = float(parts[2])
        mem_total = float(parts[3])
        temperature = float(parts[4])
        power_usage = float(parts[5]) if parts[5] != '[N/A]' else 0.0

        return jsonify({
            "gpu_name": gpu_name,
            "gpu_util": gpu_util,
            "mem_used_mb": mem_used,
            "mem_total_mb": mem_total,
            "temperature": temperature,
            "power_usage": power_usage,
            "simulation_running": status["running"],
            "simulation_progress": status["progress"],
            "n_particles": status["result"].get("n_particles", 0) if status["result"] else 0,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/save-metrics', methods=['POST'])
def save_metrics():
    """GPU 메트릭을 수집하고 MySQL에 직접 저장"""
    try:
        # GPU 메트릭 수집
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        parts = [p.strip() for p in result.stdout.strip().split(',')]

        gpu_name = parts[0]
        gpu_util = float(parts[1])
        mem_used = float(parts[2])
        mem_total = float(parts[3])
        temperature = float(parts[4])
        power_usage = float(parts[5]) if parts[5] != '[N/A]' else 0.0

        # MySQL에 저장
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO gpu_metrics
                    (gpu_name, gpu_util, mem_used_mb, mem_total_mb, temperature,
                     power_usage, simulation_running, simulation_progress, n_particles, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (gpu_name, gpu_util, mem_used, mem_total, temperature,
                     power_usage, status["running"], status["progress"],
                     status["result"].get("n_particles", 0) if status["result"] else 0,
                     "n8n")
                )
            conn.commit()
        finally:
            conn.close()

        return jsonify({
            "message": "Metrics saved to MySQL",
            "gpu_name": gpu_name,
            "gpu_util": gpu_util,
            "temperature": temperature,
            "power_usage": power_usage,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/save-result', methods=['POST'])
def save_result():
    """시뮬레이션 결과를 MySQL에 직접 저장 (종료 대기 포함)"""
    try:
        # 시뮬레이션이 아직 실행 중이면 종료될 때까지 최대 10초 대기
        for _ in range(20):
            if not status["running"]:
                break
            time.sleep(0.5)

        if not status["result"]:
            return jsonify({"error": "No simulation result available"}), 400

        r = status["result"]
        sim_status = "completed"
        if r.get("message", "").startswith("Simulation stopped"):
            sim_status = "stopped"
        elif r.get("error"):
            sim_status = "error"

        # GPU 이름이 없으면 직접 조회
        gpu_name = r.get("gpu_name", "")
        if not gpu_name:
            try:
                gpu_props = cp.cuda.runtime.getDeviceProperties(0)
                gpu_name = gpu_props["name"]
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode()
            except Exception:
                gpu_name = "unknown"

        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO simulation_results
                    (gpu_name, n_particles, n_steps, kinetic_energy,
                     gpu_memory_used_mb, cuda_version, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (gpu_name,
                     r.get("n_particles", 0),
                     r.get("n_steps", 0),
                     r.get("final_stats", {}).get("kinetic_energy", 0) if isinstance(r.get("final_stats"), dict) else 0,
                     r.get("gpu_memory_used_mb", 0),
                     r.get("cuda_version", ""),
                     sim_status)
                )
            conn.commit()
        finally:
            conn.close()

        return jsonify({
            "message": "Result saved to MySQL",
            "status": sim_status,
            "gpu_name": gpu_name,
            "n_particles": r.get("n_particles", 0),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
