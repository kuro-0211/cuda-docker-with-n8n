CREATE DATABASE IF NOT EXISTS gpu_monitoring;
USE gpu_monitoring;

CREATE TABLE IF NOT EXISTS gpu_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    gpu_name VARCHAR(100),
    gpu_util FLOAT COMMENT 'GPU 사용률 (%)',
    mem_used_mb FLOAT COMMENT 'GPU 메모리 사용량 (MB)',
    mem_total_mb FLOAT COMMENT 'GPU 메모리 전체 (MB)',
    temperature FLOAT COMMENT 'GPU 온도 (°C)',
    power_usage FLOAT COMMENT '전력 사용량 (W)',
    simulation_running BOOLEAN COMMENT '시뮬레이션 실행 중 여부',
    simulation_progress INT COMMENT '시뮬레이션 진행률 (%)',
    n_particles INT COMMENT '파티클 수',
    source VARCHAR(50) DEFAULT 'n8n' COMMENT '데이터 수집 출처'
);

CREATE TABLE IF NOT EXISTS simulation_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    gpu_name VARCHAR(100),
    n_particles INT,
    n_steps INT,
    kinetic_energy FLOAT,
    gpu_memory_used_mb INT,
    cuda_version VARCHAR(20),
    status VARCHAR(20) COMMENT 'completed / stopped / error'
);
