FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages \
    flask \
    cupy-cuda13x

WORKDIR /app
COPY app.py .

EXPOSE 5000
CMD ["python3", "app.py"]
