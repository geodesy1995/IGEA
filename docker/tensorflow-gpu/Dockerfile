FROM tensorflow/tensorflow:2.12.0-gpu

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true

WORKDIR /workspace

COPY docker/tensorflow-gpu/requirements-gpu.txt /tmp/requirements-gpu.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements-gpu.txt

CMD ["bash"]
