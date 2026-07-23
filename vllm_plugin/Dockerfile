ARG VLLM_IMAGE=vllm/vllm-openai:v0.19.0
FROM ${VLLM_IMAGE}

COPY . /opt/native-replay-vllm
RUN /usr/bin/python3 -m pip install --no-cache-dir /opt/native-replay-vllm && \
    /usr/bin/python3 /opt/native-replay-vllm/patch_sampler.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/sampler.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py && \
    grep -q "native-agent-replay post-sampling commitment" \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/sampler.py && \
    grep -q "native-agent-replay valid-sample mask" \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py
