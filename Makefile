# ===== CONFIGURATION =====
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip


MODEL_NAME := openai/clip-vit-large-patch14
ONNX_MODEL := model_repository/clip_vision.onnx
TRT_MODEL := model_repository/clip_vision_fp16.trt
TRITON_DIR := models/clip_vision/1
CONFIG_FILE := models/clip_vision/config.pbtxt



# ===== TARGETS =====

.PHONY: all
all: install export quantize triton

# 1. Create virtualenv and install requirements
.PHONY: install
install:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision torchaudio transformers onnx onnxruntime tensorrt

# 2. Export CLIP vision encoder to ONNX using HuggingFace Transformers
.PHONY: export
export:
	mkdir -p $(dir $(ONNX_MODEL))
	$(PYTHON) scripts/export_clip_onnx.py $(MODEL_NAME) $(ONNX_MODEL)

# 3. Quantize ONNX model to TensorRT engine
.PHONY: quantize
quantize:
	mkdir -p $(dir $(TRT_MODEL))
	$(PYTHON) scripts/quantize_model.py $(ONNX_MODEL) $(TRT_MODEL)

# 4. Prepare Triton Inference Server model repository layout
.PHONY: triton
triton:
	mkdir -p $(TRITON_DIR)
	cp $(TRT_MODEL) $(TRITON_DIR)/model.plan
	cp $(CONFIG_FILE) model_repository/clip_vision/

# 5. Clean all outputs (models, engine, Triton directory, venv)
.PHONY: clean
clean:
	rm -rf $(VENV_DIR) models model_repository
