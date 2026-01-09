DATA_PATH ?= /home/su/Data
IMAGE_NAME ?= demapgs
IMAGE_TAG ?= latest
BASE_TAG ?= base
PYTHON_VERSION ?= 3.10
PROJECT_ROOT := $(shell pwd)
RUN_ARGS ?= --gpus all
CMD ?= bash
SETUP_CONTAINER := $(IMAGE_NAME)-setup

DOCKER_RUN := docker run --rm $(RUN_ARGS) -it -v $(PROJECT_ROOT):/workspace -v $(DATA_PATH):/Data -w /workspace $(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: setup build install-submodules run clean

build:
	DOCKER_BUILDKIT=1 docker build --build-arg PYTHON_VERSION=$(PYTHON_VERSION) -t $(IMAGE_NAME):$(BASE_TAG) .

install-submodules:
	-@docker rm -f $(SETUP_CONTAINER) >/dev/null 2>&1 || true
	docker run $(RUN_ARGS) --name $(SETUP_CONTAINER) -v $(PROJECT_ROOT):/workspace -w /workspace $(IMAGE_NAME):$(BASE_TAG) bash -lc "\
		set -e; \
		pip install --no-build-isolation git+https://github.com/hbb1/diff-surfel-rasterization.git && \
		pip install --no-build-isolation git+https://github.com/ShuyiZhou495/diff-gaussian-rasterization.git && \
		pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d@stable && \
		cd /workspace/include/gaussian_splat_mesh && python3 setup.py install && \
		pip install -r /workspace/requirements.txt"
	docker commit $(SETUP_CONTAINER) $(IMAGE_NAME):$(IMAGE_TAG)
	docker rm $(SETUP_CONTAINER)

setup: build install-submodules

run:
	$(DOCKER_RUN) $(CMD)

clean:
	-@docker rm -f $(SETUP_CONTAINER) >/dev/null 2>&1 || true
	-@docker rmi $(IMAGE_NAME):$(IMAGE_TAG) $(IMAGE_NAME):$(BASE_TAG) >/dev/null 2>&1 || true
