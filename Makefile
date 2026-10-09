.PHONY: install test baseline cem train eval video
WORKERS ?= $(shell python3 -c "import os;print(max(1,os.cpu_count()-1))")
RUN ?= runs/ppo_base

install:
	pip install -e .

test:
	pytest -q

baseline:
	python -m bowlrl.evaluate --baseline pd --episodes 5

cem:
	python -m bowlrl.optim.cem --iters 150 --pop 96 --elites 10 --workers $(WORKERS) --out data/reference

train:
	python -m bowlrl.train --config configs/base.yaml --run $(RUN) --n-envs $(WORKERS)

eval:
	python -m bowlrl.evaluate --run $(RUN) --episodes 100

video:
	MUJOCO_GL=egl python -m bowlrl.evaluate --run $(RUN) --episodes 3 --video $(RUN)/delivery.mp4
