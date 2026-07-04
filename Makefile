.PHONY: setup-cpu setup-gpu

setup-cpu:
	uv venv
	uv pip install torch --index-url https://download.pytorch.org/whl/cpu
	uv pip install -r pyproject.toml

setup-gpu:
	uv venv
	uv pip install torch
	uv pip install -r pyproject.toml
