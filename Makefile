# DyniSafety — common Make targets (M10/M12).
# Appendix C: submit with `make lint && pytest -m "not sim"` (or `make test`).

.PHONY: help install-libero install-simplerenv install-robotwin install-robocasa test lint coverage report example-agent-install example-agent-verify

help:
	@echo "DyniSafety Make targets:"
	@echo "  make install-libero         # scripts/setup/install_backend.sh libero"
	@echo "  make install-simplerenv     # scripts/setup/install_backend.sh simplerenv"
	@echo "  make install-robotwin       # scripts/setup/install_backend.sh robotwin"
	@echo "  make install-robocasa       # scripts/setup/install_backend.sh robocasa (numpy 2 env)"
	@echo "  bash scripts/setup/install_conda_env.sh   # optional: pick backend+VLA; skip existing envs"
	@echo "  make lint                   # ruff check src tests examples"
	@echo "  make test                   # pytest -m 'not sim'"
	@echo "  make coverage               # pytest not-sim with cov fail_under=80"
	@echo "  make report                 # dynisafety report --help (no heavy compute)"
	@echo "  make example-agent-install  # pip install -e examples/example_agent"
	@echo "  make example-agent-verify   # python examples/example_agent/verify.py"
	@echo ""
	@echo "Do NOT install multiple backends into one env (see requirements/README.md)."

install-libero:
	bash scripts/setup/install_backend.sh libero

install-simplerenv:
	bash scripts/setup/install_backend.sh simplerenv

install-robotwin:
	bash scripts/setup/install_backend.sh robotwin

install-robocasa:
	bash scripts/setup/install_backend.sh robocasa

lint:
	ruff check src tests examples

example-agent-install:
	pip install -e examples/example_agent

example-agent-verify:
	python examples/example_agent/verify.py

test:
	pytest -m "not sim"

coverage:
	pytest -m "not sim" \
		--cov=dynisafety.core \
		--cov=dynisafety.config \
		--cov=dynisafety.report \
		--cov-report=term-missing \
		--cov-fail-under=80

report:
	dynisafety report --help
