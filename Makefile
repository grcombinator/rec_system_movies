.PHONY: setup format lint test docker-up docker-down

setup:
	pip install -e .

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

lint:
	ruff check src/ tests/
	mypy src/

test:
	pytest tests/ -v

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down -v