.PHONY: install test eval demo clean

install:            ## install in editable mode with dev tools
	pip install -e ".[dev]"

test:               ## run the test suite
	pytest

eval:               ## run the golden-set evaluation
	python -m nanorag --data-dir .nanorag eval --golden evals/golden.jsonl

demo:               ## end-to-end offline demo over the bundled corpus
	python -m nanorag --data-dir .nanorag ingest docs/corpus
	python -m nanorag --data-dir .nanorag ask "What is ICE and why is it dangerous?" -s
	python -m nanorag --data-dir .nanorag eval --golden evals/golden.jsonl

clean:              ## remove caches and the local index
	rm -rf .pytest_cache build dist *.egg-info
	rm -rf .nanorag
	find . -name __pycache__ -type d -exec rm -rf {} +
