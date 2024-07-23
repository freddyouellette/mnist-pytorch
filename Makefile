install: create-env install-deps

create-env:
	python -m venv .venv

install-deps:
	pip install -r requirements.txt

save-env:
	pip freeze > requirements.txt