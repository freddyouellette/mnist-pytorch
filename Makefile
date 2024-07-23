install: create-env install-deps

create-env:
	python -m venv .venv

install-deps:
	pip install -r requirements.txt

save-env:
	pip freeze > requirements.txt

notebook:
	jupyter notebook ./train.ipynb

demo:
	streamlit run ./app.py