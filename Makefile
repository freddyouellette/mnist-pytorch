install: create-env install-deps

create-env:
	[ -n "$$VIRTUAL_ENV" ] && deactivate; \
	python -m venv venv
	[ ! -f .env ] && cp .env.dist .env

install-deps:
	[ -n "$$VIRTUAL_ENV" ] && deactivate; \
	. venv/bin/activate; \
	pip install -r requirements.txt

save-env:
	[ -n "$$VIRTUAL_ENV" ] && deactivate; \
	source venv/bin/activate; \
	pip freeze > requirements.txt

notebook:
	[ -n "$$VIRTUAL_ENV" ] && deactivate; \
	. venv/bin/activate; \
	jupyter notebook ./train.ipynb

demo:
	[ -n "$$VIRTUAL_ENV" ] && deactivate; \
	. venv/bin/activate; \
	streamlit run ./demo.py

# for custom make jobs
-include Makefile.local.mk