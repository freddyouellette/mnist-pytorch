install: create-env install-deps

create-env:
	[ -n "$$VIRTUAL_ENV" ] && echo $$VIRTUAL_ENV; \
	python3 -m venv venv
	[ ! -f .env ] && cp .env.dist .env

install-deps:
	[ -n "$$VIRTUAL_ENV" ] && deactivate; . ./venv/bin/activate; \
	. venv/bin/activate; \
	pip install -r requirements.txt

save-env:
	[ -n "$$VIRTUAL_ENV" ] && deactivate; . ./venv/bin/activate; \
	source venv/bin/activate; \
	pip freeze > requirements.txt

notebook:
	[ -n "$$VIRTUAL_ENV" ] && deactivate; . ./venv/bin/activate; \
	. venv/bin/activate; \
	jupyter notebook ./train.ipynb

demo:
	[ -n "$$VIRTUAL_ENV" ] && deactivate; . ./venv/bin/activate; \
	. venv/bin/activate; \
	streamlit run ./demo.py

# for custom make jobs
-include Makefile.local.mk