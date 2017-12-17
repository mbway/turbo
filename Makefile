.PHONY: clean docs lint

clean:
	@echo removing __pycache__ directories
	@find turbo -name '__pycache__' -exec echo "> {}" \; -exec rm -rf "{}" +
	@echo
	@echo removing pyc files
	@find turbo -name '*.pyc' -delete -exec echo "> {}" \;
	@echo

docs:
	cd docs && make remake

lint:
	find turbo -name '*.py' -exec pyflakes "{}" \;

