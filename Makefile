.PHONY: clean docs lint

clean:
	@echo removing __pycache__ directories
	@find turbo -name '__pycache__' -exec echo "> {}" \; -exec rm -rf "{}" +
	@find tests -name '__pycache__' -exec echo "> {}" \; -exec rm -rf "{}" +
	@echo
	@echo removing pyc files
	@find turbo -name '*.pyc' -delete -exec echo "> {}" \;
	@find tests -name '*.pyc' -delete -exec echo "> {}" \;
	@echo
	@echo removing other files
	@rm -rf tests/htmlcov
	@rm -rf tests/prof
	@echo

docs:
	cd docs && make remake

lint:
	find turbo -name '*.py' -exec pyflakes "{}" \;

test:
	cd tests && ./run_tests
