.PHONY: help clean

help:
	python -m sphinx

clean:
	rm -f checkpoint*.json
	rm -f *.pyc

