install:
	pip install -r requirements.txt

run:
	python ScenarioX.py

clean:
	rm -rf __pycache__ *.pyc plots/*.png
