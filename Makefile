install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt black

format:
	$(shell which black) *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md 
	cat ./Results/metrics.txt >> report.md

	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.pnd)' >> report.md

	cml comment create report.md
	