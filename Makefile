.PHONY: download split eda model evaluate report all clean

all: report

download: data/raw/parks.csv

data/raw/parks.csv:
	mkdir -p data/raw
	conda run -n milestone1 python scripts/download_data.py --url="https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/parks/exports/csv?lang=en&timezone=America%2FLos_Angeles&use_labels=true&delimiter=%3B" --write_to=data/raw

split: data/processed/parks_train.csv

data/processed/parks_train.csv: data/raw/parks.csv
	mkdir -p logs data/processed results/figures results/models
	conda run -n milestone1 python scripts/split_and_validate.py \
		--raw-data=data/raw/parks.csv \
		--logs-to=logs \
		--data-to=data/processed \
		--viz-to=results/figures \
		--preprocessor-to=results/models \
		--seed=123

eda: results/tables/eda_summary_stats.csv

results/tables/eda_summary_stats.csv: data/processed/parks_train.csv
	mkdir -p results/tables results/figures
	conda run -n milestone1 python scripts/eda.py \
		--training-data=data/processed/parks_train.csv \
		--plot-to=results/figures \
		--tables-to=results/tables

model: results/models/pipe_svm_rbf_fully_trained.pickle

results/models/pipe_svm_rbf_fully_trained.pickle: data/processed/parks_train.csv results/models/parks_preprocessor.pickle
	mkdir -p results/tables results/models
	conda run -n milestone1 python scripts/fit_parks_washroom_classifier.py \
		--train-data=data/processed/parks_train.csv \
		--preprocessor=results/models/parks_preprocessor.pickle \
		--results-to=results/tables \
		--pipeline-to=results/models \
		--seed=123

evaluate: results/tables/test_scores.csv

results/tables/test_scores.csv: data/processed/parks_test.csv results/models/pipe_svm_rbf_fully_trained.pickle
	mkdir -p results/tables results/figures
	conda run -n milestone1 python scripts/evaluate_parks_washroom_predictor.py \
		--test-data=data/processed/parks_test.csv \
		--pipeline-from=results/models/pipe_svm_rbf_fully_trained.pickle \
		--results-to=results/tables \
		--viz-to=results/figures \
		--seed=123

report: notebooks/washrooms_in_parks.html

notebooks/washrooms_in_parks.html: results/tables/test_scores.csv notebooks/washrooms_in_parks.qmd
	quarto render notebooks/washrooms_in_parks.qmd

clean:
	rm -rf data/raw data/processed logs results/models results/figures results/tables notebooks/washrooms_in_parks.html
