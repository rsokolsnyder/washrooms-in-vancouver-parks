# Makefile
# author: Harry Yau
# date: 2025-12-13

.PHONY: all clean data split eda train evaluate report

# --------------------
# Default target
# --------------------
all: report

# --------------------
# Download data
# --------------------
data/raw/parks.csv:
	python scripts/download_data.py \
		--url="https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/parks/exports/csv?lang=en&timezone=America%2FLos_Angeles&use_labels=true&delimiter=%3B" \
		--write_to=data/raw

data: data/raw/parks.csv

# --------------------
# Split and validate
# --------------------
data/processed/parks_train.csv data/processed/parks_test.csv: data/raw/parks.csv
	python scripts/split_and_validate.py \
		--raw-data=data/raw/parks.csv \
		--logs-to=logs \
		--data-to=data/processed \
		--viz-to=results/figures \
		--preprocessor-to=results/models \
		--seed=123

split: data/processed/parks_train.csv data/processed/parks_test.csv

# --------------------
# Exploratory Data Analysis
# --------------------
eda: data/processed/parks_train.csv
	python scripts/eda.py \
		--training-data=data/processed/parks_train.csv \
		--plot-to=results/figures \
		--tables-to=results/tables

# --------------------
# Train model
# --------------------
results/models/pipe_svm_rbf_fully_trained.pickle: data/processed/parks_train.csv
	python scripts/fit_parks_washroom_classifier.py \
		--train-data=data/processed/parks_train.csv \
		--preprocessor=results/models/parks_preprocessor.pickle \
		--results-to=results/tables \
		--pipeline-to=results/models \
		--seed=123

train: results/models/pipe_svm_rbf_fully_trained.pickle

# --------------------
# Evaluate model
# --------------------
evaluate: results/models/pipe_svm_rbf_fully_trained.pickle data/processed/parks_test.csv
	python scripts/evaluate_parks_washroom_predictor.py \
		--test-data=data/processed/parks_test.csv \
		--pipeline-from=results/models/pipe_svm_rbf_fully_trained.pickle \
		--results-to=results/tables \
		--viz-to=results/figures \
		--seed=123

# --------------------
# Render report
# --------------------
notebooks/washrooms_in_parks.html: notebooks/washrooms_in_parks.qmd evaluate
	quarto render notebooks/washrooms_in_parks.qmd

report: notebooks/washrooms_in_parks.html

# --------------------
# Clean
# --------------------
clean:
	rm -rf data/processed/*
	rm -f results/figures/*
	rm -f results/tables/*
	rm -f results/models/*
	rm -f notebooks/*.html