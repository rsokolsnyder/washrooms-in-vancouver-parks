# Washrooms in Vancouver Public Parks

Authors: Rebecca Sokol-Snyder, William Song, and Chung Ki (Harry) Yau

This is a Data Science project completed as part of DSCI 522 (Data Science Workflows), a course as part of the Masters of Data Science Program at the University of British Columbia

## About this project

We retrieved the public Park data from the City of Vancouver Open Data Portal. We aim to construct a classification model using the adequate algorithm to help us evaluate the factors that influenced the construction of washroom which is one of the important amenities in public parks. We used two models, KNN and SVM RBVF, to test our data. The cross-validation result shows that SVM RBF provides a small but consistent improvement over the baseline, while KNN overfits and fails to generalize. The dataset may require richer features or alternative models to achieve stronger predictive performance on washroom availability.

## Report

The final report can be found in the [python notebook](notebooks/washrooms_in_parks.ipynb)

## Usage

- If you are trying to run this project with docker, please follow the instruction below.

### Setup

1. Clone this repository

### Running the Analysis
1. Navigate to the root directory of the project.
2. run the command below on terminal
```
docker compose up
```
3. In terminal, look for a URL start with `http://127.0.0.1:8888/lab?token=`, copy the whole URL and paste into the browser
4. To run the analysis, open notebooks/washroom-in-vancouver-parks.ipynb in Jupyter Lab and under the "Kernel" menu click "Restart Kernel and Run All Cells"
#### Clean up
1. On the terminal, type `Ctrl + c` to terminate the container and run the command below in the same location:
```
docker compose rm
```
## Dependencies

`conda` (version 23.9.0 or higher)
Python and the package dependencies listed in the [environment.yml](environment.yml) file

## License

The Washrooms in Vancouver Public Parks report contained in this repository is licensed under the Creative Commons Legal Code CC0 1.0 Universal License. More information can be found in the [license file](LICENSE). If re-using or modifying the report please provide attribution. All software code contained within this repository is licensed under the MIT license. Once again you can see the [license file](LICENSE) for more information

## References

City of Vancouver. (2025, September 22). Parks. City of Vancouver Open Data Portal. https://opendata.vancouver.ca/explore/dataset/parks/information/?sort=-parkid

Timbers, T. (n.d.). breast_cancer_predictor_py [Source code]. GitHub. https://github.com/ttimbers/breast_cancer_predictor_py
