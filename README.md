# Washrooms in Vancouver Public Parks

Authors: Rebecca Sokol-Snyder, William Song, and Chung Ki (Harry) Yau

This is a Data Science project completed as part of DSCI 522 (Data Science Workflows), a course as part of the Masters of Data Science Program at the University of British Columbia

## About this project

We retrieved the public Park data from the City of Vancouver Open Data Portal. We aim to construct a classification model using the adequate algorithm to help us evaluate the factors that influenced the construction of washroom which is one of the important amenities in public parks. We used two models, KNN and SVM RBVF, to test our data. The cross-validation result shows that SVM RBF provides a small but consistent improvement over the baseline, while KNN overfits and fails to generalize. The dataset may require richer features or alternative models to achieve stronger predictive performance on washroom availability.

## Report

The final report can be found [https://github.com/rsokolsnyder/washrooms-in-vancouver-parks/blob/main/analysis.ipynb]

## Usage

If running this project for the first time, run the following code in a terminal window from the root directory of this repository
First time running the project, run the following from the root of this repository:

```
conda-lock install --n washrooms-in-vancouver-parks conda-lock.yml
```

Then, to run the analysis, open an instance of Jupyter Lab from the root of this repository using the command:

```
jupyter lab 
```

Next, open analysis.ipynb in Jupyter Lab and under Switch/Select Kernel choose "Python [conda env:washrooms-in-vancouver-parks]".

Finally, under the "Kernel" menu, select "Restart Kernel and Run All Cells".

## Dependencies

`conda` (version 23.9.0 or higher)
Python and the package dependencies listed in the [environment.yml](environment.yml) file

## License

The Washrooms in Vancouver Public Parks report contained in this repository is licensed under the Creative Commons Legal Code CC0 1.0 Universal License. More information can be found in the [license file](LICENSE). If re-using or modifying the report please provide attribution. All software code contained within this repository is licensed under the MIT license. Once again you can see the [license file](LICENSE) for more information

## References

City of Vancouver. (2025, September 22). Parks. City of Vancouver Open Data Portal. https://opendata.vancouver.ca/explore/dataset/parks/information/?sort=-parkid

Timbers, T. (n.d.). breast_cancer_predictor_py [Source code]. GitHub. https://github.com/ttimbers/breast_cancer_predictor_py