# Playing with Scilearn kit in Jupyter Notebooks

Working through Scklearn kit 
Influenced heavily by Python Machine Learning (Sebastian Raschka) Chapter 3.

Many of the examples use the Iris Data Set, which comes bundled with SciLearnKit. This is loaded, split into test:training sets and standardised before it is used for the modelling.

The utilities Python Module has a number of useful functions that are used in the notebooks:

*  `get_iris_data(test_size=0.3)`  -> Iris Data Set retrieval, split into train/test set (default 30%, can be changed by setting test_size parameter) and standardised. Returns the `X_train_std, y_train, X_test_std, y_test, X_combined_std, y_combined`


* `plot_decision_regions(X,y,classifier, test_idx=None, resolution=0.02)`  -> plots the data and the decision regions produced by a model
