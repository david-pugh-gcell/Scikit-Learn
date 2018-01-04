# Playing with Scilearn kit in Jupyter Notebooks

Working through [Scikit-learn](http://scikit-learn.org/stable/) library to 
Influenced heavily by Python Machine Learning (Sebastian Raschka) Chapter 3.

## Data
Many of the examples use the [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris), which comes bundled with Scikit-Learn. This is loaded, split into test:training sets and standardised before it is used for the modelling. 

Data splitting is perfeomed using the [test_train_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) utility, defaulting with a test size of 30%.

```Python
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

The [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) is used for standardisation - the fit function estimates the sample mean and std deviation for each feature dimension in the training data, and these are used to transform the training and test data sets. 

```Python
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

## Utilities Module
The utilities Python Module has a number of useful functions that are used in the notebooks:

*  `get_iris_data(test_size=0.3)`  -> Iris Data Set retrieval, split into train/test set (default 30%, can be changed by setting test_size parameter) and standardised. Returns the `X_train_std, y_train, X_test_std, y_test, X_combined_std, y_combined`


* `plot_decision_regions(X,y,classifier, test_idx=None, resolution=0.02)`  -> plots the data and the decision regions produced by a model
