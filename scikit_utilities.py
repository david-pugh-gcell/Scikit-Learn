from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def get_iris_data():

    #Load Iris Dataset from SKLearn
    iris = datasets.load_iris()

    #To see a snapshot use Pandas to create dataframe
    #irisdf = pd.DataFrame(datasets.load_iris().data)
    #print(irisdf.head(10))

    #X should be all the rows and columns 3 and 4
    X=iris.data[:, [2,3]]
    y=iris.target


    #The flower class names are stored as integrers for optimal perfromance
    print(np.unique(y))

    ##split the dataset into training and test datasets - 30% will be test data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print('Training Data set is %d values long' %len(X_train))

    #Feature scaling for optimal performance
    """
    StandardScaler fit estimates the sample mean and std deviation for each feature dimension in the training data  
    The transform method we standardise teh training data using these estimated mean and std dev. We also use same paramters to scale the test dataset so they are comparible. 
    """
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    return X_train_std, y_train, X_test_std, y_test, X_combined_std, y_combined, X_train, X_test


#Plot convienece function to visualise decision boundary
def plot_decision_regions(X,y,classifier, test_idx=None, resolution=0.02):
    #Define markers and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #get min and max values for the 2 features, and create grid arrays 
    x1_min , x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min , x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    #Flatten the grid arrays to create a matrix that has same number of columns as the training data , then feed to the Perceptron
    # ravel Return a contiguous flattened array, ie flatten to 1D array
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    #Then reshape the resultant data to the same grid with same dimensions as xx1, xx2 so we can plot it
    Z = Z.reshape(xx1.shape)
    #Draw contour plot
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #for each classification label in teh output data plot the training data point with appropriate marker/color
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], 
                    alpha=0.8, 
                    c=cmap(idx), 
                    marker = markers[idx], 
                    label = cl)
    #Highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1], 
                    c='', facecolors='none', edgecolors='black',
                    alpha=1.0, 
                    linewidths=1, 
                    marker='o', 
                    s=55, 
                    label='test set') 

    return plt