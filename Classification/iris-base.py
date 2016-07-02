from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
X = pd.DataFrame(iris.data,columns=iris.feature_names)
Y = iris.target
