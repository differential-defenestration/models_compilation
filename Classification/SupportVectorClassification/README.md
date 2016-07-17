# Support Vector Machines (SVM)

SVM's are based on the concept of **decision planes**, which define boundaries (**hyperplanes**) that separate groups of data points. The goal is to use these decision boundaries to separate different points that belong to different classes.

The data points plugged into the SVM are rearranged using a set of mathematical functions (based on which **kernel** you select) and are then mapped out into that new feature space, with the hope that these new rearranged points will now form distinct groups which can be separated cleanly by the decision boundaries that will be drawn. The decision boundaries (**hyperplanes**) take on a shape defined by whichever **kernel** is chosen.

**Support Vectors** are actually just the data points that lie closest to the **hyperplane** that is constructed. The goal is to maximize the distance between the hyperplane and the nearest points (defined as the **margin**) - that way you are finding the maximum distance between the different groups of data points. The number of support vectors for each model changes depending on the parameters you set within the model.


#####  From Scikit-Learn:  
"Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.
The advantages of support vector machines are:
* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:
* If the number of features is much greater than the number of samples, the method is likely to give poor performances.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation"


##### More Resources  
[Idiot's Guide to SVM's](http://www.cs.ucf.edu/courses/cap6412/fall2009/papers/Berwick2003.pdf)

[Support Vector Machine Tutorial](http://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf)
