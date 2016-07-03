import matplotlib.pyplot as plt
from sklearn import metrics
'''
Classification metrics
'''
modelname = "STRING OF MODEL NAME"

accuracy = metrics.accuracy_score(y_test, y_preds)
precision = metrics.precision_score(y_test,y_preds)
recall = metrics.recall_score(y_test,y_preds)
classification_report = metrics.classification_report(y_test, y_preds)
confusion_matrix = metrics.confusion_matrix(y_test, y_preds)

print modelname,"Accuracy Score: ", accuracy
print modelname,"Precision: ", precision
print modelname,"Recall: ", recall
print
print confusion_matrix
print
print classification_report


'''ROC Curve '''

# We have already imported metrics which is what we need to build the ROC Curve

# ROC - Reciever Operating Characteristic, used to measure basically the False Positive
# rate (x-axis) against the true positive rate (y-axis)

## Insert your instantiated and fit model below
model = lgr_fit


#Here we are creating our predicted probabilities. This goes beyond whether something is
# predicted as a 1 or a zero, but is their actual probability (from 0 to 1) that the output
# is a 1. For example, if the predic_proba value is .34, there is a 34% chance that datapoint
# is a 1
y_probs = model.predict_proba(X_test)

#False Positive Rate, True Poisitive Rate
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_probs[:,1])
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

#### Plot the ROC Curve

# choosing a background style for our graph
plt.style.use('ggplot')
# increasing the size of the figure
plt.figure(figsize = (13, 11))
#giving the graph a title
plt.title('Receiver Operating Characteristic', fontsize = 24)
# plotting the fpr and tpr that were defined above on our graph
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
# creating a legend in the lower right section of the graph with the plot label
plt.legend(loc='lower right', fontsize = 24)
#drawing a dashed straight line from points (0, 0) to (1, 1)
plt.plot([0,1],[0,1],'r--')
# setting the scale for each axis
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
# drawing the axis lines and x and y = 0
plt.axhline(0, color='black')
plt.axvline(0, color='black')
#labeling the x and y axes
plt.ylabel('True Positive Rate', fontsize = 24)
plt.xlabel('False Positive Rate', fontsize = 24)
# showing our plot
plt.show()


'''
Regression Metrics
'''
modelname = "STRING OF MODEL NAME"

explained_variance_score = metrics.explained_variance_score(y_test, y_preds)
absolute_error = metrics.mean_absolute_error(y_test, y_preds)
mean_squared_error = metrics.mean_squared_error(y_test, y_preds)
median_absolute_error = metrics.median_absolute_error(y_test, y_preds)
r2_score = metrics.r2_score(y_test, y_preds)

print modelname,"Explained Variance Score: ",explained_variance_score
print modelname,"Absolute Error: ",absolute_error
print modelname,"Mean Squared Error: ",mean_squared_error
print modelname,"Median Absolute Error: ",median_absolute_error
print modelname,"R-Sqaured Score: ",r2_score

# plot predicted values vs true values
plt.scatter(y_preds,y_test)
# plot best fit line (r^2)
fit = np.polyfit(y_preds, y_test, deg=1)
plt.plot(y_preds, fit[0] * y_preds + fit[1], color='red')
# choosing a background style for our graph
plt.style.use('ggplot')
# increasing the size of the figure
plt.figure(figsize = (13, 11))
#giving the graph a title
plt.title('Predicted vs True', fontsize = 24)
# set labels
plt.ylabel('True Values', fontsize = 24)
plt.xlabel('Predicted Values', fontsize = 24)
plt.show()
