import matplotlib.pyplot as plt
from sklearn import metrics

modelname = "STRING OF MODEL NAME"

accuracy = metrics.accuracy_score(y_test, y_preds)
precision = metrics.precision_score(y_test,y_preds)
recall = metrics.recall_score(y_test,y_preds)
classification_report = metrics.classification_report(y_test, y_preds)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

print modelname,"Accuracy Score: ", accuracy
print modelname,"Precision: ", precision
print modelname,"Recall: ", recall
print classification_report
print confusion_matrix


'''ROC Curve '''

##MODELNAME##_accuracy = metrics.accuracy_score(y_preds,y_test)

print "##MODEL NAME## Accuracy Score: ",##MODELNAME##_accuracy
print(metrics.classification_report(y_test, y_preds, target_names=iris.target_names))


model = ##INSERT MODEL##

#Probability
y_probs = model.predict_proba(X_test)

#False Positive Rate, True Poisitive Rate
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_probs[:,1])
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

#Plot the ROC Curve
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
