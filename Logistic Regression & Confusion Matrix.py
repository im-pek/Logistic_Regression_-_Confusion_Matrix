import pandas as pd
col_names = ['Product', 'Consumer_complaint_narrative']

# load dataset
df = pd.read_csv(r'C:\Users\yunni\Desktop\Consumer_Complaints\training_testing.csv', encoding='utf-8', header=0)#, names=col_names)

print (df)

consumer_complaints=df[df.Consumer_complaint_narrative.notnull()] ###

print (consumer_complaints)

from sklearn.preprocessing import LabelEncoder
number=LabelEncoder() #numbering features

consumer_complaints['Product']=number.fit_transform(consumer_complaints['Product'].astype('str'))
consumer_complaints['Consumer_complaint_narrative']=consumer_complaints['Consumer_complaint_narrative'].astype('str')

#print (consumer_complaints['Product'])

#feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
#X = consumer_complaints[feature_cols] # Features

X = consumer_complaints.Consumer_complaint_narrative # Features
y = consumer_complaints.Product # Target variable

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0) #16.667% of data used for model testing

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

X_train= X_train.values.reshape(-1, 1)
y_train= y_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
y_test= y_test.values.reshape(-1, 1)

# fit the model with data
logreg.fit(X_train,y_train) ##

#
y_pred=logreg.predict(X_test)

# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print (cnf_matrix)

###

# =============================================================================
# commaed_matrix=[]
# 
# for item in cnf_matrix:
#     commaed=list(item)
#     commaed_matrix.append(commaed)
# 
# #print (commaed_matrix)
#     
# import csv
# 
# with open("output.csv", "a") as fp:
#     wr = csv.writer(fp, dialect='excel')
#     wr.writerow(commaed_matrix)
# =============================================================================

###

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

class_names=[0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

###

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred))
#print("Recall:",metrics.recall_score(y_test, y_pred))