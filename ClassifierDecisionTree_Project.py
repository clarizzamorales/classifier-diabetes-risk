#!/usr/bin/env python
# coding: utf-8

# In[29]:

"""
****************************************************************************************

Project: Diabetes Risk Classifier
Organization: DiscoverAI
Author: Clarizza Morales
Date: Fall 2022
Version: 1.2    
Description:
This program (python script from upyter notebook) implements a decision tree classifier to predict
diabetes from a CDC dataset to identify factors of low-risk and high-risk diabetes.
It reads data from an Excel file, processes it, computes entropy,
and trains a decision tree model. The model's performance is evaluated
by calculating the average accuracy score over multiple iterations.
*****************************************************************************************
"""


from sklearn import tree
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np

# Import dataset and the decision tree 
# classifier from sklearn 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree


# In[22]:


#Import Dataset as a CSV file
data_frame = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")


# In[23]:


x, y = data_frame.BMI, data_frame.Diabetes_binary #getting the data in x and y (x is data, y is labels)


# In[30]:


print(data_frame.BMI)


# In[25]:


X = x.values.reshape(-1, 1)
print('This is the X arr', X)
Y = y.values.reshape(-1, 1)
print('This is the Y arr',Y)


# In[26]:


arr1 = X.reshape(126840, 2)
print ('After reshaping having dimension 4x2:')
print (arr1)
print ('\n')
arr2 = Y.reshape(126840, 2)
print ('After reshaping having dimension 4x2:')
print (arr1)
print ('\n')


# In[27]:


# Instantiate a decision tree classifier
decision_tree_classifier = tree.DecisionTreeClassifier()
# Train the decision tree on the data (X,y)
decision_tree_classifier = decision_tree_classifier.fit(arr1,arr2)


# In[28]:


#Plot the newly trained decision tree classifier
tree.plot_tree(decision_tree_classifier)


# In[ ]:





# In[32]:


# Load the data and split it into arrays X and y
X_train,X_test,y_train,y_test=train_test_split(arr1,arr2,test_size=0.90)

# TODO::Instantiate a decision tree classifier
clf = tree.DecisionTreeClassifier()
# TODO::Train the decision tree on the data (X_train,y_train)
clf = clf.fit(X_train,y_train)


# In[33]:


# Save the sixth row of the testing data (X_test) as single_sample.
single_sample = X_test[5]


# In[34]:


single_sample


# In[35]:


# Test the decision tree (clf) on the sixth row of our testing data. 
clf.predict([single_sample])


# In[43]:


#Now we want to predict the labels for all the values in the testing set X_test.
y_pred = clf.predict(X_test)
# TODO: Print the predicted values for the test set (y_pred)
print("y_pred:","\n",y_pred)
# TODO: Print the given labels for test set (y_test)
print("y_test:","\n",y_test)


# In[44]:


#We can use these values to evaluate the performance of our model. 
#We first compare the predicted and true labels and store the comparison in an array of Boolean values (match). 
#Each Boolean value indicates if the predicted label matches the true label.
match = clf.predict(X_test) == y_test

# Print the comparison array (match)
print(match)


# In[45]:


#We can use the numpy library to count the number of true predictions. 
#We then compute the percentage of true predictions as our accuracy.
import numpy as np

correct = np.count_nonzero(match)
accuracy = correct/len(match)

#Print the accuracy
print(accuracy)


# In[47]:


#Visualization
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("Diabetes",directory='SupervisedLearning') 


# In[50]:


#The export_graphviz exporter also supports a variety of aesthetic options, including coloring nodes by their 
#class (or value for regression) and using explicit variable and class names if desired. 
#The results are saved in an output file Diabetes.pdf:
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=['BMI', 'Diabetes_binary'],
                     class_names=sorted(y.unique()),
                     special_characters=True)  
graph = graphviz.Source(dot_data)
graph.render("Diabetes_color",directory='SupervisedLearning')


# In[ ]:




