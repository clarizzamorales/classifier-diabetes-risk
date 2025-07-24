"""
****************************************************************************************

Project: Diabetes Risk Classifier
Organization: DiscoverAI
Author: Clarizza Morales
Date: Fall 2022
Version: 2.0

Description:
This program implements a decision tree classifier to predict 
diabetes risk based on health (behavioral and clinical) indicators.
It reads data from an Excel file, processes it, computes entropy,
and trains a decision tree model. The model's performance is evaluated
by calculating the average accuracy score over multiple iterations.
The decision tree is visualized and saved as a PDF file.

Dependencies: pandas, numpy, matplotlib, scipy, sklearn, graphviz
*****************************************************************************************
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz

def main():
    try:
        df = pd.read_excel("diabetes_binary_health_indicators_BRFSS2015.xlsx")
    except FileNotFoundError:
        print("Excel file not found.")
        return

    df.drop_duplicates(inplace=True)
    df.loc[df["Diabetes_binary"] == 0, "Diabetes_binary"] = 'No Diabetes'
    df.loc[df["Diabetes_binary"] == 1, "Diabetes_binary"] = 'Diabetes'
    X = df.drop(['Diabetes_binary','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack','Fruits','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income'], axis=1)
    y = df['Diabetes_binary']

    k = len(y.unique())
    maxE = np.log2(k) 
    p_data = y.value_counts(normalize=True)
    entropy = scipy.stats.entropy(p_data)
    normalizedE = entropy/maxE

    avg_score = 0.0
    ntimes = 30
    for _ in range(ntimes):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        avg_score += accuracy_score(y_test, predictions)
    avg_score /= ntimes

    print('Normalized entropy value: %.3f'% normalizedE)
    print('Average accuracy score: %.3f' % avg_score)      

    dot_data = tree.export_graphviz(model, out_file=None,
                        feature_names=X.columns,
                        class_names=sorted(y.unique()),
                        label='all',
                        rounded=True,
                        filled=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")

if __name__ == "__main__":
    main()