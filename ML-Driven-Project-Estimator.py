from mpl_toolkits.mplot3d import Axes3D 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt # plotting 
import numpy as np # linear algebra 
#import os # accessing directory structure 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 
data = pd.read_csv('data.csv') 
X_train, X_test, y_train, y_test = train_test_split(data.drop('Team selections', axis=1), 
data['Team selections'], test_size=0.79, random_state=42) 
rf = RandomForestClassifier(n_estimators=80, random_state=42) 
rf.fit(X_train, y_train) 
y_pred = rf.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 
# print accuracy score 
print('Accuracy:', accuracy*100) 
data['Project'] = data['Effort'] / data['Team size'] 
# Get the inputs from the user 
new_loc = int(input("Enter the estimated size of the new project (in LOC): ")) 
new_cost = data['Project'].mean() * new_loc 
print(f"The estimated cost for a project of {new_loc} LOC is: ${new_cost:.2f}") 
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow): 
    nunique = df.nunique() 
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying 
    nRow, nCol = df.shape 
    columnNames = list(df) 
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow 
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor 
= 'w', edgecolor = 'k') 
        plt.subplot(nGraphRow, nGraphPerRow, i + 1) 
        columnDf = df.iloc[:, i] 
            valueCounts = columnDf.value_counts() 
        plt.ylabel('counts') 
        plt.xticks(rotation = 90) 
        plt.title(f'{columnNames[i]} (column {i})') 
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0) 
    plt.show() 
def plotCorrelationMatrix(df, graphWidth): 
    filename = df.dataframeName 
    df = df.dropna('columns') # drop columns with NaN 
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more 
        print(f'No correlation plots shown: The number of non-NaN or constant columns 
    corr = df.corr() 
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', 
edgecolor='k') 
    corrMat = plt.matshow(corr, fignum = 1) 
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90) 
    plt.yticks(range(len(corr.columns)), corr.columns) 
    plt.gca().xaxis.tick_bottom() 
    plt.colorbar(corrMat) 
    plt.title(f'Correlation Matrix for {filename}', fontsize=15) 
    plt.show() 
nRowsRead = 100 # specify 'None' if want to read whole file 
df1 = pd.read_csv('data.csv', delimiter=',', nrows = nRowsRead) 
df1.dataframeName = 'data.csv' 
nRow, nCol = df1.shape 
print(f'There are {nRow} rows and {nCol} columns') 
plt.figure() 
plt.plot(df1) 
plt.ion() 
plt.show() 