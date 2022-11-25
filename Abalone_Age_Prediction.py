import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn import tree

import warnings
warnings.filterwarnings("ignore")

import plotly.graph_objs as go
import plotly.offline as py


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# %% Read csv file

data = pd.read_csv("abalone.csv")
data['age'] = data['Rings']+1.5
data.drop('Rings', axis=1, inplace=True)
# %% Check distribution of each variable

print('This dataset has {} observations with {} features.'.format(data.shape[0], data.shape[1]))
data.hist(figsize=(20, 10), grid=False, layout=(2, 4), bins=30)

# %% Checking missing values and see distribution of each variable

# Check if dataset has missing values
missing_values = data.isnull().sum().sort_values(ascending = False)
percentage_missing_values = (missing_values/len(data))*100
Check_missing_values = pd.concat([missing_values, percentage_missing_values], axis=1, keys= ['Missing values', '% Missing'])

print(Check_missing_values)

data.info()
data.shape
data.describe()

# %% See
sns.countplot(x = 'Sex', data = data, palette="Set3")

plt.figure(figsize = (20,7))
sns.swarmplot(x = 'Sex', y = 'age', data = data, hue = 'Sex')
sns.violinplot(x = 'Sex', y = 'age', data = data)

# Male : age majority lies in between 7.5 years to 19 years
# Female: age majority lies in between 8 years to 19 years
# Immature: age majority lies in between 6 years to < 10 years
# %%

# # create dummy variables
# sex_dummy = pd.get_dummies(data["Sex"])
# df = data.copy()

# df[sex_dummy.columns] = sex_dummy
# df = df.drop(["Sex"], axis = 1)

# %% pairplot
print(data.groupby('Sex')[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight', 'age']].mean().sort_values('age'))

numerical_features = data.select_dtypes(include=[np.number]).columns
categorical_features = data.select_dtypes(include=[np.object]).columns

sns.pairplot(data[numerical_features])

# %% Heatmap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(data.corr(),vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)

# Whole Weight is almost linearly varying with all other features except age
# Height has the least linearity with remaining features
# Age is most linearly proportional with Shell Weight followed by Diameter and length
# Age is least correlated with Shucked Weight

# %% Outliers handlings
# 'Viscera weight' outliers removal
var1 = 'Viscera weight'
plt.scatter(x = data[var1], y = data['age'],)
plt.grid(True)

data.drop(data[(data['Viscera weight']> 0.5) & (data['age'] < 20)].index, inplace=True)
data.drop(data[(data['Viscera weight']<0.5) & (data['age'] > 25)].index, inplace=True)

# %% 'Shell weight' outliers removal
var2 = 'Shell weight'
plt.scatter(x = data[var2], y = data['age'],)
plt.grid(True)
data.drop(data[(data['Shell weight']> 0.6) & (data['age'] < 25)].index, inplace=True)
data.drop(data[(data['Shell weight']< 0.8) & (data['age'] > 25)].index, inplace=True)

# %% 'Viscera weight' outliers removal
var3 = 'Shucked weight'
plt.scatter(x = data[var3], y = data["age"],)
plt.grid(True)
data.drop(data[(data['Shucked weight']>= 1) & (data['age'] < 20)].index, inplace=True)
data.drop(data[(data['Shucked weight']<1) & (data['age'] > 20)].index, inplace=True)

# %% 'Whole weight' outliers removal
var4 = 'Whole weight'
plt.scatter(x = data[var4], y = data['age'],)
plt.grid(True)
data.drop(data[(data['Whole weight']>= 2.5) & (data['age'] < 25)].index, inplace=True)
data.drop(data[(data['Whole weight']<2.5) & (data['age'] > 25)].index, inplace=True)


# %% "Diameter" outliers removal
var5 = "Diameter"
plt.scatter(x = data[var5], y = data["age"],)
plt.grid(True)
data.drop(data[(data['Diameter']<0.1) & (data['age'] < 5)].index, inplace=True)
data.drop(data[(data['Diameter']<0.6) & (data['age'] > 25)].index, inplace=True)
data.drop(data[(data['Diameter']>=0.6) & (data['age']< 25)].index, inplace=True)

# %% 'Height' outliers removal
var = 'Height'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)
data.drop(data[(data['Height']>0.4) & (data['age'] < 15)].index, inplace=True)
data.drop(data[(data['Height']<0.4) & (data['age'] > 25)].index, inplace=True)

# %% 'Length' outliers removal
var = 'Length'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)

data.drop(data[(data['Length']<0.1) & (data['age'] < 5)].index, inplace=True)
data.drop(data[(data['Length']<0.8) & (data['age'] > 25)].index, inplace=True)
data.drop(data[(data['Length']>=0.8) & (data['age']< 25)].index, inplace=True)

# %% Preprocessing, Modeling, Evaluation

# - pre-processing
# - suitable model selection
# - modeling
# - hyperparamaters turning using GridSearchCV
# - evaluation

# data = pd.get_dummies(data)
# dummy_data = data.copy()

# X = data.drop('age', axis = 1)
# y = data['age']
data.drop('Sex', axis = 1, inplace = True)
data.head()

data['age'].value_counts()
data["age"].mean()

df = data.copy()
Age = []
for i in df['age']:
    if i > 11.12:
        Age.append('1')
    else:
        Age.append('0')
df['Age'] = Age
df.drop('age', axis = 1, inplace = True)
df.head()



# %%
X = df.drop('Age', axis = 1).values
y = df['Age'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% Logistic Regression


lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print('Classification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
plot_confusion_matrix(lr, X_test, y_test)

lr_train_acc = lr.score(X_train, y_train)
print('Training Score: ', lr_train_acc)
lr_test_acc = lr.score(X_test, y_test)
print('Testing Score: ', lr_test_acc)

# %% Support Vector Classifiers


svc = SVC(C = 1, gamma= 1)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print('Classification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
plot_confusion_matrix(svc, X_test, y_test)

svc_train_acc = svc.score(X_train, y_train) 
print('Training Score: ', svc_train_acc)
svc_test_acc = svc.score(X_test, y_test)
print('Testing Score: ', svc_test_acc)

# %% K Nearest Neighbour Classifier


error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors= i)
    knn.fit(X_train, y_train)
    y_predi = knn.predict(X_test)
    error_rate.append(np.mean(y_test != y_predi))
    
plt.figure(figsize = (10,8))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

index_min_val = error_rate.index(min(error_rate))

knn = KNeighborsClassifier(n_neighbors= index_min_val)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('Classification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
plot_confusion_matrix(knn, X_test, y_test)
knn_train_acc = knn.score(X_train, y_train) 
print('Training Score: ', knn_train_acc)
knn_test_acc = knn.score(X_test, y_test)
print('Testing Score: ', knn_test_acc)


# %%  Decision Tree
dt = DecisionTreeClassifier(max_depth = 5)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print('Classification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
plot_confusion_matrix(dt, X_test, y_test)

dt_train_acc = dt.score(X_train, y_train) 
print('Training Score: ', dt_train_acc)
dt_test_acc = dt.score(X_test, y_test)
print('Testing Score: ', dt_test_acc)

# %% Random Forest Classifier

rf = RandomForestClassifier(n_estimators= 150, max_depth= 5)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print('Classification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
plot_confusion_matrix(rf, X_test, y_test)

rf_train_acc = rf.score(X_train, y_train) 
print('Training Score: ', rf_train_acc)
rf_test_acc = rf.score(X_test, y_test)
print('Testing Score: ', rf_test_acc)

fn=df.columns
cn=df["Age"]

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('rf_individualtree.png')

# For many estimations
# fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (4,4), dpi=800)
# for index in range(0, 5):
#     tree.plot_tree(rf.estimators_[index],
#                    feature_names = fn, 
#                    class_names=cn,
#                    filled = True,
#                    ax = axes[index]);

#     axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
# fig.savefig('rf_5trees.png')

# %% AdaBoost Classifier

adb = AdaBoostClassifier(n_estimators= 100)
adb.fit(X_train, y_train)
y_pred = adb.predict(X_test)

print('Classification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
plot_confusion_matrix(adb, X_test, y_test)
adb_train_acc = adb.score(X_train, y_train) 
print('Training Score: ', adb_train_acc)
adb_test_acc = adb.score(X_test, y_test)
print('Testing Score: ', adb_test_acc)

# %% Gradient Boosting

gdb = GradientBoostingClassifier(n_estimators= 200, max_depth = 2, min_samples_leaf= 2)
gdb.fit(X_train, y_train)
y_pred = gdb.predict(X_test)

print('Classification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
plot_confusion_matrix(gdb, X_test, y_test)

gdb_train_acc = gdb.score(X_train, y_train) 
print('Training Score: ', gdb_train_acc)
gdb_test_acc = gdb.score(X_test, y_test)
print('Testing Score: ', gdb_test_acc)


# %% XGBoost Classifier
xgb = XGBClassifier(objective = "binary:logistic", n_estimators = 100, max_depth = 3, subsample = 0.8, colsample_bytree = 0.6, learning_rate = 0.1)
y_train = [int(i) for i in y_train]
y_test = [int(i) for i in y_test]

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print('Classification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
xgb_train_acc = xgb.score(X_train, y_train) 
print('Training Score: ', xgb_train_acc)
xgb_test_acc = xgb.score(X_test, y_test)
print('Testing Score: ', xgb_test_acc)

# %% Plotting Accuracy graph using plotly
x = ['Logistic Regression','SVC', 'KNN', 'Decision Tree','Random Forest','AdaBoost','Gradient Boosting','XGBoost']
y1 = [lr_train_acc, svc_train_acc, knn_train_acc, dt_train_acc, rf_train_acc, adb_train_acc, gdb_train_acc, xgb_train_acc]
y2 = [lr_test_acc, svc_test_acc, knn_test_acc, dt_test_acc, rf_test_acc, adb_test_acc, gdb_test_acc, xgb_test_acc]

trace1 = go.Bar(x = x, y = y1, name = 'Training Accuracy', marker = dict(color = 'cyan'))
trace2 = go.Bar(x = x, y = y2, name = 'Testing Accuracy', marker = dict(color = 'violet'))
data = [trace1,trace2]
layout = go.Layout(title = 'Accuracy Plot', width = 750)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
