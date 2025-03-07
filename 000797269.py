# ****************************************************************
# TO RUN 
# ****************************************************************
# - Ensure you have the Covid dataset in the current directory. It must be called 'latestdata 2.csv'
# - Execute python3 000797269.py from the current directory
# - Wait a few minutes and the graphs should appear
# ****************************************************************

import pandas as pd
import copy
import math
import datetime
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import timeit

# Read in CSV
try:
    df = pd.read_csv('latestdata 2.csv')
except:
    print('Error - dataset cannot be found in the current directory. Ensure you have the dataset in the current directory.')
    print('It must be called "latestdata 2.csv"')
    exit()



# Drop unwanted columns
del df['ID']
del df['city']
del df['country']
del df['latitude']
del df['longitude']
del df['geo_resolution']
del df['date_onset_symptoms']
del df['date_admission_hospital']
del df['symptoms']
del df['lives_in_Wuhan']
del df['travel_history_dates']
del df['travel_history_location']
del df['reported_market_exposure']
del df['chronic_disease']
del df['source']
del df['sequence_available']
del df['date_death_or_discharge']
del df['notes_for_discussion']
del df['location']
del df['admin3']
del df['admin2']
del df['admin1']
del df['admin_id']
del df['country_new']
del df['data_moderator_initials']
del df['additional_information']

# Remove nan values from key columns
df = df[df['outcome'].notna()]
df = df[df['age'].notna()]
df = df[df['sex'].notna()]
df = df[df['province'].notna()]
df = df[df['date_confirmation'].notna()]
df = df[df['travel_history_binary'].notna()]

# ****************************************************************
# CLEAN UP AGE COLUMN
# ****************************************************************

for i in range(0, len(df['age'].values)):
    if isinstance(df['age'].values[i], str):
        if ('-' in df['age'].values[i]) and (len(df['age'].values[i]) == 5):
            final = float((int(df['age'].values[i][3:5]) + int(df['age'].values[i][:2])) / 2)
        else:
            try:
                final = float(df['age'].values[i])
            except:
                final = float('nan')
        df['age'].values[i] = final
df = df[df['age'].notna()]

df['age'] = pd.to_numeric(df['age'])

# ****************************************************************
# CLEAN UP + TRANSFORM OUTCOME COLUMN 
# ****************************************************************

dead_words = ['death', 'died', 'Death', 'dead', 'Dead', 'Died', 'Deceased']
alive_words = ['discharge', 'discharged', 'Discharged', 'Discharged from hospital', 'not hospitalized', 'recovered', 'recovering at home 03.03.2020', 'released from quarantine', 'stable', 'treated in an intensive care unit (14.02.2020)', 'Alive', 'Recovered', 'Stable', 'stable condition', 'Migrated', 'Migrated_Other']
unclear_words = ['critical condition, intubated as of 14.02.2020', 'severe', 'Critical condition', 'severe illness', 'unstable', 'critical condition', 'Hospitalized', 'Symptoms only improved with cough. Currently hospitalized for follow-up.', 'https://www.mspbs.gov.py/covid-19.php', 'Under treatment', 'Receiving Treatment']

for word in unclear_words:
    df = df.drop(df[df['outcome'] == word].index)

for i in range(0, len(df['outcome'].values)):
    if df['outcome'].values[i] in dead_words:
        df['outcome'].values[i] = 'dead'
    elif df['outcome'].values[i] in alive_words:
        df['outcome'].values[i] = 'alive'


# ********************************************************************
# ONE HOT ENCODING + FEATURE TRANSFORMATION
# ********************************************************************

# Encoding for 'sex' column
dummy = pd.get_dummies(df['sex'], prefix='sex', drop_first=True)
df = pd.concat([df, dummy], axis=1)
df = df.drop('sex', axis=1)

# Encoding for 'chronic_disease_binary' column
dummy = pd.get_dummies(df['chronic_disease_binary'], prefix='chronic_disease_binary', drop_first=True)
df = pd.concat([df, dummy], axis=1)
df = df.drop('chronic_disease_binary', axis=1)

# Encoding for 'travel_history_binary' column
dummy = pd.get_dummies(df['travel_history_binary'], prefix='travel_history_binary', drop_first=True)
df = pd.concat([df, dummy], axis=1)
df = df.drop('travel_history_binary', axis=1)

# Encoding for 'outcome' column
dummy = pd.get_dummies(df['outcome'], prefix='outcome', drop_first=True)
df = pd.concat([df, dummy], axis=1)
df = df.drop('outcome', axis=1)

# Encoding for 'province' column
dummy = pd.get_dummies(df['province'], prefix='province', drop_first=True)
df = pd.concat([df, dummy], axis=1)
df = df.drop('province', axis=1)

# Encoding date confirmation as number of days elapsed since the start of the year
df['date_confirmation'] = pd.to_datetime(df['date_confirmation'], dayfirst=True)
df['day_count'] = (df['date_confirmation'] - datetime.datetime(2020, 1, 1)).dt.days
df = df.drop('date_confirmation', axis=1)

stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_o, test_o in stratified.split(df, df['outcome_dead']):
        for k in range (0, len(train_o)):
            train_o[k] = (df.iloc[[train_o[k]]].index[0])
        for m in range (0, len(test_o)):
            test_o[m] = (df.iloc[[test_o[m]]].index[0])
        strat_train_o = df.reindex(train_o)
        strat_test_o = df.reindex(test_o)

X_train = strat_train_o.drop('outcome_dead', axis=1)
X_test = strat_test_o.drop('outcome_dead', axis=1)
Y_train = strat_train_o['outcome_dead']
Y_test = strat_test_o['outcome_dead']


alg_labels = ['Logistic Regression', 'K-nearest neighbors', 'LinearSVC']
times = []

# Logistic Regression
start1 = timeit.default_timer()
clf = LogisticRegression(max_iter=50000000).fit(X_train, Y_train)
Y_pred1 = clf.predict(X_test)
stop1 = timeit.default_timer()

times.append(stop1 - start1)

# K-nearest neighbours
start2 = timeit.default_timer()
knn = KNeighborsClassifier().fit(X_train, Y_train)
Y_pred2 = knn.predict(X_test)
stop2 = timeit.default_timer()

times.append(stop2 - start2)

# Support Vector Machine
start3 = timeit.default_timer()
svm = LinearSVC(max_iter=50000000).fit(X_train, Y_train)
Y_pred3 = svm.predict(X_test)
stop3 = timeit.default_timer()

times.append(stop3 - start3)


# Plot Algorithm Run Times
fig, ax = plt.subplots()
y_pos = np.arange(len(alg_labels))
ax.barh(y_pos, times, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(alg_labels)
ax.invert_yaxis()
ax.set_xlabel('Seconds')
ax.set_title('Algorithm Run Times')
fig.tight_layout()
plt.show()


# Plot Overfitting/Underfitting Bar Chart
test_scores = []
test_scores.append(clf.score(X_test, Y_test))
test_scores.append(knn.score(X_test, Y_test))
test_scores.append(svm.score(X_test, Y_test))
train_scores = []
train_scores.append(clf.score(X_train, Y_train))
train_scores.append(knn.score(X_train, Y_train))
train_scores.append(svm.score(X_train, Y_train))
x = np.arange(len(alg_labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, test_scores, width, label='Test Scores')
rects2 = ax.bar(x + width/2, train_scores, width, label='Train Scores')
ax.set_ylabel('Score')
ax.set_title('Test and Train Scores to identify Overfitting/Underfitting')
ax.set_xticks(x)
ax.set_xticklabels(alg_labels)
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
fig.tight_layout()
plt.show()


# Plot Accuracy and Balanced Accuracy Bar Chart
accuracy_list = []
accuracy_list.append(accuracy_score(Y_test, Y_pred1))
accuracy_list.append(accuracy_score(Y_test, Y_pred2))
accuracy_list.append(accuracy_score(Y_test, Y_pred3))
balanced_accuracy_list = []
balanced_accuracy_list.append(balanced_accuracy_score(Y_test, Y_pred1))
balanced_accuracy_list.append(balanced_accuracy_score(Y_test, Y_pred2))
balanced_accuracy_list.append(balanced_accuracy_score(Y_test, Y_pred3))
x = np.arange(len(alg_labels))
width = 0.35
fig, ax = plt.subplots()
ax.set_ylabel('Score')
ax.set_title('Accuracy and Balanced Accuracy Scores')
ax.set_xticks(x)
ax.set_xticklabels(alg_labels)
ax.bar_label(ax.bar(x - width/2, accuracy_list, width, label='Accuracy'), padding=3)
ax.bar_label(ax.bar(x + width/2, balanced_accuracy_list, width, label='Balanced Accuracy'), padding=3)
fig.tight_layout()
plt.show()


# Plot Area under the ROC curve Graph
areas = []
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred1)
areas.append(auc(fpr, tpr))
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred2)
areas.append(auc(fpr, tpr))
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred3)
areas.append(auc(fpr, tpr))
fig, ax = plt.subplots()
y_pos = np.arange(len(alg_labels))
ax.barh(y_pos, areas, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(alg_labels)
ax.invert_yaxis()
ax.set_xlabel('Score')
ax.set_title('Area under the ROC curve')
fig.tight_layout()
plt.show()