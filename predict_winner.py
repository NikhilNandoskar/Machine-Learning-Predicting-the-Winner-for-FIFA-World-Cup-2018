#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 21:21:31 2018

@author: nxn59
"""

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Part1: Datat Analysis

# Importing Files
data_matches = pd.read_csv('WorldCupMatches.csv')
data_players = pd.read_csv('WorldCupPlayers.csv')
data_cups = pd.read_csv('WorldCups.csv')

# EDA
print(data_matches.describe())
print(data_players.describe())
print(data_cups.describe())

# Observing total number of rows and columns in the dataset
print(data_matches.head())
print(data_matches.tail())
print(data_matches.shape)   #(4572, 20)

print(data_players.head())
print(data_players.tail())
print(data_players.shape)   #(37784, 9)

print(data_cups.head())
print(data_cups.tail())
print(data_cups.shape)     #(20, 10)

# Observing all the columns in the dataset
print(list(data_matches))
print(list(data_players))
print(list(data_cups))

# Finding Missing Values
print(data_matches.isnull().sum())
print(data_players.isnull().sum())
print(data_cups.isnull().sum())

# Handling missing data
# Cleaning data_players
x1 = np.array(data_players['Event'].isna())
x2 = (data_players['Position'].isna())   #True = 0

for i in range(len(x1)):
    if x1[i] == False:
        x2[i] = x2.replace(True, False).all()

x2_value = x2.astype(int)
x2_value = x2_value.replace([0,1], ['C','GK'])
# Replaciing Position column with x2_value in the dataset
data_players = data_players.drop('Position', axis = 1)
data_players_new = pd.concat([data_players, x2_value], axis = 1)

# Cleaning data_matches
data_matches_new = data_matches.loc[0:851,:]
data_matches_new['Attendance'] = data_matches_new['Attendance'].fillna(data_matches_new['Attendance'].value_counts().idxmax())

# Cleaning data_cups
data_cups['Attendance'] = data_cups['Attendance'].str.replace('.', '')

# Visualisation
# Analysis of Worldcup
# Year vs Attendance
plt.figure(figsize = (10,10))
sns.set_style("darkgrid")
attendance = data_cups.groupby('Year')['Attendance'].sum().reset_index()
attendance['Year'] = attendance['Year'].astype(int)
attendance['Attendance'] = attendance['Attendance'].astype(int)
sns.barplot(attendance['Year'], attendance['Attendance'],  linewidth = 2,  palette = "muted")
plt.grid(True)
plt.title("Attendence by Years", color='r', fontsize = 20, loc = 'center' )
plt.savefig('/home/nxn59/STAT_Data_Mining/fifa-world-cup/Year_Attendance')

# Year vs Qualified teams
plt.figure(figsize = (10,10))
quali = data_cups.groupby('Year')['QualifiedTeams'].sum().reset_index()
quali['Year'] = quali['Year'].astype(int)
quali['QualifiedTeams'] = quali['QualifiedTeams'].astype(int)
sns.barplot(quali['Year'], quali['QualifiedTeams'],  linewidth = 2,  palette = "Blues_d")
plt.title("Qualified Teams by Year", color='b', fontsize = 20, loc = 'center' )
plt.savefig('/home/nxn59/STAT_Data_Mining/fifa-world-cup/Year_Qualified_Teams')

# Year vs MatchesPlayed
plt.figure(figsize = (10,10))
m_p = data_cups.groupby('Year')['MatchesPlayed'].sum().reset_index()
m_p['Year'] = quali['Year'].astype(int)
m_p['MatchesPlayed'] = m_p['MatchesPlayed'].astype(int)
sns.barplot(m_p['Year'], m_p['MatchesPlayed'],  linewidth = 2,  palette = "Blues_d")
plt.title("Matches Played by Teams over the Years", color='b', fontsize = 20, loc = 'center' )
plt.savefig('/home/nxn59/STAT_Data_Mining/fifa-world-cup/Year_Matches_Played_Teams')

# Goals scored in worldcup
plt.figure(figsize = (10,10))
g_s = data_cups.groupby('Year')['GoalsScored'].sum().reset_index()
g_s['Year'] = g_s['Year'].astype(int)
g_s['GoalsScored'] = g_s['GoalsScored'].astype(int)
sns.barplot(g_s['Year'], g_s['GoalsScored'],  linewidth = 2,  palette = 'muted')
plt.title("Goals Scored over the Years", color='b', fontsize = 20, loc = 'center' )
plt.savefig('/home/nxn59/STAT_Data_Mining/fifa-world-cup/Year_Goals_Scored')


# Wining Teams
w = data_cups['Winner'].value_counts()
r_u = data_cups['Runners-Up'].value_counts()
t_p = data_cups['Third'].value_counts()
f_p = data_cups['Fourth'].value_counts()
full_time = pd.concat([w, r_u, t_p, f_p], axis = 1, sort = False)
full_time = full_time.sort_values(['Winner', 'Runners-Up', 'Third', 'Fourth'], ascending = False).fillna(0)
print(full_time)
print(full_time.sum(axis = 1))
full_time.plot(y=['Winner', 'Runners-Up', 'Third','Fourth'], kind="bar", color =['gold','silver','brown','blue'], figsize=(10, 10), width=1)
plt.title("Titles won by each Team", color='b', fontsize = 20, loc = 'center' )
plt.savefig('/home/nxn59/STAT_Data_Mining/fifa-world-cup/Wining_Teams')


# Analysis of Worldcup Matches
# Slearn Libraries
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV

def replace_country_name(df):
    if(df['Home Team Name'] in ['German DR', 'Germany FR']):
        df['Home Team Name'] = 'Germany'
    elif(df['Home Team Name'] == 'Soviet Union'):
        df['Home Team Name'] = 'Russia'
        
    if(df['Away Team Name'] in ['German DR', 'Germany FR']):
        df['Away Team Name'] = 'Germany'
    elif(df['Away Team Name'] == 'Soviet Union'):
        df['Away Team Name'] = 'Russia'   
        
    return df

data_matches_new = data_matches_new.apply(replace_country_name, axis = 1)

# Creating a list of all teams playing football worlcup:
team_name = {}
index = 0
for idx, row in data_matches_new.iterrows():
    name = row['Home Team Name']
    if (name not in team_name.keys()):
        team_name[name] = index
        index += 1
        
    name = row['Away Team Name']
    if (name not in team_name.keys()):
        team_name[name] = index
        index += 1

# Drop unwanted data from data_new_matches
data_matches_to_use = data_matches_new.drop(['Datetime', 'Stage', 'Stadium', 'City', 'Win conditions', 
                                             'Attendance', 'Half-time Home Goals', 'Half-time Away Goals', 
                                             'Referee', 'Assistant 1', 'Assistant 2',
                                             'RoundID', 'MatchID', 'Home Team Initials', 'Away Team Initials'], 1)
    
# Count champions 
winners =  data_cups['Winner'].map(lambda n: 'Germany' if n == 'Germany FR' else n).value_counts()

# Home and Away Team champions
data_matches_to_use["Home Team Champion"] = 0
data_matches_to_use["Away Team Champion"] = 0

def winners_wc(df):
    if(winners.get(df['Home Team Name']) != None):
        df['Home Team Champion'] = winners.get(df['Home Team Name'])
    if(winners.get(df['Away Team Name']) != None):
        df['Away Team Champion'] = winners.get(df['Away Team Name'])
    
    return df

data_matches_to_use = data_matches_to_use.apply(winners_wc, axis = 1)

# Winner of particular match
# For Home team win value is 1, Away team win value is 2 and for Draw value is 0
data_matches_to_use['Winner'] = '-'

def match_winner(df):
    if (int(df['Home Team Goals']) == int(df['Away Team Goals'])):
        df['Winner'] = 0
    elif(int(df['Home Team Goals']) > int(df['Away Team Goals'])):
        df['Winner'] = 1
    else:
        df['Winner'] = 2
    
    return df

data_matches_to_use = data_matches_to_use.apply(match_winner, axis = 1)

# Replace Team names by corresponding id:
def replace_name(df):
    df['Home Team Name'] = team_name[df['Home Team Name']]
    df['Away Team Name'] = team_name[df['Away Team Name']]
    
    return df

Team_id = data_matches_to_use.apply(replace_name, axis = 1)

# Now we dont need number of goals and year
Team_id = Team_id.drop(['Year','Home Team Goals', 'Away Team Goals'], 1)

# Preparing Training and Testing Data:
X = Team_id.iloc[:, 0:4].values
y = Team_id.iloc[:, -1].values

X = np.array(X, dtype = 'f')
y = np.array(y, dtype = 'f')

from sklearn.preprocessing import LabelEncoder
LabelEncoder_res=LabelEncoder()
y=LabelEncoder_res.fit_transform(y)

X, y = shuffle(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Scalling of Data:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Building ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

#seed
np.random.seed(47)
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
LR = LogisticRegression(random_state = 47)
parameters = {'penalty': ['l2'], 'C' : [ 0.1, 1, 10], 'multi_class' :['multinomial'], 'solver': ['lbfgs', 'newton-cg', 'saga']}
LR = GridSearchCV(LR, param_grid= parameters ,cv=5)
LR.fit(X_train, y_train)
score_train_acc = LR.score(X_train, y_train)
score_test_acc = LR.score(X_test, y_test)
print(score_train_acc) #62.42
print(score_test_acc)  #54.25
y_pred_LR = LR.predict(X_test)
print(classification_report(y_test, y_pred_LR))
print(confusion_matrix(y_test, y_pred_LR, labels=range(3)))

#SGDClassifier
from sklearn.linear_model import SGDClassifier
SGD = SGDClassifier(loss = 'log', penalty = 'elasticnet')
SGD.fit(X_train, y_train)
score_train_acc = SGD.score(X_train, y_train)
score_test_acc = SGD.score(X_test, y_test)
print(score_train_acc) #62.42
print(score_test_acc)  #54.25
y_pred_SGD = SGD.predict(X_test)
print(classification_report(y_test, y_pred_SGD))
print(confusion_matrix(y_test, y_pred_SGD, labels=range(3)))


# SVM
from sklearn.svm import SVC
svm_model = SVC()
#svm_model = SVC(random_state = 47, C = 0.1, kernel='sigmoid', class_weight = 'balanced', probability= True)
parameters = {'C' : [ 0.1, 0.001, 1]}#, 'multi_class' :['multinomial', 'ovr'], 'solver': ['lbfgs', 'newton-cg']}
svm_model = GridSearchCV(svm_model, param_grid= parameters ,cv=5)
svm_model.fit(X_train, y_train)
score_train_acc = svm_model.score(X_train, y_train)
score_test_acc = svm_model.score(X_test, y_test)
print(score_train_acc) #54.79
print(score_test_acc)  #51.31       
y_pred_SVM = svm_model.predict(X_test)
print(classification_report(y_test, y_pred_SVM))
print(confusion_matrix(y_test, y_pred_SVM, labels=range(3)))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier()
dc.fit(X_train, y_train)
score_train_acc = dc.score(X_train, y_train)
score_test_acc = dc.score(X_test, y_test)
print(score_train_acc) #91.78
print(score_test_acc)  #46.62
y_pred_dc = dc.predict(X_test) 
print(classification_report(y_test, y_pred_dc))
print(confusion_matrix(y_test, y_pred_dc, labels=range(3)))

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights = 'uniform', algorithm='auto')
knn.fit(X_train, y_train)
score_train_acc = knn.score(X_train, y_train)
score_test_acc = knn.score(X_test, y_test)
print(score_train_acc) #69.27
print(score_test_acc)  #52.78
y_pred_knn = knn.predict(X_test) 
print(classification_report(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn, labels=range(3)))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
score_train_acc = nb.score(X_train, y_train)
score_test_acc = nb.score(X_test, y_test)
print(score_train_acc) #62.03
print(score_test_acc)  #59.23
y_pred_nb = nb.predict(X_test) 
print(classification_report(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb, labels=range(3)))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rc = RandomForestClassifier(n_estimators=100)
rc.fit(X_train, y_train)
score_train_acc = rc.score(X_train, y_train)
score_test_acc = rc.score(X_test, y_test)
print(score_train_acc) #91.58
print(score_test_acc)  #51.31
y_pred_rc = rc.predict(X_test) 
print(classification_report(y_test, y_pred_rc))
print(confusion_matrix(y_test, y_pred_rc, labels=range(3)))

# Bagging Classifier
from sklearn.ensemble import BaggingClassifier
bc = BaggingClassifier(n_estimators=100)
bc.fit(X_train, y_train)
score_train_acc = bc.score(X_train, y_train)
score_test_acc = bc.score(X_test, y_test)
print(score_train_acc) #91.78
print(score_test_acc)  #48.09
y_pred_bc = bc.predict(X_test) 
print(classification_report(y_test, y_pred_bc))
print(confusion_matrix(y_test, y_pred_bc, labels=range(3)))

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gbc.fit(X_train, y_train)
score_train_acc = gbc.score(X_train, y_train)
score_test_acc = gbc.score(X_test, y_test)
print(score_train_acc) #58.70
print(score_test_acc)  #55.13
y_pred_gbc = gbc.predict(X_test) 
print(classification_report(y_test, y_pred_gbc))
print(confusion_matrix(y_test, y_pred_gbc, labels=range(3)))

# XGBoost
import xgboost as xgb
XGB = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=0.4, learning_rate=0.01,
       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
       n_estimators=40, n_jobs=1, nthread=None, objective='multi:softprob',
       random_state=0, reg_alpha=1e-05, reg_lambda=1, scale_pos_weight=1,
       seed=2, silent=True, subsample=0.8)
XGB.fit(X_train, y_train)
score_train_acc = XGB.score(X_train, y_train)
score_test_acc = XGB.score(X_test, y_test)
print(score_train_acc) #66.53
print(score_test_acc)  #56.59      
y_pred_XGB = XGB.predict(X_test)
print(classification_report(y_test, y_pred_XGB))
print(confusion_matrix(y_test, y_pred_XGB, labels=range(3)))




# TODO: Create the parameters list you wish to tune

parameters = { 'learning_rate' : [0.001,0.01, 0.1, 1],
               'n_estimators' : [40, 100],
               'max_depth': [3, 6],
               'min_child_weight': [1, 3],
               'gamma':[0.4],
               'subsample' : [0.5, 0.8],
               'colsample_bytree' : [0.8],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5]
             } 

clf = xgb.XGBClassifier(seed=2)

grid_obj = GridSearchCV(clf,
                        param_grid=parameters,
                        cv=5)
grid_obj = grid_obj.fit(X_train,y_train)
clf = grid_obj.best_estimator_
print(clf)


def prediction(team1, team2):
  id1 = team_name[team1]
  id2 = team_name[team2]
  championship1 = winners.get(team1) if winners.get(team1) != None else 0
  championship2 = winners.get(team2) if winners.get(team2) != None else 0

  t = np.array([id1, id2, championship1, championship2]).astype('float64')
  t = np.reshape(t, (1,-1))
  y_rc = LR.predict_proba(t)[0]

  text = ('Chance for '+team1+' to win against '+team2+' is {}\nChance for '+team2+' to win against '+team1+' is {}\nChance for '+team1+' and '+team2+' draw is {}').format(y_rc[1]*100,y_rc[2]*100,y_rc[0]*100)
  return y_rc, text

fixtures_wc = pd.read_csv('fixtures.csv')
fixtures_wc = fixtures_wc.drop(['Round Number', 'Date', 'Location', 'Result'], 1)


fix = fixtures_wc.loc[0:47, :]

#Group A
for i in range(len(fix)):
     array = (fix['Group'] == 'Group A')
     index = []
     for ar in range(len(array)):
         if array[ar] == True:
             index.append(ar)
     
for indx in range(len(index)):
    corr_row = fix.loc[index[indx]]
         
    probs, text = prediction(corr_row['Home Team'], corr_row['Away Team'])
    print('Results \n', text)
#GROUP B    
for i in range(len(fix)):
     array = (fix['Group'] == 'Group B')
     index = []
     for ar in range(len(array)):
         if array[ar] == True:
             index.append(ar)
     
for indx in range(len(index)):
    corr_row = fix.loc[index[indx]]
         
    probs, text = prediction(corr_row['Home Team'], corr_row['Away Team'])
    print('Results \n', text)    
#Group C        
for i in range(len(fix)):
     array = (fix['Group'] == 'Group C')
     index = []
     for ar in range(len(array)):
         if array[ar] == True:
             index.append(ar)
     
for indx in range(len(index)):
    corr_row = fix.loc[index[indx]]
         
    probs, text = prediction(corr_row['Home Team'], corr_row['Away Team'])
    print('Results \n', text)    
    
#Group D        
for i in range(len(fix)):
     array = (fix['Group'] == 'Group D')
     index = []
     for ar in range(len(array)):
         if array[ar] == True:
             index.append(ar)
     
for indx in range(len(index)):
    corr_row = fix.loc[index[indx]]
         
    probs, text = prediction(corr_row['Home Team'], corr_row['Away Team'])
    print('Results \n', text)  

#Group E        
for i in range(len(fix)):
     array = (fix['Group'] == 'Group E')
     index = []
     for ar in range(len(array)):
         if array[ar] == True:
             index.append(ar)
     
for indx in range(len(index)):
    corr_row = fix.loc[index[indx]]
         
    probs, text = prediction(corr_row['Home Team'], corr_row['Away Team'])
    print('Results \n', text) 

#Group F        
for i in range(len(fix)):
     array = (fix['Group'] == 'Group F')
     index = []
     for ar in range(len(array)):
         if array[ar] == True:
             index.append(ar)
     
for indx in range(len(index)):
    corr_row = fix.loc[index[indx]]
         
    probs, text = prediction(corr_row['Home Team'], corr_row['Away Team'])
    print('Results \n', text) 

#Group G        
for i in range(len(fix)):
     array = (fix['Group'] == 'Group G')
     index = []
     for ar in range(len(array)):
         if array[ar] == True:
             index.append(ar)
     
for indx in range(len(index)):
    corr_row = fix.loc[index[indx]]
         
    probs, text = prediction(corr_row['Home Team'], corr_row['Away Team'])
    print('Results \n', text)

#Group H        
for i in range(len(fix)):
     array = (fix['Group'] == 'Group H')
     index = []
     for ar in range(len(array)):
         if array[ar] == True:
             index.append(ar)
     
for indx in range(len(index)):
    corr_row = fix.loc[index[indx]]
         
    probs, text = prediction(corr_row['Home Team'], corr_row['Away Team'])
    print('Results \n', text)    

# Round of 16
round_16 = pd.read_csv('team_name_16.csv')
qualified_teams = np.array(round_16).astype(str)


team_name_16 = {}
index = 0
for idx, row in round_16.iterrows():
    name = row['Winners']
    if (name not in team_name_16.keys()):
        team_name_16[name] = index
        index += 1
        
    name = row['Runners Up']
    if (name not in team_name_16.keys()):
        team_name_16[name] = index
        index += 1

def prediction_16(df):
  team1 = df[0]
  team2 = df[1]
  id1 = team_name_16[team1]
  id2 = team_name_16[team2]
  championship1 = winners.get(team1) if winners.get(team1) != None else 0
  championship2 = winners.get(team2) if winners.get(team2) != None else 0

  t = np.array([id1, id2, championship1, championship2]).astype('float64')
  t = np.reshape(t, (1,-1))
  y_xgb = LR.predict_proba(t)[0]

  text = ('Chance for '+team1+' to win against '+team2+' is {}\nChance for '+team2+' to win against '+team1+' is {}\nChance for '+team1+' and '+team2+' draw is {}').format(y_xgb[1]*100,y_xgb[2]*100,y_xgb[0]*100)
  return y_xgb, text

for j in range(len(qualified_teams)):
    probs, text = prediction_16(qualified_teams[j])
    print(text)
    
# Quater_Finals
quater_finals = pd.read_csv('Quater_finals.csv')
quater_round_teams = np.array(quater_finals).astype(str)

team_name_8 = {}
index = 0
for idx, row in quater_finals.iterrows():
    name = row['Quarter 1']
    if (name not in team_name_8.keys()):
        team_name_8[name] = index
        index += 1
        
    name = row['Quarter 2']
    if (name not in team_name_8.keys()):
        team_name_8[name] = index
        index += 1

def prediction_8(df):
  team1 = df[0]
  team2 = df[1]
  id1 = team_name_8[team1]
  id2 = team_name_8[team2]
  championship1 = winners.get(team1) if winners.get(team1) != None else 0
  championship2 = winners.get(team2) if winners.get(team2) != None else 0

  t = np.array([id1, id2, championship1, championship2]).astype('float64')
  t = np.reshape(t, (1,-1))
  y_xgb = LR.predict_proba(t)[0]

  text = ('Chance for '+team1+' to win against '+team2+' is {}\nChance for '+team2+' to win against '+team1+' is {}\nChance for '+team1+' and '+team2+' draw is {}').format(y_xgb[1]*100,y_xgb[2]*100,y_xgb[0]*100)
  return y_xgb, text
    
for q in range(len(quater_round_teams)):
    probs, text = prediction_8(quater_round_teams[q])
    print(text)                    
    
# Semi Finals
semi_finals = pd.read_csv('Semi_finals.csv')
semi_finals_teams = np.array(semi_finals).astype(str)

team_name_4 = {}
index = 0
for idx, row in semi_finals.iterrows():
    name = row['Sem1']
    if (name not in team_name_4.keys()):
        team_name_4[name] = index
        index += 1
        
    name = row['Sem2']
    if (name not in team_name_4.keys()):
        team_name_4[name] = index
        index += 1

def prediction_4(df):
  team1 = df[0]
  team2 = df[1]
  id1 = team_name_4[team1]
  id2 = team_name_4[team2]
  championship1 = winners.get(team1) if winners.get(team1) != None else 0
  championship2 = winners.get(team2) if winners.get(team2) != None else 0

  t = np.array([id1, id2, championship1, championship2]).astype('float64')
  t = np.reshape(t, (1,-1))
  y_xgb = LR.predict_proba(t)[0]

  text = ('Chance for '+team1+' to win against '+team2+' is {}\nChance for '+team2+' to win against '+team1+' is {}\nChance for '+team1+' and '+team2+' draw is {}').format(y_xgb[1]*100,y_xgb[2]*100,y_xgb[0]*100)
  return y_xgb, text

for s in range(len(semi_finals_teams)):
    probs, text = prediction_4(semi_finals_teams[s])
    print(text) 

# Third Place:
probs, text = prediction('Senegal', 'Denmark')
print(text)

# Finals:
probs, text = prediction('Argentina', 'England')
print(text)

##############################################################################
# K-Fold Cross Vaidation

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle = True, random_state=47)
kf.get_n_splits(X)
print(kf)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=None)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)