# Predicting-the-Winner-for-FIFA-World-Cup-2018
The dataset was obtained from Kaggle. It had following three different files:
1) Matches: Contains the results of all the matches played in the world cup 
2) Players: Information regarding player name, coach name, line-up, position and goals scored 
3) World Cups: Contains information like attendance, titles won by all qualifying teams

In the first phase of this project, we cleaned the dataset for visualization. We plotted various graphs like Attendance per year, Qualified teams per year, Matches played by teams over the years, Goals scored over the years, Titles won by each team. 

The second phase of this project was building various machine learning models using scikit-learn. Following are the various models we built:
1) Logistic Regression
2) Support Vector Machine
3) K-Nearest Neighbors
4) Decision Tree
5) Random Forest
6) Bagging Classifier
7) Gradient Boosting
8) XGBoost
9) Naive Bayes

We observed 'over-fitting' on the majority of the models. The only model which had the least accuracy was Support Vector Machine (SVM). After that prediction of every match of the world cup was made using SVM. The finalist was 'England' with a winning percentage of 41.26. To avoid the problem of overfitting we then used K-Fold cross-validation. As expected all the models performed better and amongst Logistic Regression showed the least overfitting. Again prediction of each match was carried out and 'England' won with a prediction of 80.38%. Our model can become more realistic if we get stats of all the players recent performances and data of all the teams playing the qualifying rounds.
Edit project See-Food
Project name
