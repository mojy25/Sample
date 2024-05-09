"""
Step 7-8
Team win predictions from given performance
"""

# 1. Import Grades
# 2. Feature Engineering
    # 2A. Bullpen as 1 feature - weighted average of bullpen grades with playing time
    # 2B. Divisional Relationships
# 3. Make Models with LOGO
# 4. Predict Win Probabilities
# 5. Find Best Fit model
# 6. Create best Ensemble model from best performing models of KNN, CNN, and XGBoost
# 7. Move on to predictive grades as features and projecting player performance/predictors and team win evaluations from there

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import neighbors, ensemble, linear_model, metrics, model_selection, preprocessing
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import keras

class Data():
    """
    Contains individual player grades, team records, and fixed-length team grades
    """

    def __init__(self):
        self.grades = pd.read_csv(r" Redacted ")
    # Drop Players from grades that have Team - - -
        self.grades = self.grades[self.grades['Team'] != '- - -']
        self.teams = pd.read_csv(r" Redacted ", index_col=['Season', 'Team'])
        self.MLColumns = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'SP1', 'SP2', 'SP3', 'SP4', 'Bullpen', 'HCum', 'SPCum', 'BullCum', 'HPT', 'SPPT', 'BullPT', 'Division']
        self.MLgrades = pd.read_csv(r" Redacted ", index_col=['Season', 'Team'])
        self.separateRelievers = 1

    def equalLengthTeams(self):
        """
        Each team should have 9 Position Players, 4 Starting Pitchers, and a weighted average of their bullpen
        0-pad teams with not enough Position Players or Starters, in the event of 0 bullpen pitchers, 0-pad that as well. 
        Keep sums of Cumulative Experience for all position groups
        """
        hitMedian = 9
        spMedian = 4

    # Correct number of Hitters   
        for (season, team), group in self.grades.groupby(['Season', 'Team']):
        # Number of Players is the rowcount of the group shape
            group = group[group['POS'].isin(['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH'])]
            sortedGroup = group.sort_values(by='Grade', ascending=False)
            sortedGroup.reset_index(drop=True, inplace=True)
            self.MLgrades.loc[(season, team), 'HCum'] = group['Cumulative Experience'].sum()
            self.MLgrades.loc[(season, team), 'HPT'] = group['Playing Time'].sum()
            num = group.shape[0]
        # Group needs to be 0-Padded:
            if num < hitMedian:
                count = 1
                for i in range(len(sortedGroup)):
                    self.MLgrades.loc[(season, team), f'H{count}'] = sortedGroup.loc[i, 'Grade']
                    count += 1
                for i in range(num, hitMedian):
                    self.MLgrades.loc[(season, team), f'H{count}'] = 0
                    count += 1
        # Group needs to be trimmed:
            elif num >= hitMedian:
                count = 1
                for i in range(hitMedian):
                    self.MLgrades.loc[(season, team), f'H{count}'] = sortedGroup.loc[i, 'Grade']
                    count += 1

    ## Correct number of Starting Pitchers
        for (season, team), group in self.grades.groupby(['Season', 'Team']):
            group = group[group['POS'] == 'SP']
            sortedGroup = group.sort_values(by='Grade', ascending=False)
            sortedGroup.reset_index(drop=True, inplace=True)
            self.MLgrades.loc[(season, team), 'SPCum'] = group['Cumulative Experience'].sum()
            self.MLgrades.loc[(season, team), 'SPPT'] = group['Playing Time'].sum()
            num = group.shape[0]
            if num < spMedian:
                count = 1
                for i in range(len(sortedGroup)):
                    self.MLgrades.loc[(season, team), f'SP{count}'] = sortedGroup.loc[i, 'Grade']
                    count += 1
                for i in range(num, spMedian):
                    self.MLgrades.loc[(season, team), f'SP{count}'] = 0
                    count += 1
            elif num >= spMedian:
                count = 1
                for i in range(spMedian):
                    self.MLgrades.loc[(season, team), f'SP{count}'] = sortedGroup.loc[i, 'Grade']
                    count += 1
## Create Bullpen Feature
        for (season, team), group in self.grades.groupby(['Season', 'Team']):
            group = group[group['POS'] == 'RP']
            self.MLgrades.loc[(season, team), 'BullCum'] = group['Cumulative Experience'].sum()
            self.MLgrades.loc[(season, team), 'BullPT'] = group['Playing Time'].sum()
            # If not separating the relievers, create aggregate metric for Bullpen
            self.MLgrades.loc[(season, team), 'Bullpen'] = round(np.dot(group['Grade'] / 100, group['Playing Time']) / group['Playing Time'].sum(), 3)
    # If separating the Relievers, include 
            
        self.MLgrades.to_csv(r" Redacted ")

        print(self.MLgrades.describe())


    def categoricalFeatures(self):
        """
        Include:
        Previous Season's Winning Percentage
        Number of New Players
        Difference in Cumulative Experience
        Difference in Average Grade of New Players vs. Lost Players
        Previous Season - Division Inter-Divisional Strength
        Previous Season - Division Intra-Divisional Strength
        """
    # Drop 2020 and Sort
        self.teams = self.teams[self.teams.index.get_level_values(0) != 2020]
        self.MLgrades = self.MLgrades[self.MLgrades.index.get_level_values(0) != 2020]
        self.MLgrades.sort_index(level=[0, 1], ascending=True, inplace=True)
        self.teams.sort_index(level=[0, 1], ascending=True, inplace=True)
        self.players = pd.read_csv(r" Redacted ", index_col=['Season', 'Team'])
        self.players.drop(self.players.index.get_level_values(1) == '- - -', inplace=True)
        self.players = self.players.sort_index(level=[0, 1], ascending=True)

    # Need to manually create Division feature
        self.teams['Division'] = 0
        for (season, team), group in self.teams.groupby(['Season', 'Team']):
            if season <= 1993:
                if team in ['BOS', 'NYY', 'BAL', 'TOR', 'MIL', 'CLE', 'DET']:
                    self.teams.loc[(season, team), 'Division'] = 1
                elif team in ['KCR', 'MIN', 'CHW', 'TEX', 'OAK', 'SEA', 'CAL']:
                    self.teams.loc[(season, team), 'Division'] = 2
                elif team in ['CHC', 'FLA', 'MON', 'NYM', 'PHI', 'PIT', 'STL', 'WSN']:
                    self.teams.loc[(season, team), 'Division'] = 3
                elif team in ['LAD', 'SFG', 'SDP', 'COL', 'ATL', 'HOU', 'CIN']:
                    self.teams.loc[(season, team), 'Division'] = 4
                else:
                    self.teams.loc[(season, team), 'Division'] = 0
                    print(season, team)
            elif season <= 1997:
                if team in ['BAL', 'BOS', 'DET', 'NYY', 'TOR']:
                    self.teams.loc[(season, team), 'Division'] = 1
                elif team in ['CHW', 'CLE', 'KCR', 'MIN', 'MIL']:
                    self.teams.loc[(season, team), 'Division'] = 2
                elif team in ['CAL', 'OAK', 'SEA', 'TEX', 'ANA']:
                    self.teams.loc[(season, team), 'Division'] = 3
                elif team in ['ATL', 'FLA', 'MON', 'NYM', 'PHI', 'MIA', 'WSN']:
                    self.teams.loc[(season, team), 'Division'] = 4
                elif team in ['CHC', 'CIN', 'HOU', 'PIT', 'STL']:
                    self.teams.loc[(season, team), 'Division'] = 5
                elif team in ['COL', 'LAD', 'SDP', 'SFG']:
                    self.teams.loc[(season, team), 'Division'] = 6
                else:
                    self.teams.loc[(season, team), 'Division'] = 0
                    print(season, team)
            elif season <= 2012:
                if team in ['BAL', 'BOS', 'NYY', 'TBR', 'TOR', 'TBD']:
                    self.teams.loc[(season, team), 'Division'] = 1
                elif team in ['CHW', 'CLE', 'DET', 'KCR', 'MIN']:
                    self.teams.loc[(season, team), 'Division'] = 2
                elif team in ['ANA', 'OAK', 'SEA', 'TEX', 'LAA']:
                    self.teams.loc[(season, team), 'Division'] = 3
                elif team in ['ATL', 'FLA', 'NYM', 'PHI', 'WSN', 'MON', 'MIA']:
                    self.teams.loc[(season, team), 'Division'] = 4
                elif team in ['CHC', 'CIN', 'MIL', 'PIT', 'STL', 'HOU']:
                    self.teams.loc[(season, team), 'Division'] = 5
                elif team in ['ARI', 'COL', 'LAD', 'SDP', 'SFG']:
                    self.teams.loc[(season, team), 'Division'] = 6
                else:
                    self.teams.loc[(season, team), 'Division'] = 0
                    print(season, team)
            else:
                if team in ['BAL', 'BOS', 'NYY', 'TBR', 'TOR']:
                    self.teams.loc[(season, team), 'Division'] = 1
                elif team in ['CHW', 'CLE', 'DET', 'KCR', 'MIN']:
                    self.teams.loc[(season, team), 'Division'] = 2
                elif team in ['HOU', 'OAK', 'SEA', 'TEX', 'LAA']:
                    self.teams.loc[(season, team), 'Division'] = 3
                elif team in ['ATL', 'NYM', 'PHI', 'WSN', 'MIA']:
                    self.teams.loc[(season, team), 'Division'] = 4
                elif team in ['CHC', 'CIN', 'MIL', 'PIT', 'STL']:
                    self.teams.loc[(season, team), 'Division'] = 5
                elif team in ['ARI', 'COL', 'LAD', 'SDP', 'SFG']:
                    self.teams.loc[(season, team), 'Division'] = 6
                else:
                    self.teams.loc[(season, team), 'Division'] = 0
                    print(season, team)
    
    # Create New Columns
        self.MLgrades['New Players'] = 0
        self.MLgrades['Diff Cumulative Experience'] = 0
        self.MLgrades['Diff Grade'] = 0
        self.MLgrades['Inter-Divisional Strength'] = 0
        self.MLgrades['Intra-Divisional Strength'] = 0
        self.MLgrades['Prev Win %'] = 0

    # Create Previous Season Winning Percentage
        for (season, team), group in self.MLgrades.groupby(['Season', 'Team']):
            tempGroup = group.copy(deep=True)
    # Find the previous season for the same team
            if (season - 1, team) in self.MLgrades.index:
                self.MLgrades.loc[(season, team), 'Prev Win %'] = self.teams.loc[(season - 1, team), '%']
                prevGroup = self.MLgrades.loc[(season - 1, team)].copy(deep=True)
                self.MLgrades.loc[(season, team), 'Diff Cumulative Experience'] = round((tempGroup['HCum'].sum() + tempGroup['SPCum'].sum() + tempGroup['BullCum'].sum()) - (prevGroup['HCum'].sum() + prevGroup['SPCum'].sum() + prevGroup['BullCum'].sum()), 3)
        
        # Create Divisional Strength
                self.MLgrades.loc[(season, team), 'Inter-Divisional Strength'] = round(self.teams.loc[(self.teams.index.get_level_values(0) == season - 1) & (self.teams['Division'] == self.teams.loc[(season, team), 'Division']), '%'].mean(), 3)
                self.MLgrades.loc[(season, team), 'Intra-Divisional Strength'] = round(self.MLgrades.loc[(season, team), 'Prev Win %'] - self.MLgrades.loc[(season, team), 'Inter-Divisional Strength'], 3)

        # Create player groups
                oldPlayerGroup = self.players.loc[(season - 1, team)].copy(deep=True)
                newPlayerGroup = self.players.loc[(season, team)].copy(deep=True)
        # Create Difference in Average Grade of New Players vs. Lost Players
                for player in oldPlayerGroup['PlayerID']:
                    for player2 in newPlayerGroup['PlayerID']:
                        if player == player2:
                            newPlayerGroup = newPlayerGroup[newPlayerGroup['PlayerID'] != player]   
                            oldPlayerGroup = oldPlayerGroup[oldPlayerGroup['PlayerID'] != player]
                            break
        # New Players is the length of the tempGroup
                self.MLgrades.loc[(season, team), 'New Players'] = newPlayerGroup.shape[0]
                if newPlayerGroup.shape[0] > 0 and oldPlayerGroup.shape[0] > 0:
                    self.MLgrades.loc[(season, team), 'Diff Grade'] = round(newPlayerGroup['Grade'].mean() - oldPlayerGroup['Grade'].mean(), 3)
                else:
                    self.MLgrades.loc[(season, team), 'Diff Grade'] = 0


    # If it doesn't exist (expansion teams and 2021), assign the mean
            else:
                self.MLgrades.loc[(season, team), 'Prev Win %'] = .5
                self.MLgrades.loc[(season, team), 'Diff Cumulative Experience'] = 0
                self.MLgrades.loc[(season, team), 'New Players'] = 0
                self.MLgrades.loc[(season, team), 'Diff Grade'] = 0
                self.MLgrades.loc[(season, team), 'Inter-Divisional Strength'] = .5
                self.MLgrades.loc[(season, team), 'Intra-Divisional Strength'] = 0

        self.MLgrades.to_csv(r" Redacted ")


class Models():
    """
    KNN
    CNN
    XGBoost
    Linear Regression
    Ensemble
    Aggregate
    """
    
    def __init__(self):
        """
        Initialize models and train/test data
        x-y train/test and groups are used for storing data 
        knnModel, xgbModel, linearModel, cnnModel are used for storing models
        NOTE: Saving aggregate testing for later, interested in seeing the difference betweent he individual algorithm results right now
        ensembleModel will be experimented with along with other aggregation methods to test if there's a way to make a more productive model out of these algorithms
        preds Holds the predictions for each model
        """
        self.xTrain, self.xTest, self.yTrain, self.yTest = None, None, None, None
        self.xTrainNorm, self.xTestNorm = None, None
        self.knnModel = neighbors.KNeighborsRegressor()
        self.xgbModel = xgb.XGBRegressor()
        self.linearModel = linear_model.LinearRegression()
        self.cnnModel = keras.models.Sequential()
        # self.cnnModel.add(keras.layers.Input(shape=(self.xTrainNorm.shape[1],)))
        self.cnnModel.add(keras.layers.Dense(1024, activation='sigmoid'))
        self.cnnModel.add(keras.layers.Dense(1024, activation='tanh'))
        self.cnnModel.add(keras.layers.Dense(512, activation='sigmoid'))
        self.cnnModel.add(keras.layers.Dense(512, activation='tanh'))
        self.cnnModel.add(keras.layers.Dense(256, activation='sigmoid'))
        self.cnnModel.add(keras.layers.Dense(256, activation='tanh'))
        self.cnnModel.add(keras.layers.Dense(16, activation='sigmoid'))
        self.cnnModel.add(keras.layers.Dense(1, activation='tanh'))
        self.ensembleModel = ensemble.VotingRegressor(estimators=[('knn', self.knnModel), ('xgb', self.xgbModel), ('linear', self.linearModel), ('cnn', self.cnnModel)])
        self.groups = None
        self.knnPreds, self.xgbPreds, self.linearPreds, self.cnnPreds, self.ensemblePreds, self.truth = [], [], [], [], [], []
        self.aggregatePreds, self.votingPreds = [], []
        self.teams = pd.read_csv(r" Redacted ", index_col=['Season', 'Team'])
        self.MLgrades = pd.read_csv(r" Redacted ", index_col=['Season', 'Team'])

    def LOGO(self):
        """
        Leave One Group Out Validation
        One Season will be tested at a time, and the data will be postprocessed so that the mean of the league wide predictions is 50%
        """
    # Set up LOGO Loop
        logo = model_selection.LeaveOneGroupOut()
        # self.MLgrades = self.MLgrades[self.MLgrades.index.get_level_values(0) >= 1998]
        # self.teams = self.teams[self.teams.index.get_level_values(0) >= 1998]
        self.MLgrades = self.MLgrades[self.MLgrades.index.get_level_values(0) != 2020]
        self.teams = self.teams[self.teams.index.get_level_values(0) != 2020]
    
    # Sort indices
        self.MLgrades.sort_index(level=[0, 1], ascending=True, inplace=True)
        self.teams.sort_index(level=[0, 1], ascending=True, inplace=True)

    # Check that data matches:
        print(self.MLgrades.head(), self.teams.head())

    # Groups are Seasons for LOGO
        self.groups = self.MLgrades.index.get_level_values('Season')
    # 1 NA for Bullpen (0 qualified relievers), 9 NAs for Inter/Intra Divisional Strength (1994) - caused by division expansions from 4 to 6, filled with .5 and 0
        self.MLgrades['Inter-Divisional Strength'].fillna(.5, inplace=True)
        self.MLgrades.fillna(0, inplace=True)

        for trainIndex, testIndex in logo.split(self.MLgrades, self.teams['%'], self.groups):
    # Copy all Features over
            self.xTrain, self.xTest = self.MLgrades.iloc[trainIndex].copy(deep=True), self.MLgrades.iloc[testIndex].copy(deep=True)
    # Normalize Data - using fit_transform for training data
            scaleLoop = preprocessing.MinMaxScaler()
    # initialize the normalized dataframes
            self.xTrainNorm, self.xTestNorm = self.xTrain.copy(deep=True), self.xTest.copy(deep=True)
            self.xTrainNorm.loc[:, ['HCum', 'SPCum', 'BullCum', 'HPT', 'SPPT', 'BullPT', 'New Players', 'Diff Cumulative Experience', 'Diff Grade', 'Inter-Divisional Strength', 'Intra-Divisional Strength', 'Prev Win %']] = scaleLoop.fit_transform(self.xTrainNorm.loc[:, ['HCum', 'SPCum', 'BullCum', 'HPT', 'SPPT', 'BullPT', 'New Players', 'Diff Cumulative Experience', 'Diff Grade', 'Inter-Divisional Strength', 'Intra-Divisional Strength', 'Prev Win %']])
    # Normalize Test Data - Don't fit just transform to prevent data leakage
            self.xTestNorm.loc[:, ['HCum', 'SPCum', 'BullCum', 'HPT', 'SPPT', 'BullPT', 'New Players', 'Diff Cumulative Experience', 'Diff Grade', 'Inter-Divisional Strength', 'Intra-Divisional Strength', 'Prev Win %']] = scaleLoop.transform(self.xTestNorm.loc[:, ['HCum', 'SPCum', 'BullCum', 'HPT', 'SPPT', 'BullPT', 'New Players', 'Diff Cumulative Experience', 'Diff Grade', 'Inter-Divisional Strength', 'Intra-Divisional Strength', 'Prev Win %']])
    # loc not needed since labels are a Series object
            self.yTrain, self.yTest = self.teams.iloc[trainIndex]['%'], self.teams.iloc[testIndex]['%']

    # Drop other features to see if they're harming model
            self.xTrain.drop(['New Players', 'Intra-Divisional Strength', 'Prev Win %'], axis=1, inplace=True)
            self.xTest.drop(['New Players', 'Intra-Divisional Strength', 'Prev Win %'], axis=1, inplace=True)
            self.xTrainNorm.drop(['New Players', 'Intra-Divisional Strength', 'Prev Win %'], axis=1, inplace=True)
            self.xTestNorm.drop(['New Players', 'Intra-Divisional Strength', 'Prev Win %'], axis=1, inplace=True)
    # Fit Models - Calling appropriate functions - Need to append predictions one at a time
            tempKNN = self.KNN()
            tempXGB = self.XGBoost()
            tempLinear = self.Linear()
            tempCNN = self.CNN()
            tempEn = self.votingMod()
    # Append Predictions
            for i in range(len(testIndex)):
                self.knnPreds.append(tempKNN[i])
                self.xgbPreds.append(tempXGB[i])
                self.linearPreds.append(tempLinear[i])
                self.cnnPreds.append(tempCNN[i])
                self.truth.append(self.yTest.iloc[i])
                self.aggregatePreds.append((tempKNN[i] + tempXGB[i] + tempLinear[i] + tempCNN[i]) / 4)
                self.ensemblePreds.append(tempEn[i])

            print('Season:', self.groups[testIndex[0]])

    # Display Metrics
        print('KNN Metrics:')
        self.printMetrics(self.truth, self.knnPreds)
        print('XGBoost Metrics:')
        self.printMetrics(self.truth, self.xgbPreds)
        print('Linear Metrics:')
        self.printMetrics(self.truth, self.linearPreds)
        print('CNN Metrics:')
        self.printMetrics(self.truth, self.cnnPreds)
        print('Ensemble Metrics:')
        self.printMetrics(self.truth, self.ensemblePreds)
        print('Aggregate Metrics:')
        self.printMetrics(self.truth, self.aggregatePreds)
    
    # Plot Results
        plt.scatter(self.truth, self.knnPreds, label='KNN', color='blue')
        plt.scatter(self.truth, self.xgbPreds, label='XGBoost', color='pink')
        plt.scatter(self.truth, self.linearPreds, label='Linear', color='yellow')
        plt.plot(self.truth, self.cnnPreds, label='CNN', color='green')
        plt.show()

    # Plotly Scatter Plot
        from plotly import graph_objects as go
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=self.truth, y=self.knnPreds, mode='markers', name='KNN Predictions'))
        fig.add_trace(go.Scatter(x=self.truth, y=self.xgbPreds, mode='markers', name='XGBoost Predictions'))
        fig.add_trace(go.Scatter(x=self.truth, y=self.linearPreds, mode='markers', name='Linear Predictions'))
        fig.add_trace(go.Scatter(x=self.truth, y=self.cnnPreds, mode='markers', name='CNN Predictions'))
        fig.add_trace(go.Scatter(x=self.truth, y=self.ensemblePreds, mode='markers', name='Ensemble Predictions'))
        fig.add_trace(go.Scatter(x=self.truth, y=self.aggregatePreds, mode='markers', name='Aggregate Predictions'))

        fig.show()
            
    
    def KNN(self):
        """
        KNN Model - Keep Notes of the Best hyperparameters for the model
        n-neighbors - only about 1200 total teams so neighbors can be large but shouldn't be absurd
        leaf size - Learn more about how this affects the model
        distance metric - Minkowski worked better than Euclidean for finding similar players based on reduced variance findings
        algorithm - experiment with all, could set to auto for grid-search to prevent algorithm not pairing with hyperparameters well
        -----------------------------------
        KNN Requires normalized data
        """
        self.knnModel = neighbors.KNeighborsRegressor(n_neighbors=10, leaf_size=5, p=1, algorithm='auto')
        self.knnModel.fit(self.xTrainNorm, self.yTrain)
    # Predictions need to be postprocessed to make the mean equal to .500
        preds = self.knnModel.predict(self.xTestNorm)
        preds = preds + (0.5 - preds.mean())
        return preds

    def XGBoost(self):
        """
        XGBoost Regressor - Keep Notes of the Best hyperparameters for the model
        n_estimators - range from small to large
        learning_rate - range from small to large
        max_depth - range from small to large
        scale_pos_weight - range from small to large
        booster - NOT including gblinear at first, will see how the nonlinear regression performs vs. the linear regression model first.
        Linear typically isn't a bad fit fo the baseball stats and a lot of the nonlinearities could be handled in the preprocessing
        -----------------------------------
        XGBoost does not require normalized data
        """
        self.xgbModel = xgb.XGBRegressor(n_estimators=100, learning_rate=.1, gamma=0, min_child_weight=25, booster='gbtree')
    # Create validation set
        xTrain, xVal, yTrain, yVal = model_selection.train_test_split(self.xTrain, self.yTrain, test_size=0.1, shuffle=True)    
        self.xgbModel.fit(xTrain, yTrain, eval_set=[(xVal, yVal)], verbose=False)

    # Predictions need to be postprocessed to make the mean equal to .500
        preds = self.xgbModel.predict(self.xTest)
        preds = preds + (0.5 - preds.mean())
        return preds
    
    def Linear(self):
        """
        Linear Model - Pretty Simple here, no need for a grid mod, just fit the model and predict
        ----------------------------------- 
        Linear Regression does not require normalized data
        """
        self.linearModel = linear_model.LinearRegression()
        self.linearModel.fit(self.xTrain, self.yTrain)
        preds = self.linearModel.predict(self.xTest)
        preds = preds + (0.5 - preds.mean())
        return preds
    
    def CNN(self):
        """
        Keras Sequential Model - Does not work with the SKLearn GridSearchCV, will do manually using random.choice()
        Amount of Layers - range from 1 to 9
        Amount of Nodes - range from 16 to 200
        Activation Functions - range from relu, elu, selu, sigmoid
        Optimizers - range from adam, sgd, rmsprop
        Loss Function - MAE
        Dropout Rate - range from 0.05 to 0.5
        -------------------------------------
        CNN Requires normalized data
        Use Validation Split for Training
        """
    # Restart Model each time
        optimizer = keras.optimizers.Adam(learning_rate=0.00025)
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
        self.cnnModel.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_absolute_error'])
        history = self.cnnModel.fit(self.xTrainNorm, self.yTrain, validation_split=0.125, shuffle=True, epochs=100, batch_size=5, callbacks=callbacks)

    # Predict
        preds = self.cnnModel.predict(self.xTestNorm)
        preds = preds + (0.5 - preds.mean())
        return preds
    
    def createCNNMod(self):
        """
        This function is to use the Sequential Model to create a NN Model
        """
        mod = keras.models.Sequential()
        # self.cnnModel.add(keras.layers.Input(shape=(self.xTrainNorm.shape[1],)))
        mod.add(keras.layers.Dense(1024, activation='sigmoid'))
        mod.add(keras.layers.Dense(1024, activation='tanh'))
        mod.add(keras.layers.Dense(512, activation='sigmoid'))
        mod.add(keras.layers.Dense(512, activation='tanh'))
        mod.add(keras.layers.Dense(256, activation='sigmoid'))
        mod.add(keras.layers.Dense(256, activation='tanh'))
        mod.add(keras.layers.Dense(16, activation='sigmoid'))
        mod.add(keras.layers.Dense(1, activation='tanh'))
        return mod
    
    def votingMod(self):
        """
        Testing Voting Regressor with:
        KNN
        XGBoost
        Linear
        Sequential NN
        and Comparing to a weighted average of the individual predictions
        """
        kerasModel = self.createCNNMod()
        keras_regressor = KerasRegressorWrapper(kerasModel)
        self.ensembleModel = ensemble.VotingRegressor(estimators=[('knn', self.knnModel), ('xgb', self.xgbModel), ('linear', self.linearModel), ('cnn', keras_regressor)])
        preds = self.ensembleModel.fit(self.xTrainNorm, self.yTrain).predict(self.xTestNorm)
        preds = preds + (0.5 - np.mean(preds))
        return preds
            
    def printMetrics(self, truth, preds):
        """
        Print Any Model Metrics
        """
        print('\nR^2:\t\t', round(metrics.r2_score(truth, preds), 3))
        print('MAE:\t\t', round(metrics.mean_absolute_error(truth, preds), 3))
        print('Median Absolute Error:\t', round(metrics.median_absolute_error(truth, preds), 3))
        print('Max Error:\t\t', round(metrics.max_error(truth, preds), 3), '\n\n')

    def gridMods(self):
        """
        Randomized Grid-Search CV for XGBoost and KNN to optimize hyperparameters
        Play around with CNN architecture
        """

    # Make KNN Grid mod
        paramKNN = {
            'n_neighbors': range(5, 50, 5),
            'leaf_size': range(5, 50, 5),
            'p': [1, 2, 3, 4, 5],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        gridKNN = model_selection.RandomizedSearchCV(neighbors.KNeighborsRegressor(), param_distributions=paramKNN, n_iter=10, cv=5, verbose=0, n_jobs=-1, scoring='r2')
        gridKNN.fit(self.xTrainNorm, self.yTrain)
        print('KNN Best Params:', gridKNN.best_params_)
        print('KNN Best Score:', gridKNN.best_score_)

    # Make XGBoost Grid mod
        paramXGB = {
            'n_estimators': range(100, 1000, 50),
            'learning_rate': [0.01, .05, .1, .25, .5, .75, 1],
            'scale_pos_weight': [2, 2.5, 3, 3.5, 4, 4.5],
            'booster': ['gbtree', 'dart']
        }
        gridXGB = model_selection.RandomizedSearchCV(xgb.XGBRegressor(), param_distributions=paramXGB, n_iter=25, cv=5, verbose=0, n_jobs=-1, scoring='r2')
        gridXGB.fit(self.xTrain, self.yTrain)
        print('XGBoost Best Params:', gridXGB.best_params_)
        print('XGBoost Best Score:', gridXGB.best_score_)
    # Plot feature importance of best model
        xgb.plot_importance(gridXGB.best_estimator_)
        plt.show()


    # Make CNN Grid mod - I don't think this will work
        mod = keras.models.Sequential()
        mod.add(keras.layers.Input(shape=(self.xTrainNorm.shape[1],)))
        mod.add(keras.layers.Dense(1024, activation='sigmoid'))
        mod.add(keras.layers.Dense(512, activation='tanh'))
        mod.add(keras.layers.Dense(512, activation='sigmoid'))
        mod.add(keras.layers.Dense(512, activation='tanh'))
        mod.add(keras.layers.Dense(256, activation='sigmoid'))
        mod.add(keras.layers.Dense(256, activation='tanh'))
        mod.add(keras.layers.Dense(64, activation='sigmoid'))
        mod.add(keras.layers.Dense(1, activation='tanh'))

        optimizer = keras.optimizers.Adam(learning_rate=0.00025)
        callbackES = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
        mod.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_absolute_error'])
        history = mod.fit(self.xTrainNorm, self.yTrain, validation_split=0.125, shuffle=True, epochs=100, batch_size=5, callbacks=callbackES)

        self.printMetrics(self.yTest, mod.predict(self.xTestNorm))


class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model
    def fit(self, X, y):
        optimizer = keras.optimizers.Adam(learning_rate=0.00025)
        callbacksn = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
        self.model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_absolute_error'])
        self.model.fit(X, y, validation_split=0.125, shuffle=True, epochs=100, batch_size=5, callbacks=callbacksn)
        return self
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions.flatten()
    def get_params(self, deep=True):
        return {"model": self.model}
    def set_params(self, **params):
        if 'model' in params:
            self.model = params['model']
        return self


def main():
    """
    Loop Control, currently running for ML Grades Feature Testing CSV Creation
    """
    # data = Data()
    # data.categoricalFeatures()
    m = Models()
    m.LOGO()
    
    
if __name__ == '__main__':
    """
    If name = main block
    """
    main()
