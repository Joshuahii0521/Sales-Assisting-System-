###########Function#####################
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTENC
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def oversample(df, numericColumn, itemColumns):
  """
  Purpose: To make the item class balance
  Input: a data frame, numericColumn indicate which column is numeric, itemColumns indicating which column is the item class
  Output: data frame with balance class
  """
  
  nonItemColumns = [i for i in list(df.columns) if i not in itemColumns]
  output = pd.DataFrame()
  categoricalColumn = set(df.columns) - set(numericColumn) - set(itemColumns)
  
  for i in range(0, len(itemColumns)):
    temp = pd.concat([df[nonItemColumns], df[itemColumns[i]]], axis = 1).dropna()
    smote = SMOTENC(categorical_features = [temp[nonItemColumns].columns.get_loc(i) for i in categoricalColumn])
    x_smote, y_smote = smote.fit_resample(temp[nonItemColumns], temp[itemColumns[i]])
    output = pd.concat([output, pd.concat([x_smote, y_smote], axis = 1)], axis = 0)
  return output

def MBRS_fit(df, numericColumn, itemColumns):
  """
  Purpose:Train the model-based recommendation system
  Input: a data frame, numericColumn indicate which column is numeric, itemColumns indicating which column is the item to be recommended
  Output: a set of model with a scaler to normalize data and a classifier for each item that predict whether a user will buy the item
  """
  
  model = {}
  df = df.copy()
  nonItemColumns = [i for i in list(df.columns) if i not in itemColumns]
  
  
  #Normalization
  model['scaler'] = preprocessing.StandardScaler().fit(df[numericColumn])
  df[numericColumn] = model['scaler'].transform(df[numericColumn])
  
  #Training
  for i in range(0, len(itemColumns)):
    
    #subset data frame by removing item column not in question
    item = itemColumns[i]
    temp_itemColumns = itemColumns.copy()
    temp_itemColumns.remove(item)
    temp_df = df.drop(temp_itemColumns, axis = 1).dropna()
    
    #Train the model
    clf = LogisticRegression(max_iter = 10000)
    clf = clf.fit(temp_df[nonItemColumns], temp_df[item])
    
    #Store the model
    model[item] = clf
  
  return model

def MBRS_predict(df, numericColumn, itemColumns, model):
  """
  Purpose: Predict and recommend using the model given
  Input: a data frame, numericColumn indicate which column is numeric, itemColumns indicating which column is the item to be recommended, a set of model
  Output: original input data frame with the additional column of predicted class and probability for buying each item and recommendation for each user
  """
  
  nonItemColumns = [i for i in list(df.columns) if i not in itemColumns]
  df = df.dropna(subset = nonItemColumns).copy()
  prediction_df = pd.DataFrame(columns = [i + "_predclass" for i in itemColumns] + [i + "_prob" for i in itemColumns] + ["Recommendation"], index = df.index)
  
  #Normalization
  df[numericColumn] = model['scaler'].transform(df[numericColumn])

  #Predict
  for j in range(0, len(itemColumns)):
    item = itemColumns[j]
    prediction_df[item + "_prob"] = pd.DataFrame(model[item].predict_proba(df[nonItemColumns]), index = df.index)[1]
    prediction_df[item + "_predclass"] = pd.DataFrame(model[item].predict(df[nonItemColumns]), index = df.index)[0]
    
  #Recommend  
  prediction_df["Recommendation"] = prediction_df[[i + "_prob" for i in itemColumns]].idxmax(axis = 1) # Select the item column with max probability
  prediction_df["Recommendation"] = prediction_df["Recommendation"].str.replace("_prob", "")
  
  return pd.concat([df, prediction_df], axis = 1)



numericColumn = ['NumberOfChildrenVisiting' , 'PitchSatisfactionScore', 'NumberOfTrips', 'MonthlyIncome', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'DurationOfPitch', 'Age']
itemColumns = ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King']
working_directory = "" #Set your working directory here

##################Spreading###################
df = pd.read_csv(working_directory + "\\cleaned_df_train.csv", index_col = "CustomerID")
df = pd.concat([df, df.pivot(columns = 'ProductPitched', values = "ProdTaken")], axis = 1)
df = df.drop(["ProdTaken", "ProductPitched"], axis = 1)
df.to_csv(working_directory + "\\cleaned_spreaded_df_train.csv")

df = pd.read_csv(working_directory + "\\cleaned_df_test.csv", index_col = "CustomerID")
df = pd.concat([df, df.pivot(columns = 'ProductPitched', values = "ProdTaken")], axis = 1)
df = df.drop(["ProdTaken", "ProductPitched"], axis = 1)
df.to_csv(working_directory + "\\cleaned_spreaded_df_test.csv")


#########Training############
df = pd.read_csv(working_directory + "\\cleaned_spreaded_df_train.csv", index_col = "CustomerID")

#Dummy Coding
df = pd.get_dummies(df, columns = ['CityTier', 'Occupation','PreferredPropertyStar','MaritalStatus','Designation'])


#Oversample
[df[[i]].value_counts() for i in itemColumns]
df = oversample(df, numericColumn, itemColumns)
[df[[i]].value_counts() for i in itemColumns]

#Train
model = MBRS_fit(df, numericColumn ,itemColumns)

##########Testing#############
df = pd.read_csv(working_directory + "\\cleaned_spreaded_df_train.csv", index_col = "CustomerID")
df = pd.get_dummies(df, columns = ['CityTier', 'Occupation','PreferredPropertyStar','MaritalStatus','Designation'])
prediction_df = MBRS_predict(df, numericColumn, itemColumns, model)
roc_df = prediction_df[['Basic','Basic_predclass','Basic_prob']].dropna()
roc_auc_score(roc_df['Basic'], roc_df['Basic_prob'])
