import pandas as panda
import numpy as numpy
import seaborn as seaborn
import matplotlib.pyplot as matplotlib
from statsmodels.graphics.mosaicplot import mosaic
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

categories=['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']
quantative=['age', 'avg_glucose_level', 'bmi']

#load the data set that is provided
training_set=panda.read_csv("train_2v.csv")
training_set_normalized=panda.read_csv("train_2v.csv")

#remove any null entries
training_set=training_set.dropna()
training_set_normalized=training_set_normalized.dropna()

#print number of entries that are complete
print(len(training_set.index))

#Pre-Processing Normalize quantative variables to unit vector
columns_to_normalize = ['bmi', 'age', 'avg_glucose_level']
data_to_normalize= training_set_normalized[columns_to_normalize].values
normalized_data = preprocessing.normalize(data_to_normalize)
df_temp = panda.DataFrame(normalized_data, columns=columns_to_normalize, index = training_set.index)
training_set_normalized[columns_to_normalize] = df_temp

#Data Exploration
def explore_data(data):
    #Print Column Names
    print("Column Names of Training Set:")
    print(data.columns)


    #Print summary statistic for each variable in the training set
    for variable in data.columns:
        print(data[variable].describe())

    #Plot bar chart of each category to visualize balance
    for category in categories:
        seaborn.countplot(data[category])
        matplotlib.show()

    #Plot distribution chart of each category to visualize balance
    for quant in quantative:
        seaborn.distplot(data[quant])
        matplotlib.show()

    #Correlation Matrix
    data_correlation = data.corr()
    f, ax = matplotlib.subplots(figsize=(12, 9))
    seaborn.heatmap(data_correlation, vmax=.8, square=True)
    matplotlib.show()

    #Investigate balance of stroke incidence across gender
    mosaic(data, ['gender','stroke'], axes_label=True)
    matplotlib.show()

    #Investigate balance of stroke incidence across hypertension
    mosaic(data, ['hypertension','stroke'], axes_label=True)
    matplotlib.show()

    #Investigate balance of stroke incidence across heart disease
    mosaic(data, ['heart_disease','stroke'], axes_label=True)
    matplotlib.show()

    #Investigate stroke across blood glucose
    stroke_and_bgl= panda.concat([data['avg_glucose_level'], data['stroke']], axis=1)
    f, ax = matplotlib.subplots(figsize=(8, 6))
    fig = seaborn.boxplot(x="stroke", y="avg_glucose_level", data=stroke_and_bgl)
    matplotlib.show()

    #Investigate stroke across age
    stroke_and_age= panda.concat([data['age'], data['stroke']], axis=1)
    f, ax = matplotlib.subplots(figsize=(8, 6))
    fig = seaborn.boxplot(x="stroke", y="age", data=stroke_and_age)
    matplotlib.show()

#explore data on the orginal complete set
explore_data(training_set)

#explore data on the normalized set
explore_data(training_set_normalized)

#Pre-Processing: convert catageorical variables into dummy variables
training_set_dummies=panda.get_dummies(training_set)

#rename columns to avoid spaces
training_set_dummies.rename(columns={'smoking_status_never smoked': 'smoking_status_never_smoked', 'smoking_status_formerly smoked': 'smoking_status_formerly_smoked'}, inplace=True)

#Split Training and Test Sets
stroke = training_set_dummies.stroke # define stroke as the dependent variable
features=training_set_dummies.loc[:,["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "gender_Female",
                             "gender_Male", "gender_Other", "ever_married_No", "ever_married_Yes", 'work_type_Govt_job',
                             'work_type_Never_worked', 'work_type_Private''work_type_Self-employed', 'work_type_children', 'Residence_type_Rural',
                             'Residence_type_Urban', 'smoking_status_formerly_smoked','smoking_status_never_smoked', 'smoking_status_smokes']]

#Create training and testing vars
features_train, features_test, stroke_train, stroke_test = train_test_split(features, stroke, test_size=0.33)

# def naivebayes(dependent_train, independent_train, dependent_test, independent_test):
#     NBmodel=GaussianNB()
#     NBmodel.fit(dependent_train, independent_train)
#     print(NBmodel.score(dependent_test, independent_test))
#     NB_prediction=NBmodel.predict_proba(dependent_test)
#     return NB_prediction
#
# naivebayes(features_train, stroke_train, features_test, stroke_test)


