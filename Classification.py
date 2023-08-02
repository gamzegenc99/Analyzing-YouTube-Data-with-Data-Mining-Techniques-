
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import mean_squared_error
import math
from collections import Counter
from sklearn.metrics import accuracy_score,confusion_matrix, roc_auc_score,roc_curve, auc
from sklearn.svm import SVC


#Loading Dataset############
trainData = pd.read_excel(r"C:\YouTubeComments_Analyasis_Project\Datasets\Datasets_afterPreprocessing\16.Normalization_RemovalPunc_Lemma_Stopword(1111).xlsx")
trainData['Cyberbullying'] = trainData['Cyberbullying'].map({'var': 1, 'yok': 0}).astype(int)
print(trainData.head())
#print(trainData['Cyberbullying'].value_counts()) #
#0    3000
#1    3000


############ Vectorization ##################################################################################

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(trainData['Comment'].values.astype('U'))
trainDataX=X.toarray() 
trainDataY = trainData["Cyberbullying"].values
print("number of features" ,trainDataX.shape) #(6000, 21462) for 0000 raw data
#number of features (6000, 6414)
"""
Features at raw data 6000 instance ->21462 features then
after all preprocess  6000 instance->  6414 features

"""


#Splitting data into train and test data
xTrain, xTest, yTrain, yTest = train_test_split(trainDataX, trainDataY, random_state = 0, test_size = 0.2, shuffle = False)




##############################ModelTraining###############################################################

def logistic(xTrain,yTrain):
    lr = LogisticRegression()
    lr.fit(xTrain, yTrain)
    y_pred  = lr.predict(xTest)
    
    MSE = mean_squared_error(yTest, y_pred)
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error for LogisticRegression is {}". format(RMSE))
    return lr

def randomForest(x_train,y_train):
    rfc = RandomForestClassifier(random_state = 20,n_estimators=100,n_jobs=-1)
    rfc.fit(x_train, y_train)
    y_pred  = rfc.predict(xTest)
        
    MSE = mean_squared_error(yTest, y_pred)
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error for RandomForest  is {}". format(RMSE))
    return rfc



def decisionTree(x_train,y_train):
    dct = DecisionTreeClassifier(max_depth =10, random_state = 42)
    dct.fit(x_train, y_train)
    y_pred  = dct.predict(xTest)
   
    MSE = mean_squared_error(yTest, y_pred)
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error for DecisionTree is {}". format(RMSE))
    return dct
def svm(x_train, y_train):
    svm = SVC()
    svm.fit(x_train, y_train)
    y_pred = svm.predict(xTest)
    MSE = mean_squared_error(yTest, y_pred)
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error for SVM is {}". format(RMSE))
    fpr, tpr, thresholds = roc_curve(yTest, y_pred)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='SVM')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('SVM ROC curve')
    plt.show()
    return svm
def j48(x_train, y_train):
    j48 = DecisionTreeClassifier(criterion='entropy')
    j48.fit(x_train, y_train)
    y_pred = j48.predict(xTest)
    MSE = mean_squared_error(yTest, y_pred)
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error for J48 (C4.5 Decision Tree) is {}".format(RMSE))
    return j48


def GnaiveBayes(x_train,y_train):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    
    y_pred  = gnb.predict(xTest)
   
    
    MSE = mean_squared_error(yTest, y_pred)
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error for NaiveBayes is {}". format(RMSE))
    
    fpr, tpr, thresholds = roc_curve(yTest, y_pred)
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr,tpr, label='NaiveBayes')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('NaiveBayes ROC curve')
    plt.show()
    return gnb

def naiveBayesMultinomial(x_train, y_train):
    nbm = MultinomialNB()
    nbm.fit(x_train, y_train)
    y_pred = nbm.predict(xTest)
    MSE = mean_squared_error(yTest, y_pred)
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error for Naive Bayes Multinomial is {}". format(RMSE))
    fpr, tpr, thresholds = roc_curve(yTest, y_pred)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Naive Bayes Multinomial')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('Naive Bayes Multinomial ROC curve')
    plt.show()
    return nbm


### Plotting Decision Tree #########################3
fig = plt.figure(figsize=(35,30))
tree.plot_tree(decisionTree(xTrain,yTrain), filled=True)
plt.show()
#creating confusing matrix###############
con_mat = confusion_matrix(yTrain,yTrain)
print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


##################### Prediction ##########################################################

def predictReport(xTrain, xTest,yTrain, yTest):##accuracy report
    print("Classification report for Gaussian Naive Bayes:\n")
    print(classification_report(yTest, GnaiveBayes(xTrain,yTrain).predict(xTest))+"\n")
    print("Classification report for Random Forest:\n")
    print(classification_report(yTest, randomForest(xTrain,yTrain).predict(xTest))+"\n")
    print("Classification report for decisionTree:\n")
    print(classification_report(yTest, decisionTree(xTrain,yTrain).predict(xTest))+"\n")
    print("Classification report for  Logistic :\n")
    print(classification_report(yTest, logistic(xTrain,yTrain).predict(xTest))+"\n")
    print("Classification report for SVM:\n")
    print(classification_report(yTest, svm(xTrain, yTrain).predict(xTest)) + "\n")
    print("Classification report for J48 (C4.5 Decision Tree):\n")
    print(classification_report(yTest, j48(xTrain, yTrain).predict(xTest)) + "\n")
    print("Classification report for Naive Bayes Multinomial:\n")
    print(classification_report(yTest, naiveBayesMultinomial(xTrain, yTrain).predict(xTest)) + "\n")

print("Report for classification Prediction")
predictReport(xTrain, xTest, yTrain, yTest)







































