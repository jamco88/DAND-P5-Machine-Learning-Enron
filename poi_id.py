#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn import feature_selection
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from pprint import pprint
import timeit
import numpy as np
import pandas

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


TotalFeatures = len(data_dict["SKILLING JEFFREY K"])
people = len(data_dict)
TotalObs = TotalFeatures*people

print("The number of features is:", str(TotalFeatures))
    
count = 0
for key, value in data_dict.iteritems():
	if data_dict[key]["poi"]==1:
		count += 1

print "The total number of PoI's in the initial dataset is {0} out of {1}:".format(count, people)
print("The total number of observations is:", str(TotalObs))
#are there any features with many missing values?
#for this it's easier to make the switch from a python dictionary to a dataframe
df = pandas.DataFrame.from_dict(list(data_dict.values()),dtype="float64")
employees = pandas.Series(list(data_dict.keys()))
df.set_index(employees, inplace=True)
print "Number of NaN's for each feature:"
df.isnull().sum()

#Visualisations / Data Exploration

import matplotlib.pyplot as plt

x = df[["bonus"]]
y = df[["salary"]]
cpoi = df[["poi"]]

plt.scatter(x,y,c=cpoi)
plt.show()

df2 = df.sort_values(by='salary', ascending=0)


ps = df2.count(axis=1)
ps.sort_values(ascending=True)

df2 = df2.drop({"LOCKHART EUGENE E"})
df2 = df2.drop(["TOTAL"])
df2 = df2.drop(["THE TRAVEL AGENCY IN THE PARK"])
len(df2.index)

x = df2.total_stock_value
y = df2.salary
cpoi = df2.poi

plt.scatter(x,y,c=cpoi)

plt.show()


#New Feature Creation
df2["Fromfract"] = df2.from_poi_to_this_person.divide(df2.from_messages, axis="index").fillna(0)
df2["Tofract"] = df2.from_this_person_to_poi.divide(df2.to_messages, axis="index").fillna(0)
df2["SaltoPay"] = df2.salary.divide(df2.total_payments, axis="index").fillna(0)
df2["ESVtoTSV"] = df2.exercised_stock_options.divide(df2.total_stock_value, axis="index").fillna(0)
df2["RStoTSV"] = df2.restricted_stock.divide(df2.total_stock_value, axis="index").fillna(0)

df2.corr(method="pearson")

df2.fillna(0)

#conver float nan to string NaN, otherwise the featureformat function won't deal with it.
df2 = df2.replace(np.nan, "NaN", regex=True)

# create a dictionary from the dataframe
data_dict = df2.to_dict('index')

my_dataset = data_dict

features_list = ['poi','salary', 'bonus', 'expenses', "total_payments", "exercised_stock_options",
"restricted_stock", "long_term_incentive", "deferral_payments", "deferred_income", "director_fees",
"other", "shared_receipt_with_poi", "total_payments", "total_stock_value", "Fromfract", "Tofract",
"SaltoPay", "ESVtoTSV", "RStoTSV"]


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#create default featurelist, datadict for comparison
default_features_list = ['poi','salary', 'bonus', 'expenses', "total_payments", "exercised_stock_options",
"restricted_stock", "long_term_incentive", "deferral_payments", "deferred_income", "director_fees",
"other", "shared_receipt_with_poi", "total_payments", "total_stock_value"]
#passing in the above list means that the new features are ignored
default_data =featureFormat(my_dataset, default_features_list, sort_keys = True)
#default features, default labels
dlabels, dfeatures = targetFeatureSplit(default_data)
dX_train, dX_test, dy_train, dy_test = train_test_split(dfeatures, dlabels, test_size=0.3, random_state=42)

#Investigate most important features using SSS to ensure robust importance results.
#Compare the results for both the default feature list and the feature list I created with my new ratios.
#repurposed some of the code from tester.py 
from sklearn.cross_validation import StratifiedShuffleSplit


#Cross validate and access Feature Importances on decisiontreeclassifier with 100 folds
clf = DecisionTreeClassifier(random_state=42)

cv = StratifiedShuffleSplit(labels, 100, random_state = 42)

FeatureValue = np.zeros(len(features_list)-1)
featrscore = np.zeros(len(features_list)-1)
featpscore = np.zeros(len(features_list)-1)

for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

    fit1 = clf.fit(features_train, labels_train)
        #add values to list
    FeatureValue += fit1.feature_importances_  
    pred = clf.predict(features_test)
    featrscore += recall_score(pred, labels_test)  
    featpscore += precision_score(pred, labels_test)
        
clf2 = DecisionTreeClassifier(random_state=42)       


FeatureValueDefault = np.zeros(len(default_features_list)-1)
featrscored = np.zeros(len(features_list)-1)
featpscored = np.zeros(len(features_list)-1)


for train_idx, test_idx in cv: 
    dfeatures_train = []
    dfeatures_test  = []
    dlabels_train   = []
    dlabels_test    = []
    for ii in train_idx:
        dfeatures_train.append( dfeatures[ii] )
        dlabels_train.append( dlabels[ii] )
    for jj in test_idx:
        dfeatures_test.append( dfeatures[jj] )
        dlabels_test.append( dlabels[jj] )

    fit2 = clf2.fit(dfeatures_train, dlabels_train)
 
    FeatureValueDefault += fit2.feature_importances_
    pred2 = clf2.predict(dfeatures_test)
    featrscored += recall_score(pred2, dlabels_test)  
    featpscored += precision_score(pred2, dlabels_test)
    
    
    
#average over the 100 folds
FeatureValue = FeatureValue/100
featrscore = featrscore/100
featpscore = featpscore/100

FeatureValueDefault = FeatureValueDefault/100
featrscored = featrscored/100
featpscored = featpscored/100


print "New Featureset:"
print "{} Recall & {} Precision".format(featrscore[1], featpscore[1])


print "Old Featureset:"
print "{} Recall & {} Precision".format(featrscored[1], featpscored[1])

ranks1 = np.argsort(FeatureValue)[::-1]
print "All features for Updated Featurelist"
for n in range(len(features_list)-1):
    print "{} feature: {} ({})".format(n, features_list[n+1], FeatureValue[ranks1[n]])

    ranks2 = np.argsort(FeatureValueDefault)[::-1]
print "All features for Default Featurelist"
for n in range(len(default_features_list)-1):
    print "{} feature: {} ({})".format(n, default_features_list[n+1], FeatureValueDefault[ranks2[n]])  
    


#create scaler
MMSs = MMS()
#Create Transformers
SKBt = SelectKBest(k=6)
PCAt = PCA(random_state=42)
FUt = FeatureUnion([("kbest", SKBt), ("pca", PCAt)])
#Create Classifiers
GNBc = GaussianNB()
SVCc = SVC(kernel="linear", random_state=42)
DTCc = DecisionTreeClassifier(random_state=42)
RFc = RandomForestClassifier(random_state=42)

#final parameter section

#create scaler
MMSs = MMS()
#Create Transformers
SKBt = SelectKBest(k=6)


#create pipeline
Pipe = Pipeline([("MMS", MMSs),
                ("SKB", SKBt),
                 ("clf", GNBc)])
#Gauss
prm_grid = dict(
               )
    
cv_sss = StratifiedShuffleSplit(labels, 100, test_size=0.3, random_state=42)

#create a gridsearch object
grid = GridSearchCV(Pipe, prm_grid, cv = cv_sss, scoring ="f1")

#Gauss Tuner
start_time = timeit.default_timer()
grid.fit(features, labels)
print("Time Taken:  %r" % (timeit.default_timer() - start_time))
print "Best F1 Score:"
print(grid.best_score_)
print "Best parameters:"
best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(prm_grid.keys()):
       print '\t%s: %r' % (param_name, best_parameters[param_name])

clf = grid.best_estimator_
#import test_classifier
from tester import test_classifier
print "Tester Classification Report"
test_classifier(clf, my_dataset, features_list)

my_feature_list = features_list

dump_classifier_and_data(clf, my_dataset, my_feature_list)
print "Run complete"
