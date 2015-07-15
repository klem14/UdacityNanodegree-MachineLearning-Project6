#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.pipeline import make_pipeline

##Reproduce the piece of code that split data into train and test in tester.py
def split_data_classifier(features, labels):
    from sklearn.cross_validation import StratifiedShuffleSplit
    
    cv = StratifiedShuffleSplit(labels, n_iter=1000, random_state = 42)

    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append(features[ii] )
            labels_train.append(labels[ii] )
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])
   
    return features_train, features_test, labels_train, labels_test

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#For now select them all but email address. Later, only the most significant will be kept using SelectKBest function.
features_list = [ 'poi','to_messages','from_this_person_to_poi','from_messages','from_poi_to_this_person','other','salary','bonus','restricted_stock',
 'shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','total_payments','exercised_stock_options','long_term_incentive','expenses',
 'director_fees','deferral_payments','deferred_income','loan_advances'] # You will need to use more features

#Number of features to keep
k = 12

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict.pop('TOTAL')


### Task 3: Create new feature(s)
for employee in data_dict.keys():
    data_dict[employee]['from_poi_percent']  = round(float(data_dict[employee]['from_poi_to_this_person'])/float(data_dict[employee]['to_messages']),4) if data_dict[employee]['to_messages']!="NaN" else "NaN"
    data_dict[employee]['to_poi_percent']  = round(float(data_dict[employee]['from_this_person_to_poi'])/float(data_dict[employee]['from_messages']),4) if data_dict[employee]['from_messages']!="NaN" else "NaN"

features_list.append("from_poi_percent")
features_list.append("to_poi_percent")

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#extract the list indexes for poi and non-poi 
poi_index = [ idx for idx,elem in enumerate(labels) if elem ==1 ]
not_poi_index = [ idx for idx,elem in enumerate(labels) if elem ==0 ]
 
### draw the scatterplot, with color-coded poi and non-poi
import matplotlib.pyplot as plt
r=plt.scatter([features[i][-2] for i in poi_index], [features[i][-1] for i in poi_index], color='r')
b=plt.scatter([features[i][-2] for i in not_poi_index], [features[i][-1] for i in not_poi_index], color='b')

plt.legend([r, b], ["POI", "non-POI"], loc=1)

#label the axes
plt.xlabel("from_poi_percent")
plt.ylabel("to_poi_percent")

plt.show() 

##scaling not required
#from sklearn.preprocessing import MinMaxScaler
#scl = MinMaxScaler()
#features = scl.fit_transform(features)

##Select and print the top x most significant features for information
from sklearn.feature_selection import SelectKBest,f_classif
anova_filter = SelectKBest(f_classif, k=k)
anova_filter.fit(features, labels)
from itertools import compress

#Extract top k features from features_list
features_list = list(compress(features_list[1:],anova_filter.get_support(indices=False)))
#sort the list by score
features_list = sorted(zip(features_list,list(anova_filter.scores_)), key = lambda tup: tup[1], reverse=True)
#Only keep the names (scores dropped)
features_list = zip(*features_list)[0]
#print features_list

features_list = [ 'poi'] + list(features_list)
 


##Split data into train and test
features_train, features_test, labels_train, labels_test = split_data_classifier(features, labels)
#from sklearn import cross_validation
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=8)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

##Seleect top k most relevant features
anova_filter = SelectKBest(f_classif, k=k)

##First model
#from sklearn.naive_bayes import GaussianNB
#algo = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

##Second model
#from sklearn.neighbors import KNeighborsClassifier
#algo = KNeighborsClassifier()
#algo = KNeighborsClassifier(algorithm='brute',n_neighbors=5,p=1)


##Third model
from sklearn.tree import DecisionTreeClassifier
#algo = DecisionTreeClassifier()
algo = DecisionTreeClassifier(criterion ='entropy',splitter='best',min_samples_split=2,max_depth=2)


clf = make_pipeline(anova_filter,algo)
clf.fit(features_train,labels_train)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

##########################################
###Uncomment below for parameter tuning###
##########################################

#from sklearn.grid_search import GridSearchCV
###params = dict(selectkbest__k=[5, 6, 7, 8, 9])
##params = dict(selectkbest__k=[5, 6, 7, 8, 9],kneighborsclassifier__n_neighbors=[5, 8, 10, 15],kneighborsclassifier__p=[1, 2, 3])
#params = dict(selectkbest__k=[5,6,7,8,9,10,11,12,14],decisiontreeclassifier__min_samples_split=[2,4,6,8,10],decisiontreeclassifier__max_depth=[2,4,6,8,10],decisiontreeclassifier__splitter=['random','best'])

#from sklearn.cross_validation import StratifiedShuffleSplit
#cv = StratifiedShuffleSplit(labels, n_iter=100, random_state=42)

#grid_search = GridSearchCV(clf, param_grid=params,cv=cv,scoring='f1')
#grid_search.fit(features,labels)

#print("Best parameters set found on development set:")
#print (grid_search.best_estimator_)
#print "Best F1 score found: {:4f}".format(grid_search.best_score_)


test_classifier(clf, my_dataset, features_list)
#    
#### Dump your classifier, dataset, and features_list so 
#### anyone can run/check your results.
dump_classifier_and_data(clf, my_dataset, features_list)

##Print prediction vs actual
#for i,j in enumerate(features):
#    print algo.predict(j[:k]), labels[i]
