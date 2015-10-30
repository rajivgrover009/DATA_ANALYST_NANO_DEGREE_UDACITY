#!/usr/bin/python

import sys
import pickle
from blaze.expr.reductions import nrows
import numpy as np
from sklearn.feature_selection.univariate_selection import f_regression

sys.path.append("../tools/")
import pprint as pp
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
from sklearn.decomposition import RandomizedPCA
import matplotlib
import matplotlib.pyplot as plt
import scipy as sc
from ggplot import *
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','Percent_bonus','long_term_incentive','exercised_stock_options','from_poi_to_this_person']




### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

df=pd.DataFrame.from_dict(data=my_dataset,orient='index')
df['Name'] = df.index # Create new columns from index column
df.index = range(len(df)) # reset the index to series of number
cols =df.columns.tolist() # get the columns list
cols = cols[-1:] + cols[:-1] # rearrange colums so that Name columns is in front
df=df[cols] # Data frame with columns reorderd

df=df.convert_objects(convert_numeric=True) # Convert columns to numeric datatype
df['poi'] = df["poi"].astype('category') # Convert poi in to a category variable

# print len(df.index)
# print len(df.columns.tolist())
# print df.describe()


# Following plot depicts that there is an outlier with salary over
salary_hist= ggplot(aes(x='salary'),data=df)+geom_histogram(binwidth=100000)+ylab("frequency")+ggtitle("Salary Histogram")
ggsave(salary_hist, "Salary_hist_with_outlier.jpeg")
#Remove the Total Row from df
df=df[df['Name'] !='TOTAL']
df=df[df['Name'] !='LOCKHART EUGENE E']


#Plot the Dataframe again
salary_hist_wo_outlier= ggplot(aes(x='salary'),data=df)+geom_histogram(binwidth=100000)+ylab("frequency")+\
                        ggtitle("Salary Histogram")
ggsave(salary_hist_wo_outlier, "Salary_hist_without_outlier.jpeg")

to_messages=ggplot(aes(x='to_messages'),data=df)+geom_histogram()+ylab("frequency")+\
       ggtitle("to_messages Histogram")
ggsave(to_messages, "to_messages.jpeg")

deferral_payments= ggplot(aes(x='deferral_payments'),data=df)+geom_histogram()+ylab("frequency")+\
       ggtitle("deferral_payments Histogram")
ggsave(deferral_payments, "deferral_payments.jpeg")

total_payments=ggplot(aes(x='total_payments'),data=df)+geom_histogram()+ylab("frequency")+\
       ggtitle("total_payments Histogram")
ggsave(total_payments, "total_payments.jpeg")

bonus= ggplot(aes(x='bonus'),data=df)+geom_histogram()+ylab("frequency")+\
       ggtitle("Bonus Histogram")
ggsave(bonus, "bonus.jpeg")

long_term_incentive= ggplot(aes(x='long_term_incentive'),data=df)+geom_histogram()+ylab("frequency")+\
      ggtitle("Long Term incentive Histogram")
ggsave(long_term_incentive, "long_term_incentive.jpeg")

from_messages= ggplot(aes(x='from_messages'),data=df)+geom_histogram()+ylab("frequency")+\
      ggtitle("from_messages Histogram")
ggsave(from_messages, "from_messages.jpeg")

#Bivariate analysis using box plots
salary_poi_bp= ggplot(aes(y='poi',x='salary'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of salary in POI VS NON-POI")
ggsave(salary_poi_bp,"salary_poi_bp.jpeg")

to_messages_poi_bp= ggplot(aes(y='poi',x='to_messages'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of to_messages in POI VS NON-POI")
ggsave(to_messages_poi_bp,"to_messages_poi_bp.jpeg")

deferral_payment_poi_bp= ggplot(aes(y='poi',x='deferral_payments'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of deferral_payments in POI VS NON-POI")
ggsave(deferral_payment_poi_bp,"deferral_payment_poi_bp.jpeg")

total_payments_poi_bp=  ggplot(aes(y='poi',x='total_payments'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of total_payments in POI VS NON-POI")
ggsave(total_payments_poi_bp,"total_payments_poi_bp.jpeg")

exercised_stock_options_poi_bp=ggplot(aes(y='poi',x='exercised_stock_options'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of exercised_stock_options in POI VS NON-POI")
ggsave(exercised_stock_options_poi_bp,"exercised_stock_options_poi_bp.jpeg")

bonus_poi_bp= ggplot(aes(y='poi',x='bonus'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of Bonus in POI VS NON-POI")
ggsave(bonus_poi_bp,"bonus_poi_bp.jpeg")

restricted_stock_poi_bp= ggplot(aes(y='poi',x='restricted_stock'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of restricted_stock in POI VS NON-POI")
ggsave(restricted_stock_poi_bp,"restricted_stock_poi_bp.jpeg")

shared_receipt_with_poi_poi_bp= ggplot(aes(y='poi',x='shared_receipt_with_poi'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of shared_receipt_with_poi in POI VS NON-POI")
ggsave(shared_receipt_with_poi_poi_bp,"shared_receipt_with_poi_poi_bp.jpeg")

restricted_stock_deferred_poi_bp =ggplot(aes(y='poi',x='restricted_stock_deferred'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of restricted_stock_deferred in POI VS NON-POI")
ggsave(restricted_stock_deferred_poi_bp,"restricted_stock_deferred_poi_bp.jpeg")

loan_advances_poi_bp= ggplot(aes(y='poi',x='loan_advances'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of loan_advances in POI VS NON-POI")
ggsave(loan_advances_poi_bp,"loan_advances_poi_bp.jpeg")

from_messages_poi_bp= ggplot(aes(y='poi',x='from_messages'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of from_messages in POI VS NON-POI")
ggsave(from_messages_poi_bp,"from_messages_poi_bp.jpeg")

other_poi_bp = ggplot(aes(y='poi',x='other'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of other in POI VS NON-POI")
ggsave(other_poi_bp,"other_poi_bp.jpeg")

from_this_person_to_poi_bp= ggplot(aes(y='poi',x='from_this_person_to_poi'),data=df)+geom_boxplot()+ylab("POI")+\
     ggtitle("Distribution of from_this_person_to_poi in POI VS NON-POI")
ggsave(from_this_person_to_poi_bp,"from_this_person_to_poi_bp.jpeg")
#from_this_person_to_poi seems intersting attribute

long_term_incenttive_poi_bp= ggplot(aes(y='poi',x='long_term_incentive'),data=df)+geom_boxplot()+ylab("POI")+\
     ggtitle("Distribution of Long Term incentive in POI VS NON-POI")
ggsave(long_term_incenttive_poi_bp,"long_term_incenttive_poi_bp.jpeg")
# #Long term incentive also seems to be interesting candidate

expenses_poi_bp= ggplot(aes(y='poi',x='expenses'),data=df)+geom_boxplot()+ylab("POI")+\
     ggtitle("Distribution of expense in POI VS NON-POI")
ggsave(expenses_poi_bp,"expenses_poi_bp.jpeg")

shared_receipt_with_poi_poi_bp= ggplot(aes(y='poi',x='shared_receipt_with_poi'),data=df)+geom_boxplot()+ylab("POI")+\
     ggtitle("Distribution of Share receipt with poi in POI VS NON-POI")

ggsave(shared_receipt_with_poi_poi_bp,"shared_receipt_with_poi_poi_bp.jpeg")

#Lets create new feature ratio_messages
df['ratio_messages']=df['from_this_person_to_poi']/df['from_messages']

#Draw the plot
ratio_messages_poi_bp= ggplot(aes(x='ratio_messages',y='poi'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of ratio_messages in POI VS NON-POI")
ggsave(ratio_messages_poi_bp,"ratio_messages_poi_bp.jpeg")

# Create new feature ratio_to_from_messages
df['ratio_to_from_messages']=(df['from_poi_to_this_person']+df['from_this_person_to_poi'])/\
                             (df['to_messages']+df['from_messages'])

#draw the plot
ratio_to_from_messages_poi_bp= ggplot(aes(x='ratio_to_from_messages',y='poi'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of ratio_to_from_messages in POI VS NON-POI")
ggsave(ratio_to_from_messages_poi_bp,"ratio_to_from_messages_poi_bp.jpeg")

#Create new feature
df['Percent_bonus']=(df['bonus']/df['salary'])*100

#Draw the plot
Percent_bonus_poi_bp= ggplot(aes(x='Percent_bonus',y='poi'),data=df)+geom_boxplot()+ylab("POI")+\
       ggtitle("Distribution of Percent_bonus in POI VS NON-POI")
ggsave(Percent_bonus_poi_bp,"Percent_bonus_poi_bp.jpeg")

# Lets get the correlation matrix
print df.corr()
# # Lets see what is the relationship among different attributes, which are strongly correlated
#
salary_bonus_poi= ggplot(aes(x='salary',y='bonus',color='poi'),data=df)+geom_point()+geom_line()+ylab("bonus")+\
       ggtitle("Relation between Salary , bonus and POI")
ggsave(salary_bonus_poi,"salary_bonus_poi.jpeg")

#Above plot clearly depicts that bonus was high for pois

salary_stock_poi= ggplot(aes(x='salary',y='exercised_stock_options',color='poi'),data=df)+geom_point()+\
                  geom_line()+ylab("exercised_stock_options")+\
                  ggtitle("Relation between Salary , exercised_stock_options and POI")
ggsave(salary_stock_poi,"salary_stock_poi.jpeg")

cols=df.columns.tolist()

#cols = ['Name','poi','Percent_bonus','long_term_incentive','exercised_stock_options','from_poi_to_this_person']
cols=['Name','poi','Percent_bonus','long_term_incentive','exercised_stock_options','from_poi_to_this_person']


# use only the columns from the dataframe so that we drop only relevant rows based on missing values
df=df.filter(items=cols)

#drop the rows with missing values
df=df.dropna(how='any')


#convert the dataframe in to dictionary
my_dataset=df.set_index('Name').T.to_dict()


data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)





### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# Note : I have commented out code for Decision Tree classifier and algorithm tuning
#Decision tree classifier
#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier(min_samples_split=2)


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# Note : I have commented out code for Decision Tree classifier and algorithm tuning
# from sklearn import grid_search, datasets
# parameters = {'criterion':('gini', 'entropy'),'splitter':('best','random'), 'min_samples_split':[2,3,4,5,6,7,8,9,10]}
# dtr = DecisionTreeClassifier()
# clf = grid_search.GridSearchCV(dtr, parameters)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#Note: I have commented the code for selectKbest
#from sklearn.feature_selection import SelectKBest,f_regression
# selection = SelectKBest(f_regression,k=2)
#
# features_train_new = selection.fit_transform(features_train,labels_train)
#
#
# print selection.get_support()



clf=clf.fit(features_train,labels_train)
#pred=clf.predict(features_test)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
