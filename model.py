import pandas as pd
import numpy as np
train_claim = pd.read_csv('Train_Claim.csv',na_values='?')
train_claim.head()
train_demo = pd.read_csv('Train_Demographics.csv')
train_policy = pd.read_csv('Train_Policy.csv')
train_target = pd.read_csv('Traindata_with_Target.csv')
#CHECKING FOR NA VALUES
print(train_claim.isna().sum())
print('\n')
print(train_demo.isna().sum())
print('\n')
print(train_policy.isna().sum())
print('\n')
print(train_target.isna().sum())
print(train_claim.shape)
print(train_demo.shape)
print(train_policy.shape)
print(train_target.shape)
train_claim.columns
print(train_claim.dtypes)
print('\n')
print(train_demo.dtypes)
print('\n')
print(train_policy.dtypes)
#EXTRACTING CATEGORICAL ATTRIBUTES FOR EDA
cat_attr_train_claim = [var for var in train_claim.columns if train_claim[var].dtypes=='object']
cat_attr_train_claim
#Plotting authorities contacted.
import seaborn as sb 
import matplotlib.pyplot as plt
sb.countplot(x=train_claim['AuthoritiesContacted'])
plt.show()
plt.figure(figsize=(10,8))
sb.countplot(x=train_claim['TypeOfIncident'],hue=train_claim['NumberOfVehicles'])
plt.show()
train_claim['AmountOfInjuryClaim'].nunique()
train_claim['Witnesses'].nunique()
plt.figure(figsize=(10,8))
sb.countplot(x=train_claim['Witnesses'],hue=train_claim['NumberOfVehicles'])
num_attr_train_claim = [ var for var in train_claim.columns if train_claim[var].dtypes =='int64']
num_attr_train_claim
plt.figure(figsize=(10,8))
sb.countplot(x=train_claim['TypeOfCollission'],hue=train_claim['NumberOfVehicles'])
plt.show()
cat_attr_train_demo = [ var for var in train_demo.columns if train_demo[var].dtypes=='object']
cat_attr_train_demo
plt.figure(figsize=(10,8))
sb.barplot(data=train_demo,x='InsuredGender',y='CapitalGains')
plt.show()
plt.figure(figsize=(10,8))
sb.barplot(data=train_demo,x='InsuredEducationLevel',y='CapitalGains',hue='InsuredGender')
plt.show()

train_claim['CustomerID'] = train_claim['CustomerID'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int)
train_claim['CustomerID']
train_demo
train_demo['CustomerID'] = train_demo['CustomerID'].astype('str').str.extract('(\d+)').astype(int)
train_demo['CustomerID'].min()
train_policy['CustomerID'] = train_policy['CustomerID'].astype('str').str.extract('(\d+)').astype(int)
train_policy['CustomerID'].min()
train_demo = train_demo.sort_values('CustomerID')
train_demo.head()
train_claim = train_claim.sort_values('CustomerID')
train_policy = train_policy.sort_values('CustomerID')
train_claim.head()
train_policy.shape
train_policy['CustomerID'].nunique()
train_policy.head()
merge1 = pd.merge(train_demo,train_claim,on='CustomerID',how='inner')
merge1.shape
merge2 = pd.merge(merge1,train_policy,on='CustomerID')
merge2.shape
final = pd.DataFrame(merge2)
type(final)
final
final.dropna(inplace=True)
final.shape
final.dtypes
from sklearn.preprocessing import OneHotEncoder,StandardScaler
cat_attr_final = [var for var in final.columns if final[var].dtypes=='object']
num_attr_final = [var for var in final.columns if final[var].dtypes=='int64']
target = pd.read_csv('Traindata_with_Target.csv')

target.shape
target['CustomerID'] = target['CustomerID'].astype('str').str.extract('(\d+)').astype(int)
final_data = pd.merge(final,target,on='CustomerID')
final_data.shape
final_data.head()
from sklearn.model_selection import train_test_split
#X = final_data.drop('ReportedFraud',axis=1)
Y = final_data['ReportedFraud']
#xtrain,xtest,
ytrain,ytest = train_test_split(Y,test_size=0.25,random_state=123)
scaler = StandardScaler()
#X_num_std = scaler.fit_transform(X[num_attr_final])
#ohe = OneHotEncoder()
#X_cat_std = ohe.fit_transform(X[cat_attr_final]).toarray()
#X_new = np.concatenate((X_num_std,X_cat_std),axis=1)
#X_train,X_test = train_test_split(X_new,test_size=0.25,random_state=1010)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y_train = le.fit_transform(ytrain)
Y_test = le.fit_transform(ytest)
corr = final_data.corr()
threshold = 0.05
a = abs(corr)
relevant_features = a[a>0.05]
relevant_features
relevant = []
for i in relevant_features.columns:
    if relevant_features[i].values.all()>0:
        relevant.append(i)
relevant.remove('CustomerID')



relevant_features_data = final_data[relevant]
relevant_features_data
X = relevant_features_data
xtrain_ref,xtest_ref = train_test_split(X,test_size=0.25,random_state=101)
scaler = StandardScaler()
X_rf_std = scaler.fit_transform(xtrain_ref)
X_rf_test = scaler.fit_transform(xtest_ref)
from sklearn.tree import DecisionTreeClassifier
dtclf = DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=10,max_features='sqrt')
dtclf.fit(X_rf_std,Y_train)
predict_new_rel = dtclf.predict(X_rf_std)
predict_new_test = dtclf.predict(X_rf_test)
from sklearn.metrics import classification_report
print(classification_report(Y_train,predict_new_rel))
print(classification_report(Y_test,predict_new_test))

#from sklearn.ensemble import RandomForestClassifier
#rdclf = RandomForestClassifier()
#rdclf.fit(X_train,Y_train)

#predict_rdf = rdclf.predict(X_train)
##predict_rf_test = rdclf.predict(X_test)
#from sklearn.metrics import classification_report
#print(classification_report(Y_train,predict_rdf))
#print(classification_report(Y_test,predict_rf_test))
#feat_importance = rdclf.feature_importances_
#feat_importance = np.argsort(feat_importance)[::-1][:10]
import pickle
pickle.dump(dtclf,open('model.pkl','wb'))
print(dtclf.predict(X_rf_test))
