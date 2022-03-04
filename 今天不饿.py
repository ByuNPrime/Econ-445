import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
lm = linear_model.LinearRegression()
from sklearn.linear_model import LogisticRegression
# import necessary packages

base = pd.read_csv('loopnet_data_ca.csv',na_values=['\\N']).fillna(np.nan)
base = base.drop(['property_type','sale_type','id','price_per_unit','address','zip','ain','crawled_id', 'property_subtype'], axis=1)
# drop unused variables

for i in range(len(base['year_renovated'])):
    if np.isnan(base['year_renovated'][i]) == True:
        base['year_renovated'][i] = 0
    else:
        base['year_renovated'][i] = 1
# preprocessing of an variable indicating whether the estate is renovated

base = base.dropna(subset=['price_usd','size_sf','lot_size_ac','year_built','opportunity_zone','cap_rate','gross_rent_multiplier'])
base['building_class'] = base['building_class'].replace(np.nan,'Missing')
base['parking_ratio'] = base['parking_ratio'].replace(np.nan, 0)
base['no_units'] = base['no_units'].replace(np.nan, 1)
base['opportunity_zone'] = base['opportunity_zone'].replace('Yes',1)
base['opportunity_zone'] = base['opportunity_zone'].replace('No',0)
base['no_stories'] = base['no_stories'].replace(np.nan, 0)
base['apartment_style'] = base['apartment_style'].replace(np.nan,'Missing')
# deal with all \N values

base['year_built'] = 2022 - base['year_built']
base = base.rename(columns={"year_built": "age"})
min_max_scaler = preprocessing.MinMaxScaler()
base[['cap_rate','gross_rent_multiplier','size_sf','no_stories','age','parking_ratio','no_units','lot_size_ac']] = min_max_scaler.fit_transform(base[['cap_rate','gross_rent_multiplier','size_sf','no_stories','age','parking_ratio','no_units','lot_size_ac']])
# using scaler

base = pd.get_dummies(base, columns=['building_class'])
base = pd.get_dummies(base, columns=['apartment_style'])
# creating level factors

base = base.drop(['building_class_Missing','apartment_style_Missing'], axis=1)
# to avoid multicollinearity

base = base.drop([0,1,2,3,4])
indep = base.drop(['price_usd'], axis=1)
division = len(indep)/10
r2 = []
explainedvariance = []
meanabsoluteerror = []
for i in range(10):
    first_stop  = i*division
    second_start = (i+1)*division
    X_train_first = indep[:int(first_stop)]
    X_train_second = indep[int(second_start):]
    X_train = pd.concat([X_train_first,X_train_second])
    X_test = indep[int(first_stop):int(second_start)]
    y_train_first = base[:int(first_stop)]
    y_train_second = base[int(second_start):]
    y_train = pd.concat([y_train_first,y_train_second])
    y_train = y_train['price_usd']
    y_test = base[int(first_stop):int(second_start)]
    y_test = y_test['price_usd']
    model = lm.fit(X_train, y_train)
    y_predontrain = lm.predict(X_train)
    r2.append(r2_score(y_train, y_predontrain))
    explainedvariance.append(explained_variance_score(y_train, y_predontrain))
    meanabsoluteerror.append(mean_absolute_error(y_train, y_predontrain))

fitness = np.array([range(10),r2,explainedvariance,meanabsoluteerror])
fitnessdf = pd.DataFrame(fitness).T
fitnessdf = fitnessdf.rename(columns={0:"i",1:"r2",2:"explainedvariance",3:"meanabsoluteerror"})
fitnessdf = fitnessdf.sort_values(by=["r2"],ascending = False)
# to find the best linear model among rolling training sets

i = fitnessdf.iloc[0,0]
first_stop  = i*division
second_start = (i+1)*division
X_train_first = indep[:int(first_stop)]
X_train_second = indep[int(second_start):]
X_train = pd.concat([X_train_first,X_train_second])
X_test = indep[int(first_stop):int(second_start)]
y_train_first = base[:int(first_stop)]
y_train_second = base[int(second_start):]
y_train = pd.concat([y_train_first,y_train_second])
y_train = y_train['price_usd']
y_test = base[int(first_stop):int(second_start)]
y_test = y_test['price_usd']
model = lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
sample = base[int(first_stop):int(second_start)]
sample['predicted_price'] = y_pred
price_spread = sample['predicted_price'] - sample['price_usd']
sample['price_spread'] = price_spread
assetranking = sample.sort_values(by=["price_spread"],ascending = False)
assetranking['index'] = assetranking.index
print(assetranking.iloc[:][:10])
candidateindex = assetranking['index'][:10].to_numpy()
# to find the five real estates with the largest price spreads

lgmodel = LogisticRegression(solver='liblinear', random_state=0)
total_pred = lm.predict(indep)
price_spread = total_pred - base['price_usd']
base['price_spread'] = price_spread
undervalue = []
for i in price_spread:
    if i > 0:
        undervalue.append(1)
    else:
        undervalue.append(0)

base['undervalue'] = undervalue

lgmodel.fit(indep,base['undervalue'])
lgoutput = lgmodel.predict(indep)
base['logis_predicted'] = lgoutput
finalanswer = base['logis_predicted'][candidateindex].to_frame()
finalanswer['index'] = finalanswer.index
finalanswer = finalanswer.sort_values(by=["logis_predicted"],ascending = False)
print(finalanswer['index'][:5])
# In conclusion, the outcome of our model implies that the real estates with
# index #8579, #8549, #8537, #8363, #8348 are those with the largest hidden value 
# from underestimation. 