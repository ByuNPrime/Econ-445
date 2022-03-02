from statistics import LinearRegression
import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
lm = linear_model.LinearRegression()
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
r2 = [0]
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
    y_test = base[int(first_stop):int(second_start)]
    model = lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    r2 = r2.append(r2_score(y_test, y_pred))

ex_var_score = explained_variance_score(y_test, y_pred)
m_absolute_error = mean_absolute_error(y_test, y_pred)
m_squared_error = mean_squared_error(y_test, y_pred)
r_2_score = r2_score(y_test, y_pred)