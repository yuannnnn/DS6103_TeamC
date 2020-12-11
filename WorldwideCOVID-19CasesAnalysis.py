#%%[markdown]
#
# # Q1
# Which countries are making progress against the pandemic? 
# Due to case counts are only meaningful if we also know how much testing a country does, what are the rate of positive tests of them?
# %%
# All packages needed here

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import Image 
import pydotplus

# Import data

dirpath = os.getcwd() 
filepath = os.path.join(dirpath ,'owid-covid-data.csv')
data = pd.read_csv(filepath)

#%%
# Standard quick checks
def dfChkBasics(dframe, valCnt = False): 
  cnt = 1
  print('\ndataframe Basic Check function -')
  
  try:
    print(f'\n{cnt}: info(): ')
    cnt+=1
    print(dframe.info())
  except: pass

  print(f'\n{cnt}: describe(): ')
  cnt+=1
  print(dframe.describe())

  print(f'\n{cnt}: head() -- ')
  cnt+=1
  print(dframe.head())

  print(f'\n{cnt}: shape: ')
  cnt+=1
  print(dframe.shape)

  if (valCnt):
    print('\nValue Counts for each feature -')
    for colname in dframe.columns :
      print(f'\n{cnt}: {colname} value_counts(): ')
      print(dframe[colname].value_counts())
      cnt +=1

dfChkBasics(data)

# %%
"""To select ten countries that have most numbers applied the condition."""

def top10(case):
    new_data = data[["location","continent","date",case]].dropna()
    latest = new_data[new_data["date"] == "2020-11-10"]
    latest = latest.sort_values([case], ascending=0)
    pick10 = latest.head(10)
    pick10 = list(pick10["location"])
    the10 = new_data.loc[new_data['location'].isin(pick10)]
    return the10
# %%
total_cases10 = top10('total_cases')
sns.lineplot(data=total_cases10, x="date", y="total_cases", hue="location")
plt.legend(frameon=False, loc='upper left')
sns.despine()
# %%
cont= total_cases10['continent'].value_counts()
pie, ax = plt.subplots()
labels = cont.keys()
plt.pie(x=cont, autopct="%.1f%%", labels=labels, pctdistance=0.5)
plt.show()
# %%
total_cases_per_million10 = top10('total_cases_per_million')
sns.lineplot(data=total_cases_per_million10, x="date", y="total_cases_per_million", hue="location")
plt.legend(frameon=False, loc='upper left')
sns.despine()
# %%
cont= total_cases_per_million10['continent'].value_counts()
pie, ax = plt.subplots()
labels = cont.keys()
plt.pie(x=cont, autopct="%.1f%%", labels=labels, pctdistance=0.5)
plt.show()
# %%
total_deaths10 = top10('total_deaths')
sns.lineplot(data=total_deaths10, x="date", y="total_deaths", hue="location")
plt.legend(frameon=False, loc='upper left')
sns.despine()
# %%
cont= total_deaths10['continent'].value_counts()
pie, ax = plt.subplots()
labels = cont.keys()
plt.pie(x=cont, autopct="%.1f%%", labels=labels, pctdistance=0.5)
plt.show()
# %%
total_deaths_per_million10 = top10('total_deaths_per_million')
total_deaths_per_million10 = total_deaths_per_million10[total_deaths_per_million10["location"] != "San Marino"]
sns.lineplot(data=total_deaths_per_million10, x="date", y="total_deaths_per_million", hue="location")
plt.legend(frameon=False, loc='upper left')
sns.despine()
# %%
cont= total_deaths_per_million10['continent'].value_counts()
pie, ax = plt.subplots()
labels = cont.keys()
plt.pie(x=cont, autopct="%.1f%%", labels=labels, pctdistance=0.5)
plt.show()

# %%
""" To find countries that are making progress against the pandemic. """
def better(case):
    thecase = data[["location","continent","date",case]].dropna()
    latest = thecase[thecase["date"] == "2020-11-01"]
    half_month_ago = thecase[thecase["date"] == "2020-10-15"]
    df = pd.merge(latest,half_month_ago,on="location")
    df["better"] =  df[case+"_x"] < df[case+"_y"]
    better = df[df["better"] == 1]
    return better
# %%
better1 = better("new_cases_smoothed")
better1
# %%
cont= better1['continent_x'].value_counts()
pie, ax = plt.subplots()
labels = cont.keys()
plt.pie(x=cont, autopct="%.1f%%", labels=labels, pctdistance=0.5)
plt.show()

# %%
names = list(better1["location"])
better1_data = data.loc[data['location'].isin(names)]
sns.lineplot(data=better1_data, x="date", y="new_cases_smoothed", hue="continent")
plt.legend(frameon=False, loc='upper left')
sns.despine()
# %%
better2 = better("new_cases_smoothed_per_million")
better2
# %%
better3 = better("positive_rate")
better3
# %%
cont= better3['continent_x'].value_counts()
pie, ax = plt.subplots()
labels = cont.keys()
plt.pie(x=cont, autopct="%.1f%%", labels=labels, pctdistance=0.5)
plt.show()
# %%
better4 = better("new_deaths_smoothed_per_million")
better4
# %%
names = list(better4["location"])
better1_data = data.loc[data['location'].isin(names)]
sns.lineplot(data=better1_data, x="date", y="new_deaths_smoothed_per_million", hue="continent")
plt.legend(frameon=False, loc='upper left')
sns.despine()
# %%
cont= better4['continent_x'].value_counts()
pie, ax = plt.subplots()
labels = cont.keys()
plt.pie(x=cont, autopct="%.1f%%", labels=labels, pctdistance=0.5)
plt.show()
# %%
result = pd.merge(better1,better2,on="location")
result = pd.merge(result,better3,on="location")
result = pd.merge(result,better4,on="location")
result = result[["location","continent_x_x"]]
result
# %%
names = list(result["location"])
better1_data = data.loc[data['location'].isin(names)]
sns.lineplot(data=better1_data, x="date", y="new_cases_smoothed", hue="location")
plt.legend(frameon=False, loc='upper left')
sns.despine()
# %% [markdown]
# # Q2 Can the Government Response Stringency Index level affect confirmed cases?
# # &
# # Q3 Can hospital conditions affect death rates?

# %%
new_data=data[['location','new_cases','stringency_index']].dropna()
new_data=new_data.reset_index(drop=True)
# %%
plt.scatter(new_data.new_cases[new_data.location=='United States'],y=new_data.stringency_index[new_data.location=='United States'])
plt.xlabel('stringency index')
plt.ylabel('new cases')
plt.title('United States')
plt.show()
plt.scatter(new_data.new_cases[new_data.location=='Russia'],y=new_data.stringency_index[new_data.location=='Russia'])
plt.xlabel('stringency index')
plt.ylabel('new cases')
plt.title('Russia')
plt.show()
plt.scatter(new_data.new_cases[new_data.location=='Mexico'],y=new_data.stringency_index[new_data.location=='Mexico'])
plt.xlabel('stringency index')
plt.ylabel('new cases')
plt.title('Mexico')
plt.show()
plt.scatter(new_data.new_cases[new_data.location=='Italy'],y=new_data.stringency_index[new_data.location=='Italy'])
plt.xlabel('stringency index')
plt.ylabel('new cases')
plt.title('Italy')
plt.show()
plt.scatter(new_data.new_cases[new_data.location=='India'],y=new_data.stringency_index[new_data.location=='India'])
plt.xlabel('stringency index')
plt.ylabel('new cases')
plt.title('India')
plt.show()
plt.scatter(new_data.new_cases[new_data.location=='United Kingdom'],y=new_data.stringency_index[new_data.location=='United Kingdom'])
plt.xlabel('stringency index')
plt.ylabel('new cases')
plt.title('United Kingdom')
plt.show()
plt.scatter(new_data.new_cases[new_data.location=='France'],y=new_data.stringency_index[new_data.location=='France'])
plt.xlabel('stringency index')
plt.ylabel('new cases')
plt.title('France')
plt.show()
plt.scatter(new_data.new_cases[new_data.location=='Colombia'],y=new_data.stringency_index[new_data.location=='Colombia'])
plt.xlabel('stringency index')
plt.ylabel('new cases')
plt.title('Colombia')
plt.show()
plt.scatter(new_data.new_cases[new_data.location=='Brazil'],y=new_data.stringency_index[new_data.location=='Brazil'])
plt.xlabel('stringency index')
plt.ylabel('new cases')
plt.title('Brazil')
plt.show()
plt.scatter(new_data.new_cases[new_data.location=='Argentina'],y=new_data.stringency_index[new_data.location=='Argentina'])
plt.xlabel('stringency index')
plt.ylabel('new cases')
plt.title('Argentina')
plt.show()
# %%
def getcorr(country):
    corr1=new_data.new_cases[new_data.location==country].corr(new_data.stringency_index[new_data.location==country])
    print(country+': %f'%corr1)
    return corr1
#%%
getcorr('United States')
getcorr('Russia')
getcorr('Mexico')
getcorr('Italy')
getcorr('India')
getcorr('United Kingdom')
getcorr('France')
getcorr('Colombia')
getcorr('Brazil')
getcorr('Argentina')
#%%

#%%
def getcorr1(country):
    case=new_data.new_cases[new_data.location==country].reset_index(drop=True)
    stringency=new_data.stringency_index[new_data.location==country].reset_index(drop=True)
    case=case[1:]-case.values[0:-1]
    corr2=case.corr(stringency[1:])
    print(country+': %f'%corr2)
    return corr2
# %%
getcorr1('United States')
getcorr1('Russia')
getcorr1('Mexico')
getcorr1('Italy')
getcorr1('India')
getcorr1('United Kingdom')
getcorr1('France')
getcorr1('Colombia')
getcorr1('Brazil')
getcorr1('Argentina')
#%%
def getcorr2(country):
    case=new_data.new_cases[new_data.location==country].reset_index(drop=True)
    stringency=new_data.stringency_index[new_data.location==country].reset_index(drop=True)
    corr3=case[4:].corr(stringency[0:-4])
    print(country+': %f'%corr3)
    return corr3
#%%
getcorr2('United States')
getcorr2('Russia')
getcorr2('Mexico')
getcorr2('Italy')
getcorr2('India')
getcorr2('United Kingdom')
getcorr2('France')
getcorr2('Colombia')
getcorr2('Brazil')
getcorr2('Argentina')
# %%
newdata=data[['total_cases','total_deaths','hospital_beds_per_thousand','cardiovasc_death_rate','hosp_patients_per_million','icu_patients_per_million']].dropna()
newdata=newdata.reset_index(drop=True)
newdata['death rate']=newdata['total_deaths']/newdata['total_cases']
cor=newdata['cardiovasc_death_rate'].corr(newdata['hospital_beds_per_thousand'])
print(cor)
#%%[markdown]
#
# # Q4
# Whether death rate is influenced by aging degree? 
# For different countries, does bigger share of older people mean bigger death rate?
# %%
# preprocess
data1 = data[['continent','location', 'date', 'total_cases', 'total_deaths','median_age','aged_65_older','aged_70_older']]

data3 = data1.dropna()
data3['deaths_rate'] = data3['total_deaths'] / data3['total_cases']
# %%
data4 = data3[data3['total_cases'] > 50]
dfChkBasics(data4)  
# %%
def selectDeathsRate(raw,x,n):
    
    latest = raw[raw['date'] == '2020-11-10'].sort_values(['deaths_rate'], ascending=x)
    temp = latest.head(n)
    temp = list(temp['location'])
    return raw.loc[raw['location'].isin(temp)]

# example: selectDeathRate(data4,0,10)

#%%
top10 = selectDeathsRate(data4,0,10)
#%%
last10 = selectDeathsRate(data4,1,10)

#%%
#
sns.lineplot(data=top10, x='date', y='deaths_rate', hue='location')
plt.legend(frameon = False, loc='upper left')
plt.title('plot of 10 countries with the highest death rate')
sns.despine()
#%%
sns.lineplot(data=last10, x='date', y='deaths_rate', hue='location')
plt.legend(frameon = False, loc='upper left')
plt.title('plot of 10 countries with the lowest death rate')
sns.despine()
#%%
show1 = top10[top10['date']== '2020-11-10'][['continent', 'location', 'deaths_rate','median_age','aged_65_older','aged_70_older']]
print(show1)
#%%
show2 = last10[last10['date']== '2020-11-10'][['continent', 'location', 'deaths_rate','median_age','aged_65_older','aged_70_older']]
print(show2)
# %%
from statsmodels.formula.api import ols
model1 = ols(formula= 'deaths_rate~aged_70_older', data= data4)
print(model1.fit().summary())
# %%
model2 = ols(formula= 'deaths_rate~aged_65_older', data= data4)
print(model2.fit().summary())

# %%
model3 = ols(formula= 'deaths_rate~median_age+aged_65_older',data= data4)
print(model3.fit().summary())
# %% [markdown]
# # Q5 What elements can affect the death rates and how can these factors predict the death rates？
# %%
data.columns
# %%
selected = data[['positive_rate','cardiovasc_death_rate','new_cases_smoothed_per_million','continent', 'location', 'date',
       'total_cases_per_million', 
       'total_deaths_per_million', 
       'new_deaths_smoothed_per_million', 
       'icu_patients_per_million', 'hosp_patients','hosp_patients_per_million',
       'weekly_icu_admissions_per_million', 
       'weekly_hosp_admissions_per_million',
       'total_tests_per_thousand', 'new_tests_per_thousand',
       'new_tests_smoothed_per_thousand',
       'stringency_index',
       'population', 'population_density', 'median_age', 'aged_65_older',
       'aged_70_older', 'gdp_per_capita', 'extreme_poverty',
       'diabetes_prevalence', 'female_smokers',
       'male_smokers', 'hospital_beds_per_thousand',
       'life_expectancy', 'human_development_index']].dropna()
selected
# %%
cor = selected.corr()
cor
# %%
pos1 = cor[cor["positive_rate"]>0.5]
pos2 = cor[cor["positive_rate"]< -0.5]
pos1
# %%
dth1 = cor[cor["cardiovasc_death_rate"]>0.5]
dth2 = cor[cor["cardiovasc_death_rate"]< -0.5]
dth2
# %%
nca1 = cor[cor["new_cases_smoothed_per_million"]>0.5]
nca2 = cor[cor["new_cases_smoothed_per_million"]< -0.5]
nca2
# %%
def newdata(dat1,dat2):
  lst = list(dat1.index)
  lst += list(dat2.index)
  ndat = data[lst].dropna().reset_index()
  ndat = ndat.drop(['index'],axis=1)
  return ndat
newdata(nca1,nca2)
# %%
newdata(dth1,dth2)
# %%
# model
from sklearn.model_selection import train_test_split
death = newdata(dth1,dth2)
y = death.iloc[:,0]
x = death.iloc[:,1:]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=2020)
X_train
# %%
clf1 = LinearRegression()
clf1.fit(X_train,y_train)
print(f'LinearRegression() train score:  {clf1.score(X_train,y_train):.3f}')
print(f'LinearRegression() test score:  {clf1.score(X_test,y_test):.3f}')

y_pred = clf1.predict(X_test)
# The coefficients
print('\nCoefficients: \n', clf1.coef_)
# The mean squared error
print('\nMean squared error: %.3f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('\nCoefficient of determination: %.3f'
      % r2_score(y_test, y_pred))

# %%
clf2 = RandomForestRegressor(max_depth=7)
clf2.fit(X_train,y_train)
print(f'RandomForestRegressor(max_depth=7) train score:  {clf2.score(X_train,y_train):.3f}')
print(f'RandomForestRegressor(max_depth=7) test score:  {clf2.score(X_test,y_test):.3f}')
# %%
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"/Library/bin/graphviz"
tree = export_graphviz(clf2.estimators_[0], out_file=None)
graph = pydotplus.graph_from_dot_data(tree) 
Image(graph.create_png())


# %%
