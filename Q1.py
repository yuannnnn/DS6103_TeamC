#%%[markdown]
#
# # Q1
# Which countries are making progress against the pandemic? 
# Due to case counts are only meaningful if we also know how much testing a country does, what are the rate of positive tests of them?
# %%
import os
import numpy as np
import pandas as pd

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
import matplotlib.pyplot as plt 
import seaborn as sns

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
# # Q5
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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
print(f'RandomForestRegressor() train score:  {clf2.score(X_train,y_train):.3f}')
print(f'RandomForestRegressor() test score:  {clf2.score(X_test,y_test):.3f}')
# %%
from sklearn.tree import export_graphviz
from IPython.display import Image 
import pydotplus

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"/Library/bin/graphviz"
tree = export_graphviz(clf2.estimators_[0], out_file=None)
graph = pydotplus.graph_from_dot_data(tree) 
Image(graph.create_png())
