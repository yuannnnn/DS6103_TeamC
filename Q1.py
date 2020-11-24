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
# %%
