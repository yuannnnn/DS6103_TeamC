#%%[markdown]
#
# # Q4
# Whether death rate is influenced by aging degree? 
# For different countries, does bigger share of older people mean bigger death rate?

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
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
# %%
