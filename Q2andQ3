#%%
import os
import numpy as np
import pandas as pd
path = os.getcwd() 
filepath = os.path.join(path ,'owid-covid-data.csv')
data = pd.read_csv(filepath)
#%%
new_data=data[['location','new_cases','stringency_index']].dropna()
new_data=new_data.reset_index(drop=True)
# %%
import matplotlib.pyplot as plt
import seaborn as sns
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
