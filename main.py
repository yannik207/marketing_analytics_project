#loading necessary packages
import pandas as pd
import numpy as np
#import statistics as st
#import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


# import dataset
# data is from kaggle https://www.kaggle.com/jackdaoud/marketing-data
df = pd.read_csv("/Users/yanniksa/Desktop/python_learning/marketing_data.csv")

# drop id column
del df['ID']


# I will run my code exploration with the CRISP DM processmodell
## 1. Data Understanding

#inspect dataset
print(df)
print(type(df))
print(df.shape)
print(df.info())

#There is a problem with the column of 'Income'
print(df.columns)
#first I have to delete the space in the header columns
df.columns = ['Year_Birth', 'Education', 'Marital_Status', 'Income',
       'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency', 'MntWines',
       'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2', 'Response', 'Complain', 'Country']


#delet dollar sign in Income and comma sign so it is possible to convert string to float
df.Income = df.Income.replace({"\,": ""}, regex = True)
df.Income = df.Income.replace({"\$": ""}, regex = True)
df.Income = df.Income.astype('float')
print(df.info())

pd.set_option("display.max.columns", None)
print(df.head())


print(df.describe())
print(df.describe(include = object))

# looking for NA's
print(df.isnull().values.any())
print(df.isnull().sum())

#checking for NA proportion
perc_missing = df.isnull().sum() * 100 / len(df)
print(perc_missing)

#calculate Income mean
Income_mean = df['Income'].mean()
print(Income_mean)

#delete NA with mean
df['Income'].fillna(value=df['Income'].mean(), inplace=True)

#checking wether there are any NA's left
print(df.isnull().values.any())

groupe_ed = df.groupby('Education')
print(groupe_ed)

for education, education_df in groupe_ed:
       print(education)
       print(education_df)

groupe_ed.get_group('Graduation')

groupe_ed.max('Income')


#groupe_ed.plot()
#plt.show()

## 2. Data Preperation
### Section 01 Explotary Data Analysis
## Are there any null values or outliers? How will you wrangle/handle them?

# filter for numeric values
df_numerics_only = df._get_numeric_data()
df_numerics_only.plot(subplots = True, kind = 'box', layout = (4, 6), patch_artist = True)


#delete customers which are born before 1900

df = df.loc[(df.Year_Birth > 1900)]
df['Year_Birth'].plot(kind = 'box')


## Are there any variables that warrant transformations?
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

## Are there any useful variables that you can engineer with the given data?
#calculate total kids at home
df['Dependents'] = df['Teenhome'] + df['Kidhome']

# calculate total amount spended
df['Mnt_total'] = df['MntFishProducts'] + df['MntMeatProducts'] + df['MntFruits'] + df['MntSweetProducts'] + df['MntWines'] + df['MntGoldProds']

# how much every houshold is spending on comparison to their income
df['share_of_wallet'] = (df['Mnt_total'] / 2) / df['Income'] * 100

# total accepted campaigns
df['Accepted_total'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']

## Do you notice any patterns or anomalies in the data? Can you plot them?

corr = df.corr()
ax = sns.clustermap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap='coolwarm',
    square=True
)



sns.lmplot(x = 'Income', y = 'Mnt_total', data = df[df['Income'] < 200000])
#plt.show()

print(df.isnull().sum())
### Section 02 Statistical Analysis
## What factors are significantly related to the number of store purchases?
#checking wether every campaign had a sginificant increase in number of store purchases
cmp1_acp = df[df['AcceptedCmp1'] == 1]
cmp1_non = df[df['AcceptedCmp1'] == 0]

result_cmp1 = stats.ttest_ind(cmp1_acp['NumStorePurchases'], cmp1_non['NumStorePurchases'])
print(result_cmp1)

cmp2_acp = df[df['AcceptedCmp2'] == 1]
cmp2_non = df[df['AcceptedCmp2'] == 0]

result_cmp2 = stats.ttest_ind(cmp2_acp['NumStorePurchases'], cmp2_non['NumStorePurchases'])
print(result_cmp2)

cmp3_acp = df[df['AcceptedCmp3'] == 1]
cmp3_non = df[df['AcceptedCmp3'] == 0]

result_cmp3 = stats.ttest_ind(cmp3_acp['NumStorePurchases'], cmp3_non['NumStorePurchases'])
print(result_cmp3)

cmp4_acp = df[df['AcceptedCmp4'] == 1]
cmp4_non = df[df['AcceptedCmp4'] == 0]

result_cmp4 = stats.ttest_ind(cmp4_acp['NumStorePurchases'], cmp4_non['NumStorePurchases'])
print(result_cmp4)

cmp5_acp = df[df['AcceptedCmp5'] == 1]
cmp5_non = df[df['AcceptedCmp5'] == 0]

result_cmp5 = stats.ttest_ind(cmp5_acp['NumStorePurchases'], cmp5_non['NumStorePurchases'])
print(result_cmp5)

del df['Dt_Customer']

# get categorical features and review number of unique values
cat = df.select_dtypes(exclude=np.number)
print("Number of unique values per categorical feature:\n", cat.nunique())

# use one hot encoder
enc = OneHotEncoder(sparse=False).fit(cat)
cat_encoded = pd.DataFrame(enc.transform(cat))
cat_encoded.columns = enc.get_feature_names(cat.columns)

# merge with numeric data
num = df.drop(columns=cat.columns)
df2 = pd.concat([cat_encoded, num], axis=1)

print(df2.isnull().sum())

# last three observations generated a NA drop
df2 = df2.fillna(df2.median())
print(df2)


# split the dataset
X = df2.drop(['NumStorePurchases'], axis = 1)
y = df2['NumStorePurchases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

lm = LinearRegression()
#train model
lm.fit(X_train, y_train)
lm.intercept_

coeff_df = pd.DataFrame(lm.coef_, X.columns)
print(coeff_df)

y_pred = lm.predict(X_test)
print(y_pred)

print(metrics.mean_absolute_error(y_test, y_pred))
print( metrics.mean_squared_error(y_test, y_pred))
print( np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred))

X2 = sm.add_constant(X_train)
model_stats = sm.OLS(y_train.values.reshape(-1,1), X2).fit()
print(model_stats.summary())

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif.round(1))


## Does US fare significantly better than the Rest of the World in terms of total purchases?
## Your supervisor insists that people who buy gold are more conservative. Therefore, people who spent an above average amount on gold in the last 2 years would have more in store purchases. Justify or refute this statement using an appropriate statistical test
## Fish has Omega 3 fatty acids which are good for the brain. Accordingly, do "Married PhD candidates" have a significant relation with amount spent on fish? What other factors are significantly related to amount spent on fish? (Hint: use your knowledge of interaction variables/effects)
## Is there a significant relationship between geographical regional and success of a campaign?

### Section 03 Data Vizualation
## Which marketing campaign is most successful?
## What does the average customer look like for this company?
## Which products are performing best?
## Which channels are underperforming?