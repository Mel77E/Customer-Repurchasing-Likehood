# Project Summary: Predicting Customer Repurchase Likelihood


## Project Objective

The goal of this project was to build a machine learning model to predict the likelihood of a customer making a repeat purchase. The "repurchased" column served as the target variable, indicating whether a customer made additional purchases after their initial one.


## Key Features Used

#####  number_of_customer_support_cases: Number of times a customer contacted customer support.

##### is_newsletter_subscriber: Indicates whether a customer subscribed to the newsletter (1-yes, 0-no).

##### apparel_purchased: Indicates if a customer also purchased apparel (1-yes, 0-no).

##### accessories_purchased: Indicates if a customer also purchased accessories (1-yes, 0-no).

##### ecom_limited_edition_purchased: Indicates if a customer purchased limited edition products (1-yes, 0-no).

##### is_subscribed_to_cyclon: Indicates if a customer subscribed to a subscription-based running shoes product (1-yes, 0-no).

##### total_revenue: The total amount of money a customer spent.

##### quantity_rma: Number of products a customer returned after purchase.

##### days_since_last_purchase: Number of days since the customer last made a purchase.

##### country: Customer's country, which was OneHotEncoded for the model.


## Importing libraires


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv("data_science_internship_assignment_input_data.csv")
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uid</th>
      <th>number_of_customer_support_cases</th>
      <th>is_newsletter_subscriber</th>
      <th>apparel_purchased</th>
      <th>accessories_purchased</th>
      <th>ecom_limited_edition_purchased</th>
      <th>is_subscribed_to_cyclon</th>
      <th>total_revenue</th>
      <th>quantity_rma</th>
      <th>country</th>
      <th>days_since_last_purchase</th>
      <th>repurchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>282</td>
      <td>0</td>
      <td>United States</td>
      <td>46</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>270</td>
      <td>0</td>
      <td>United States</td>
      <td>29</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>581</td>
      <td>0</td>
      <td>Australia</td>
      <td>97</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>107</td>
      <td>0</td>
      <td>Japan</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>97</td>
      <td>0</td>
      <td>United States</td>
      <td>219</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9943</th>
      <td>99976</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>388</td>
      <td>0</td>
      <td>Other Ecom EU</td>
      <td>777</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9944</th>
      <td>99977</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>692</td>
      <td>0</td>
      <td>United States</td>
      <td>124</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9945</th>
      <td>99987</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>436</td>
      <td>2</td>
      <td>United Kingdom</td>
      <td>306</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9946</th>
      <td>99989</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>308</td>
      <td>0</td>
      <td>United States</td>
      <td>18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9947</th>
      <td>99995</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>231</td>
      <td>0</td>
      <td>United Kingdom</td>
      <td>23</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>9948 rows × 12 columns</p>
</div>




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uid</th>
      <th>number_of_customer_support_cases</th>
      <th>is_newsletter_subscriber</th>
      <th>apparel_purchased</th>
      <th>accessories_purchased</th>
      <th>ecom_limited_edition_purchased</th>
      <th>is_subscribed_to_cyclon</th>
      <th>total_revenue</th>
      <th>quantity_rma</th>
      <th>country</th>
      <th>days_since_last_purchase</th>
      <th>repurchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>282</td>
      <td>0</td>
      <td>United States</td>
      <td>46</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>270</td>
      <td>0</td>
      <td>United States</td>
      <td>29</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>581</td>
      <td>0</td>
      <td>Australia</td>
      <td>97</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>107</td>
      <td>0</td>
      <td>Japan</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>97</td>
      <td>0</td>
      <td>United States</td>
      <td>219</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uid</th>
      <th>number_of_customer_support_cases</th>
      <th>is_newsletter_subscriber</th>
      <th>apparel_purchased</th>
      <th>accessories_purchased</th>
      <th>ecom_limited_edition_purchased</th>
      <th>is_subscribed_to_cyclon</th>
      <th>total_revenue</th>
      <th>quantity_rma</th>
      <th>country</th>
      <th>days_since_last_purchase</th>
      <th>repurchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9943</th>
      <td>99976</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>388</td>
      <td>0</td>
      <td>Other Ecom EU</td>
      <td>777</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9944</th>
      <td>99977</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>692</td>
      <td>0</td>
      <td>United States</td>
      <td>124</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9945</th>
      <td>99987</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>436</td>
      <td>2</td>
      <td>United Kingdom</td>
      <td>306</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9946</th>
      <td>99989</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>308</td>
      <td>0</td>
      <td>United States</td>
      <td>18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9947</th>
      <td>99995</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>231</td>
      <td>0</td>
      <td>United Kingdom</td>
      <td>23</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Data Preprocessing

### 1. Handinling Missing Values 



```python
print(df.isnull().sum())
```

    uid                                  0
    number_of_customer_support_cases     0
    is_newsletter_subscriber             0
    apparel_purchased                    0
    accessories_purchased                0
    ecom_limited_edition_purchased       0
    is_subscribed_to_cyclon              0
    total_revenue                        0
    quantity_rma                         0
    country                             12
    days_since_last_purchase             0
    repurchased                          0
    dtype: int64
    

country column has 12 missing values, while the rest of the columns have none. Nex step filling the missing values.


```python
df.fillna(value=0, inplace=True)
```


```python
print(df.isnull().sum())
```

    uid                                 0
    number_of_customer_support_cases    0
    is_newsletter_subscriber            0
    apparel_purchased                   0
    accessories_purchased               0
    ecom_limited_edition_purchased      0
    is_subscribed_to_cyclon             0
    total_revenue                       0
    quantity_rma                        0
    country                             0
    days_since_last_purchase            0
    repurchased                         0
    dtype: int64
    

Now there is no missing values for every column.

### 2. Exploratory Data Analysis (EDA) 


```python
df.dtypes
```




    uid                                  int64
    number_of_customer_support_cases     int64
    is_newsletter_subscriber             int64
    apparel_purchased                    int64
    accessories_purchased                int64
    ecom_limited_edition_purchased       int64
    is_subscribed_to_cyclon              int64
    total_revenue                        int64
    quantity_rma                         int64
    country                             object
    days_since_last_purchase             int64
    repurchased                          int64
    dtype: object



There are object data types present so later on we would conver them to numberical values for ML training model.


```python
 #Unique values in categorical columns

print(df['country'].unique())
```

    ['United States' 'Australia' 'Japan' 'Germany' 'Switzerland' 'Canada'
     'BeNe' 'Brazil' 'United Kingdom' 'France' 'Other Ecom EU' 'Sweden'
     'Austria' 'Norway' 'Distributors EU' 'Distributors APAC' 'Spain' 'Italy'
     'China' 0 'Denmark' 'Hong Kong' 'Distributors ROW' 'Other Ecom APAC'
     'Other Ecom ROW']
    


```python
print(df['country'].value_counts())
```

    country
    United States        6735
    Germany              1101
    United Kingdom        603
    Switzerland           331
    Canada                250
    Austria               169
    Japan                 152
    Brazil                108
    France                 92
    Australia              90
    Spain                  56
    BeNe                   54
    Italy                  46
    Distributors EU        31
    Distributors APAC      30
    China                  22
    Hong Kong              16
    Norway                 13
    0                      12
    Sweden                 11
    Other Ecom EU           9
    Denmark                 9
    Distributors ROW        4
    Other Ecom APAC         2
    Other Ecom ROW          2
    Name: count, dtype: int64
    

Top Countries by Customer Count:
United States: 6735 customers
Germany: 1101 customers
United Kingdom: 603 customers
Switzerland: 331 customers
Canada: 250 customers


```python
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x='country', y='total_revenue', data=df, estimator=sum, ci=None)
plt.title('Total Revenue by Country')
plt.xlabel('Country')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.show()
```

    C:\Users\MELDA\AppData\Local\Temp\ipykernel_10448\1849464894.py:4: FutureWarning: 
    
    The `ci` parameter is deprecated. Use `errorbar=None` for the same effect.
    
      sns.barplot(x='country', y='total_revenue', data=df, estimator=sum, ci=None)
    


    
![png](output_21_1.png)
    


The observation that the USA, Germany, and the UK have higher total revenue compared to other countries, with the USA leading and Germany having nearly half the revenue of the USA, can provide several insights and implications for customer repurchasing behavior.

#### Key Insights:

- USA: Highest total revenue, indicating a large and potentially highly engaged customer base.
- Germany: Significant total revenue but nearly half that of the USA, suggesting a smaller or less active customer base compared to the USA.
- UK: High total revenue, indicating a strong market presence.
- Other Countries: Lower total revenue, implying less market penetration or engagement.

#### Implications for Repurchasing Behavior:

#####  USA:

High total revenue suggests a large, engaged customer base with potentially high repurchase rates.
Marketing efforts can focus on maintaining high engagement through personalized offers and loyalty programs.

##### Germany:

Lower total revenue compared to the USA, despite being a high-revenue country, may indicate opportunities to boost repurchasing.
Strategies could include enhancing customer experience, addressing potential barriers to repurchasing, and offering targeted promotions.

##### UK:

Strong total revenue indicates a valuable market.
Similar strategies to the USA can be applied, focusing on sustaining customer engagement and exploring new growth opportunities.

##### Other Countries:

Lower total revenue implies limited market penetration or engagement.
Efforts could focus on market expansion, understanding local preferences, and improving product offerings.

### Investigating Patterns via Plots


```python
plt.figure(figsize=(10, 6))
plt.scatter(df['days_since_last_purchase'], df['total_revenue'], alpha=0.5)
plt.title('Scatter Plot of Total Revenue vs. Days Since Last Purchase')
plt.xlabel('Days Since Last Purchase')
plt.ylabel('Total Revenue')
plt.ylim(0, 50000)  
plt.show()
```


    
![png](output_24_0.png)
    


This scatter plot helps you visually explore the relationship between the time since a customer's last purchase and the total revenue they've generated. We can observe some outlayers starting from 20000 and 50000.There is not a strong relationship between customer's last purchases an the total revenue.


## Correlation Analysis:

#### Based on Total Revenue


```python
correlation = df['days_since_last_purchase'].corr(df['total_revenue'])
print(f"Correlation coefficient between days_since_last_purchase and total_revenue: {correlation}")
```

    Correlation coefficient between days_since_last_purchase and total_revenue: -0.07072333933490267
    

The correlation coefficient of -0.0707 indicates a very weak negative relationship between days_since_last_purchase and total_revenue. This means that, generally, as the number of days since the last purchase increases, the total revenue slightly decreases, but the relationship is very weak and almost negligible.


```python
correlation = df['ecom_limited_edition_purchased'].corr(df['total_revenue'])
print(f"Correlation coefficient between ecom_limited_edition_purchased and total_revenue: {correlation}")
```

    Correlation coefficient between ecom_limited_edition_purchased and total_revenue: 0.156634672550313
    

The correlation coefficient of 0.157 suggests a weak positive relationship between the variables. This means that as ecom_limited_edition_purchased increases, total_revenue tends to increase slightly, but the relationship is not very strong.


```python
correlation = df['is_subscribed_to_cyclon'].corr(df['total_revenue'])
print(f"Correlation coefficient between is_subscribed_to_cyclon and total_revenue: {correlation}")
```

    Correlation coefficient between is_subscribed_to_cyclon and total_revenue: 0.006580243921779357
    

A correlation coefficient of 0.007 suggests an extremely weak positive relationship between the variables. This means that there is almost no linear relationship between whether a customer is subscribed to Cyclon and the total revenue they generate.

#### Based on repurchased


```python
correlation = df['ecom_limited_edition_purchased'].corr(df['repurchased'])
print(f"Correlation coefficient between ecom_limited_edition_purchased and repurchased: {correlation}")
```

    Correlation coefficient between ecom_limited_edition_purchased and repurchased: 0.12546612132364088
    

The correlation coefficient of 0.12546612132364088 suggests a weak positive relationship between ecom_limited_edition_purchased and repurchased. While there is some positive association, it is not strong enough to be considered a reliable predictor of repurchase behavior on its own.


```python
correlation = df['days_since_last_purchase'].corr(df['repurchased'])
print(f"Correlation coefficient between days_since_last_purchase and repurchased: {correlation}")
```

    Correlation coefficient between days_since_last_purchase and repurchased: -0.2005393424384467
    

The correlation coefficient of -0.2005393424384467 suggests a weak negative relationship between days_since_last_purchase and repurchased. While there is some negative association, it is not strong enough to be considered a reliable predictor of repurchase behavior on its own.


```python
correlation = df['is_subscribed_to_cyclon'].corr(df['repurchased'])
print(f"Correlation coefficient between is_subscribed_to_cyclon and repurchased: {correlation}")
```

    Correlation coefficient between is_subscribed_to_cyclon and repurchased: 0.014372790328953272
    

The correlation coefficient of 0.014372790328953272 suggests a very weak positive relationship between is_subscribed_to_cyclon and repurchased.is value is close to zero, indicating that there is essentially no meaningful linear relationship between these two variables.

Lets test our hypothesis that is subscribers tend to be more loyal customers to the brand (is_newsletter_subscriber).

## Hypothesis Testing

Hypothesis Testing Framework
Null Hypothesis (H0): There is no difference in repurchase rates between newsletter subscribers and non-subscribers.
Alternative Hypothesis (H1): Newsletter subscribers have a higher repurchase rate compared to non-subscribers.


```python
# Creating a contingency table

contingency_table = pd.crosstab(df['is_newsletter_subscriber'], df['repurchased'])

# Displaying the contingency table

print(contingency_table)
```

    repurchased                  0     1
    is_newsletter_subscriber            
    0                         3098  2352
    1                         1943  2555
    

####  The table just above represents the counts of individuals based on their newsletter subscription status:


#####  0:     Indicates individuals who are not newsletter subscribers.

#####  1:     Indicates individuals who are newsletter subscribers.

##### 3098: Individuals who are not newsletter subscribers (is_newsletter_subscriber = 0) and also not repurchased (repurchased = 0).

##### 2352: Individuals who are not newsletter subscribers (is_newsletter_subscriber = 0) and have repurchased (repurchased = 1).

##### 1943: Individuals who are newsletter subscribers (is_newsletter_subscriber = 1) and have not repurchased (repurchased = 0).

##### 2555: Individuals who are newsletter subscribers (is_newsletter_subscriber = 1) and have repurchased (repurchased = 1).


```python
from scipy.stats import chi2_contingency

# Performing the chi-square test

chi2, p, _, _ = chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2}")
print(f"P-Value: {p}")
```

    Chi-Square Statistic: 183.06478909927932
    P-Value: 1.0382046902962443e-41
    

- A very small p-value (typically less than 0.05) suggests strong evidence against the null hypothesis. In this case, the extremely small p-value (1.0382046902962443e-41) indicates that there is a significant association between newsletter subscription status and repurchase behavior.

- Based on this result, the null hypothesis (which states no association) would be rejected and concluded that there is indeed a statistically significant association between being a newsletter subscriber and the likelihood of repurchasing.

Before other steps and model predictions, we must transform the string data which are country names with encoder.

##  Encoding Categorical Variables


```python
df = pd.read_csv('data_science_internship_assignment_input_data.csv') 

# Encoding categorical variables using LabelEncoder
label_encoder = LabelEncoder()

# Encoding  'country'
df['country_encoded'] = label_encoder.fit_transform(df['country'])
df.drop('country', axis=1, inplace=True)

# Encoding 'is_subscribed_to_cyclon'
df['is_subscribed_to_cyclon'] = label_encoder.fit_transform(df['is_subscribed_to_cyclon'])

# Convert numerical columns to integers
df['total_revenue'] = df['total_revenue'].astype(int)
df['quantity_rma'] = df['quantity_rma'].astype(int)
df['days_since_last_purchase'] = df['days_since_last_purchase'].astype(int)
df['number_of_customer_support_cases'] = df['number_of_customer_support_cases'].astype(int)

# Converting all remaining float columns to integers

float_cols = df.select_dtypes(include=['float']).columns
df[float_cols] = df[float_cols].astype(int)

# Verifying the changes

print(df.dtypes)
print(df.head())
```

    uid                                 int64
    number_of_customer_support_cases    int32
    is_newsletter_subscriber            int64
    apparel_purchased                   int64
    accessories_purchased               int64
    ecom_limited_edition_purchased      int64
    is_subscribed_to_cyclon             int64
    total_revenue                       int32
    quantity_rma                        int32
    days_since_last_purchase            int32
    repurchased                         int64
    country_encoded                     int32
    dtype: object
       uid  number_of_customer_support_cases  is_newsletter_subscriber  \
    0    5                                 0                         1   
    1   16                                 0                         1   
    2   37                                 0                         0   
    3   55                                 0                         1   
    4   80                                 0                         0   
    
       apparel_purchased  accessories_purchased  ecom_limited_edition_purchased  \
    0                  0                      0                               0   
    1                  0                      0                               0   
    2                  1                      1                               0   
    3                  0                      0                               0   
    4                  0                      0                               0   
    
       is_subscribed_to_cyclon  total_revenue  quantity_rma  \
    0                        0            282             0   
    1                        0            270             0   
    2                        0            581             0   
    3                        0            107             0   
    4                        0             97             0   
    
       days_since_last_purchase  repurchased  country_encoded  
    0                        46            1               23  
    1                        29            1               23  
    2                        97            1                0  
    3                       218            0               14  
    4                       219            0               23  
    


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uid</th>
      <th>number_of_customer_support_cases</th>
      <th>is_newsletter_subscriber</th>
      <th>apparel_purchased</th>
      <th>accessories_purchased</th>
      <th>ecom_limited_edition_purchased</th>
      <th>is_subscribed_to_cyclon</th>
      <th>total_revenue</th>
      <th>quantity_rma</th>
      <th>days_since_last_purchase</th>
      <th>repurchased</th>
      <th>country_encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>282</td>
      <td>0</td>
      <td>46</td>
      <td>1</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>270</td>
      <td>0</td>
      <td>29</td>
      <td>1</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>581</td>
      <td>0</td>
      <td>97</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>107</td>
      <td>0</td>
      <td>218</td>
      <td>0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>97</td>
      <td>0</td>
      <td>219</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9943</th>
      <td>99976</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>388</td>
      <td>0</td>
      <td>777</td>
      <td>1</td>
      <td>17</td>
    </tr>
    <tr>
      <th>9944</th>
      <td>99977</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>692</td>
      <td>0</td>
      <td>124</td>
      <td>1</td>
      <td>23</td>
    </tr>
    <tr>
      <th>9945</th>
      <td>99987</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>436</td>
      <td>2</td>
      <td>306</td>
      <td>1</td>
      <td>22</td>
    </tr>
    <tr>
      <th>9946</th>
      <td>99989</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>308</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>23</td>
    </tr>
    <tr>
      <th>9947</th>
      <td>99995</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>231</td>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
<p>9948 rows × 12 columns</p>
</div>



checking the columns as numbers.


```python
# Plot distribution of days since last purchase

plt.figure(figsize=(8, 6))
sns.histplot(df['days_since_last_purchase'], kde=True)
plt.title('Distribution of Days Since Last Purchase')
plt.xlabel('Days Since Last Purchase')
plt.ylabel('Frequency')
plt.show()
```

    C:\Users\MELDA\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](output_54_1.png)
    


- From the histogram above, there is a high long tail and is not perfectly normally distributed. This suggests a skewed distribution where most customers make purchases relatively frequently, but some have longer intervals between purchases.

- After 500 days since the last purchase, there is a significant drop in the number of customers who make purchases. This could indicate a cutoff point where customer activity declines sharply, potentially suggesting a customer retention challenge after a certain period.


```python
repurchased = np.random.randint(0, 2, size=1000) 

# Calculating mode, median, and standard deviation

mode = pd.Series(repurchased).mode()[0]
median = np.median(repurchased)
std_dev = np.std(repurchased)

print(f"Mode: {mode}")
print(f"Median: {median}")
print(f"Standard Deviation: {std_dev}")
```

    Mode: 0
    Median: 0.0
    Standard Deviation: 0.49959983987187184
    

##### Mode: 0: 
This indicates that the most common value in the dataset is 0, which suggests that in this simulated dataset, more customers did not repurchase (represented as 0).

##### Median: 0.0:
The median value is 0.0, which means that half of the customers in the dataset did not repurchase.

##### Standard Deviation: 0.4998999899979995:
The standard deviation is approximately 0.5, indicating a moderate spread or variability in the repurchase status data. This means that while a majority did not repurchase (0), there is some variation around this central tendency.


```python
plt.figure(figsize=(8, 6))
sns.countplot(x='repurchased', data=df)
plt.title('Count Plot of Repurchased')
plt.xlabel('Repurchased')
plt.ylabel('Count')
plt.show()
```


    
![png](output_58_0.png)
    


- The plot indicates that the number of customers who repurchased is roughly equal between subscribers and non-subscribers. 
- This observation might suggest that subscription status (being a subscriber or non-subscriber) does not strongly influence whether customers make repeat purchases.


```python
corr_matrix = df.corr()

# Creating a heatmap with scores

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='PiYG', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix Heatmap with Scores')
plt.show()
```


    
![png](output_60_0.png)
    


## Model Training and Evaluation:


```python
X = df.drop(columns=['uid', 'repurchased'])  # Droping the unique identifier and target variable
y = df['repurchased']
```

##### Spliting data into training and testing sets: 


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

##### Training the model on the training set and evaluating on the testing set using metrics like accuracy, precision, recall, and F1-score: 


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Initializing and training  a Logistic Regression model

model = LogisticRegression(max_iter=1000, C=0.001, penalty='l2', solver='liblinear')
model.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(C=0.001, max_iter=1000, solver=&#x27;liblinear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(C=0.001, max_iter=1000, solver=&#x27;liblinear&#x27;)</pre></div></div></div></div></div>



##### Model Performance Metrics:


```python
# Predicting on test set and evaluation

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.86      0.86      0.86       998
               1       0.86      0.85      0.86       992
    
        accuracy                           0.86      1990
       macro avg       0.86      0.86      0.86      1990
    weighted avg       0.86      0.86      0.86      1990
    
    ROC AUC Score: 0.8592830822936195
    

- Class 0 (not repurchased)  indicates Class 1 (repurchased) of both class of customers.This model achieves balanced precision and recall scores across both classes (0 and 1), indicating effective classification performance.

- The F1-score, which balances precision and recall, is also consistent at 0.86 for both classes. The ROC AUC score of 0.859 suggests that the model is reasonably good at distinguishing between customers who repurchase and those who do not.
- Overall, these metrics collectively demonstrate that the model is performing well in predicting customer repurchases based on the provided data.


```python
from sklearn.metrics import log_loss

logloss = log_loss(y_test, model.predict_proba(X_test))
print("Log Loss:", logloss)
```

    Log Loss: 0.3270464840014065
    

A log loss of 0.327 indicates that your model is making good probability predictions, as it is relatively low. Lower values are better, and values close to 0 are ideal.


```python
customer_probabilities = model.predict_proba(X)[:, 1]

# Creating a DataFrame with customer IDs and their probability scores

customer_likelihood_df = pd.DataFrame({
    'uid': df['uid'],
    'repurchase_probability': customer_probabilities
})

# Displaying the DataFrame

print(customer_likelihood_df.head())
```

       uid  repurchase_probability
    0    5                0.614168
    1   16                0.584798
    2   37                0.998786
    3   55                0.295396
    4   80                0.110692
    

##### Each customer (identified by their unique ID) making a repeat purchase. Here's what the provided example data means:

-  uid 5 has  a  61.42% 
-  uid 16 has a 58.48% 
-  uid 37 has a 99.88%
-  uid 55 has a 29.54% 
-  uid 80 has an 11.07% ,  probability of making a repeat purchase.

## Further Steps and Approaches Suggestions:

#### 1.Enhanced Feature Engineering:
Integrating more customer behavior metrics and product details (e.g., purchase frequency, product preferences).

#### 2.Advanced Modeling Techniques:
Consider ensemble methods or neural networks for better predictive power (e.g., RandomForest, Gradient Boosting, Deep Learning).

#### 3.Customer Segmentation:
Use clustering to tailor strategies for different customer groups (e.g., RFM analysis, K-means clustering).

#### 4. Predictive Analytics:
Implementing time series analysis to forecast repurchase probabilities (e.g., ARIMA, Prophet).

#### 5.Feedback Loop:
Establishing a continuous feedback mechanism to refine strategies (e.g., customer surveys, A/B testing).
