<h1>Star Ratings Cutpoints using Clustering</h1>
<p>This tutorial will guide you thru how CMS calculates the cutpoints used in the Star Ratings program.</p>
<p>CMS publishes the data related to their Star Ratings on their <a href="https://www.cms.gov/Medicare/Prescription-Drug-Coverage/PrescriptionDrugCovGenIn/PerformanceData">Part C and D Performance Data</a> webpage. I have downloaded the <a href="https://www.cms.gov/files/zip/2022-star-ratings-data-table-oct-06-2021.zip">2022 Star Ratings Data Table</a> and unzipped it to a directory on my computer.</p>
<p>Now, I need to read the excel file into a dataframe. I've also used a converter to remove some spaces after the contract IDs and renamed some of the columns.</p>


```python
import pandas as pd
import numpy as np
import os
import re

#update with your own path; prefix your string for paths with r in windows
file = r'D:\projects\python\cutpoints\2022 Star Ratings Data Table - Measure Data (Oct 06 2021).csv'

#read excel file; converter strip an whitespace from contract_id
stars = pd.read_csv(file, skiprows=lambda x: x in [0,1,3], converters={'Unnamed: 0': lambda x: x.strip()})
#add and/or rename some column names
stars = stars.rename({'Unnamed: 0':'CONTRACT_ID','Unnamed: 1':'Organization Type','Unnamed: 2':'Contract Name','Unnamed: 3':'Organization Marketing Name','Unnamed: 4':'Parent Organization'}, axis='columns')
stars = stars.rename(columns=lambda x: re.sub("\s+", "_", re.sub('[^A-Za-z0-9\s]+',' ', x.lower()).strip()))
```

<p>I create a new column for the contract type. The contract type is determined by looking at the first character of the contract ID.</p>


```python
#create column for contract type
stars['contract_type'] = stars['contract_id'].apply(lambda x: 'MAPD' if x[0] == 'H' or x[0] == 'R' else 'PDP')
```

<p>Next, I need to transform and clean up the data a bit. I use the melt() function to reorient the data and create a column for the measure name and rate.</p>


```python
#transform columns for measure rates to rows for measure and rate
stars = pd.melt(stars, id_vars=['contract_id', 'contract_name', 'organization_marketing_name', 'organization_type', 'parent_organization', 'contract_type'], var_name='measure', value_name='rate')
```

<p>Then, I apply a function that attempts to convert the rates to numbers and drops contracts without a rate.</p>


```python
def try_number(str):
    if '%' in str:
        return float(str.strip('%'))/100
    elif not any(c.isalpha() for c in str):
        return float(str)
    else:
        return np.nan
    
#convert rate to number
stars['rate'] = stars['rate'].apply(try_number)
#drop any rows where the contract did not receive a rate
stars = stars.dropna()
#sort by measure and rate
stars = stars.sort_values(['measure','rate'])
```

<p>Take a look at the cleaned up data.</p>


```python
stars
```

    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    

<p>Now, the real work begins.</p>
<p>CMS using uses mean resampling to determine the cutpoints. They create 10 equal-sized groups and then apply the clustering algorithm 10 types leaving one group out each time.</p>
<p>For the purpose of this tutorial, I have limited the analysis to MAPD contracts and the Medication Adherence for Cholesterol (Statins) measure.</p>


```python
measure = 'd10_medication_adherence_for_cholesterol_statins'
contract_type = 'MAPD'

#filter data for appropriate measures and contract type
measure_data = stars[(stars['measure'] == measure) & (stars['contract_type'] == contract_type)]
```

<p>I use scikit-learn's KFolds to create the 10 groups.</p>


```python
from sklearn.model_selection import KFold

#create 10 folds in data; use the same random state to get the same results
kf = KFold(n_splits=10, shuffle=True, random_state=999)
```

<p>I use scikit-learn's KFolds to create the 10 groups and KMeans algorithm to create the 5 clusters. I loop over the groups and use the KMeans algorithm to create the 5 clusters. To do this, I fit the model to each sample's data and assign each contract in the sample to a cluster.</p>
<p>Since higher rates reflect better performance, I use the minimum rate for each cluster for the cutpoint. The clusters are not ordered so sort them based upon the minimum rate and assign a label 1-5 stars.</p>


```python
#turn off some of the warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans

#create dataframe to store data across samples
samples_data = pd.DataFrame()

#loop over folds
for sample in kf.split(measure_data):
    #create sample data for training with clusting
    sample_data = measure_data.iloc[sample[0]]
    #reshape data for clustering
    sample_data_x = sample_data['rate'].to_numpy().reshape(-1,1)
    #setup clusting with 5 clusters
    kmeans = KMeans(n_clusters=5, random_state=0)
    #fit model using sample data
    kmeans.fit(sample_data_x)
    #predict clusters using the sample data
    sample_data['stars'] = kmeans.predict(sample_data_x)
    #calculate the min and max for each cluster
    sample_data = sample_data[['stars','rate']].groupby('stars').agg(['min']).reset_index()
    #rename the columns
    sample_data.columns = ['stars','rate']
    #sort the data based upon the min_rate (because higher rates are better for adherence measures)
    #and assign label 1-5 stars
    sample_data = sample_data.sort_values('rate')
    sample_data['stars'] = [1,2,3,4,5]
    #concatenate cutpoints for each sample 
    samples_data = pd.concat([samples_data, sample_data])
```

<p>Using plotly, I plot the data data using a scatter plot to view all of the points.</p>


```python
from plotly.offline import init_notebook_mode, iplot
import plotly.express as px

init_notebook_mode(connected=True)

plot_data = samples_data[samples_data['stars'] != 1]
plot_data['x'] = plot_data['rate']
plot_data['y'] = 0
plot_data['color'] = plot_data['stars'].astype(str)
plot_data['size'] = 20

fig = px.scatter(plot_data, x="x", y="y", color="color", size="size", color_discrete_sequence=['red','orange','green','blue'])
fig.update_xaxes(showgrid=False, title="Rate", tickformat = '.0%')
fig.update_yaxes(showgrid=False, title="", zeroline=True, zerolinecolor='black', zerolinewidth=3, showticklabels=False)
fig.update_layout(height=300, plot_bgcolor='white')

fig.write_html(r'D:\projects\python\cutpoints\plot.html', full_html=False, include_plotlyjs='cdn')
fig.show()
```

<iframe src="https://github.com/joelmontavon/stars_cutpoints/blob/main/plot.html"></iframe>

<p>To create the final cutpoints, I use the minimum rate across all of the groups for each cluster.</p>


```python
#group by stars and calculate the min across the samples
cutpoints = samples_data.groupby('stars').agg('min').reset_index()
```

<p>Take a look at the output. It may not exactly match CMS' results because they take out some contracts that are impacted by a natural disaster. And, the sampling is random and may not yield the same results.</p>


```python
cutpoints[cutpoints['stars'] != 1]
```

    |    |   stars |   rate |
    |---:|--------:|-------:|
    |  1 |       2 |   0.75 |
    |  2 |       3 |   0.82 |
    |  3 |       4 |   0.86 |
    |  4 |       5 |   0.9  |
    

<p>I can now put this all together and loop over several measures and contract types. I use pivot() to reorient the data.</p>


```python
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans

#create dataframe to store data across measures
thresholds_data = pd.DataFrame()
#identify adherence measures
measures = [measure for measure in stars['measure'].unique() if "adherence" in measure]

#loop thru measures and contract types
for measure in measures:
    for contract_type in ['MAPD', 'PDP']:
        #filter data for appropriate measures and contract type
        measure_data = stars[(stars['measure'] == measure) & (stars['contract_type'] == contract_type)]
        #create 10 folds in data; use the same random state to get the same results
        kf = KFold(n_splits=10, shuffle=True, random_state=999)
        #create dataframe to capture thresholds for each sample
        samples_data = pd.DataFrame()
        #loop over folds
        for sample in kf.split(measure_data):
            #create sample data for training with clusting
            sample_data = measure_data.iloc[sample[0]]
            #reshape data for clustering
            sample_data_x = sample_data['rate'].to_numpy().reshape(-1,1)
            #setup clusting with 5 clusters
            kmeans = KMeans(n_clusters=5, random_state=0)
            #fit model using sample data
            kmeans.fit(sample_data_x)
            #predict clusters using the sample data
            sample_data['stars'] = kmeans.predict(sample_data_x)
            #calculate the min and max for each cluster
            sample_data = sample_data[['stars','rate']].groupby('stars').agg(['min']).reset_index()
            #rename the columns
            sample_data.columns = ['stars','rate']
            #sort the data based upon the min_rate (because higher rates are better for adherence measures)
            #and assign label 1-5 stars
            sample_data = sample_data.sort_values('rate')
            sample_data['stars'] = [1,2,3,4,5]
            #append thresholds for each sample 
            samples_data = pd.concat([samples_data, sample_data])
        #group by stars and calculate the min across the samples
        samples_data = samples_data[['stars','rate']].groupby('stars').agg('min').reset_index()
        #add columns for measure and contract type
        samples_data['measure'] = measure
        samples_data['contract_type'] = contract_type
        #append thresholds for each measure and contract_type
        thresholds_data = pd.concat([thresholds_data, samples_data])
        
#pivot the data
thresholds_data.pivot(index=['contract_type','stars'], columns='measure')['rate']
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
      <th>measure</th>
      <th>d08_medication_adherence_for_diabetes_medications</th>
      <th>d09_medication_adherence_for_hypertension_ras_antagonists</th>
      <th>d10_medication_adherence_for_cholesterol_statins</th>
    </tr>
    <tr>
      <th>contract_type</th>
      <th>stars</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">MAPD</th>
      <th>1</th>
      <td>0.67</td>
      <td>0.48</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.77</td>
      <td>0.72</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.83</td>
      <td>0.83</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.87</td>
      <td>0.87</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.90</td>
      <td>0.90</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">PDP</th>
      <th>1</th>
      <td>0.79</td>
      <td>0.83</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.82</td>
      <td>0.85</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.85</td>
      <td>0.87</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.87</td>
      <td>0.89</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.89</td>
      <td>0.91</td>
      <td>0.89</td>
    </tr>
  </tbody>
</table>
</div>


