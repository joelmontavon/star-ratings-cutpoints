<h1>Stars Cutpoints using Clustering</h1>
<p>This tutorial will guide you thru how CMS calculates the cutpoints used in the Star Ratings program.</p>
<p>CMS publishes the data related to their Star Ratings on their <a href="https://www.cms.gov/Medicare/Prescription-Drug-Coverage/PrescriptionDrugCovGenIn/PerformanceData">Part C and D Performance Data</a> webpage. I have downloaded the <a href="https://www.cms.gov/files/zip/2022-star-ratings-data-table-oct-06-2021.zip">2022 Star Ratings Data Table</a> and unzipped it to a directory on my computer.</p>
<p>Now, I need to read the excel file into a dataframe. I've also used a converter to remove some spaces after the contract IDs and renamed some of the columns.</p>
Downloads:
https://www.cms.gov/Medicare/Prescription-Drug-Coverage/PrescriptionDrugCovGenIn/PerformanceData


```python
import pandas as pd
import numpy as np
import os
import re

#update with path to downloaded file
my_path = r'C:\Users\joelm\Documents\projects\python\cutpoints' + os.sep

def try_number(str):
    try:
        return float(str)
    except:
        return np.nan

#read excel file; strip an whitespace from contract_id
stars = pd.read_csv(my_path + '2022 Star Ratings Data Table - Measure Data (Oct 06 2021).csv', skiprows=lambda x: x in [0,1,3], converters={0: lambda x: x.strip()})
#add and/or rename some column names
stars = stars.rename({0:'CONTRACT_ID',1:'Organization Type',2:'Contract Name',3:'Organization Marketing Name','Unnamed: 4':'Parent Organization'}, axis='columns')
stars = stars.rename(columns=lambda x: re.sub("\s+", "_", re.sub('[^A-Za-z0-9\s]+',' ', x.lower())))
stars
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
      <th>unnamed_0</th>
      <th>unnamed_1</th>
      <th>unnamed_2</th>
      <th>unnamed_3</th>
      <th>parent_organization</th>
      <th>c01_breast_cancer_screening</th>
      <th>c02_colorectal_cancer_screening</th>
      <th>c03_annual_flu_vaccine</th>
      <th>c04_monitoring_physical_activity</th>
      <th>c05_special_needs_plan_snp_care_management</th>
      <th>...</th>
      <th>d03_members_choosing_to_leave_the_plan</th>
      <th>d04_drug_plan_quality_improvement</th>
      <th>d05_rating_of_drug_plan</th>
      <th>d06_getting_needed_prescription_drugs</th>
      <th>d07_mpf_price_accuracy</th>
      <th>d08_medication_adherence_for_diabetes_medications</th>
      <th>d09_medication_adherence_for_hypertension_ras_antagonists_</th>
      <th>d10_medication_adherence_for_cholesterol_statins_</th>
      <th>d11_mtm_program_completion_rate_for_cmr</th>
      <th>d12_statin_use_in_persons_with_diabetes_supd_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E0654</td>
      <td>Employer/Union Only Direct Contract PDP</td>
      <td>IBT VOLUNTARY EMPLOYEE BENEFITS TRUST</td>
      <td>TEAMStar Medicare Part D Prescription Drug Pro...</td>
      <td>IBT Voluntary Employee Benefits Trust</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>...</td>
      <td>Not enough data available</td>
      <td>Medicare shows only a Star Rating for this topic</td>
      <td>83</td>
      <td>91</td>
      <td>Not enough data available</td>
      <td>85%</td>
      <td>89%</td>
      <td>87%</td>
      <td>61%</td>
      <td>81%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E3014</td>
      <td>Employer/Union Only Direct Contract PDP</td>
      <td>PSERS HOP PROGRAM</td>
      <td>Pennsylvania Public School Employees Retiremen...</td>
      <td>Commonwealth of PA Pub Schools Retirement System</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>...</td>
      <td>Not enough data available</td>
      <td>Medicare shows only a Star Rating for this topic</td>
      <td>88</td>
      <td>94</td>
      <td>Not enough data available</td>
      <td>90%</td>
      <td>92%</td>
      <td>90%</td>
      <td>68%</td>
      <td>83%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E4744</td>
      <td>Employer/Union Only Direct Contract PDP</td>
      <td>MODOT/MSHP MEDICAL AND LIFE INSURANCE PLAN</td>
      <td>MISSOURI DEPARTMENT OF TRANSPORTATION</td>
      <td>Missouri Highways and Transportation Commission</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>...</td>
      <td>Not enough data available</td>
      <td>Medicare shows only a Star Rating for this topic</td>
      <td>86</td>
      <td>93</td>
      <td>Not enough data available</td>
      <td>87%</td>
      <td>89%</td>
      <td>89%</td>
      <td>53%</td>
      <td>76%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>H0022</td>
      <td>Demo</td>
      <td>BUCKEYE COMMUNITY HEALTH PLAN, INC.</td>
      <td>Buckeye Health Plan - MyCare Ohio</td>
      <td>Centene Corporation</td>
      <td>No data available</td>
      <td>No data available</td>
      <td>No data available</td>
      <td>No data available</td>
      <td>No data available</td>
      <td>...</td>
      <td>No data available</td>
      <td>No data available</td>
      <td>No data available</td>
      <td>No data available</td>
      <td>No data available</td>
      <td>No data available</td>
      <td>No data available</td>
      <td>No data available</td>
      <td>No data available</td>
      <td>No data available</td>
    </tr>
    <tr>
      <th>4</th>
      <td>H0028</td>
      <td>Local CCP</td>
      <td>CHA HMO, INC.</td>
      <td>Humana</td>
      <td>Humana Inc.</td>
      <td>71%</td>
      <td>79%</td>
      <td>79%</td>
      <td>47%</td>
      <td>79%</td>
      <td>...</td>
      <td>14%</td>
      <td>Medicare shows only a Star Rating for this topic</td>
      <td>87</td>
      <td>90</td>
      <td>90</td>
      <td>84%</td>
      <td>88%</td>
      <td>87%</td>
      <td>85%</td>
      <td>84%</td>
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
      <th>845</th>
      <td>S8182</td>
      <td>PDP</td>
      <td>AMERIGROUP INSURANCE COMPANY</td>
      <td>Amerigroup</td>
      <td>Anthem Inc.</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>...</td>
      <td>28%</td>
      <td>Not enough data available</td>
      <td>79</td>
      <td>85</td>
      <td>71</td>
      <td>85%</td>
      <td>85%</td>
      <td>84%</td>
      <td>53%</td>
      <td>75%</td>
    </tr>
    <tr>
      <th>846</th>
      <td>S8677</td>
      <td>PDP</td>
      <td>PROVIDENCE HEALTH ASSURANCE</td>
      <td>Providence Health Assurance</td>
      <td>Providence Health &amp; Services</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>...</td>
      <td>Not enough data available</td>
      <td>Not enough data available</td>
      <td>Plan too small to be measured</td>
      <td>Plan too small to be measured</td>
      <td>Not enough data available</td>
      <td>Not enough data available</td>
      <td>Not enough data available</td>
      <td>Not enough data available</td>
      <td>Plan not required to report</td>
      <td>Not enough data available</td>
    </tr>
    <tr>
      <th>847</th>
      <td>S8841</td>
      <td>PDP</td>
      <td>OPTUM INSURANCE OF OHIO, INC.</td>
      <td>Optum Insurance of Ohio, Inc.</td>
      <td>UnitedHealth Group, Inc.</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>...</td>
      <td>Not enough data available</td>
      <td>Medicare shows only a Star Rating for this topic</td>
      <td>90</td>
      <td>92</td>
      <td>Not enough data available</td>
      <td>86%</td>
      <td>89%</td>
      <td>86%</td>
      <td>62%</td>
      <td>80%</td>
    </tr>
    <tr>
      <th>848</th>
      <td>S9325</td>
      <td>PDP</td>
      <td>PRESIDENTIAL LIFE INSURANCE COMPANY</td>
      <td>Exemplar Health</td>
      <td>Exemplar Health Benefits Administrator, LLC</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>...</td>
      <td>Not enough data available</td>
      <td>Plan too new to be measured</td>
      <td>Plan too new to be measured</td>
      <td>Plan too new to be measured</td>
      <td>Plan too new to be measured</td>
      <td>Plan too new to be measured</td>
      <td>Plan too new to be measured</td>
      <td>Plan too new to be measured</td>
      <td>Plan too new to be measured</td>
      <td>Plan too new to be measured</td>
    </tr>
    <tr>
      <th>849</th>
      <td>S9701</td>
      <td>PDP</td>
      <td>DEAN HEALTH INSURANCE, INC.</td>
      <td>Navitus MedicareRx</td>
      <td>SSM Health Care Corporation</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>Plan not required to report measure</td>
      <td>...</td>
      <td>Not enough data available</td>
      <td>Medicare shows only a Star Rating for this topic</td>
      <td>89</td>
      <td>93</td>
      <td>Not enough data available</td>
      <td>90%</td>
      <td>91%</td>
      <td>90%</td>
      <td>62%</td>
      <td>83%</td>
    </tr>
  </tbody>
</table>
<p>850 rows × 45 columns</p>
</div>




```python
#transform columns for measure rates to rows for measure and rate
stars = pd.melt(stars, id_vars=['contract_id', 'contract_name', 'organization_marketing_name', 'organization_type', 'parent_organization'], var_name='measure', value_name='rate')
#convert rate to number
stars['rate'] = stars['rate'].apply(try_number)
#create column for contract type
stars['contract_type'] = stars['contract_id'].apply(lambda x: 'MAPD' if x[0] == 'H' or x[0] == 'R' else 'PDP')
#drop any rows where the contract did not receive a rate
stars = stars.dropna()
#sort by measure and rate
stars = stars.sort_values(['measure','rate'])
print(stars)
```


```python
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

#create list to store data across measures
thresholds_data = []
#identify adherence measures
measures = [measure for measure in stars['measure'].unique() if "adherence" in measure]
#loop thru measures and contract types
for measure in measures:
    for contract_type in ['MAPD', 'PDP']:
        #filter data for appropriate measures and contract type
        measure_data = stars[(stars['measure'] == measure) & (stars['contract_type'] == contract_type)]
        #create 10 folds in data
        kf = KFold(n_splits=10, shuffle=True, random_state=999)
        #create list to capture thresholds for each sample
        samples_data = []
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
            sample_data = sample_data[['stars','rate']].groupby('stars').agg(['min','max']).reset_index()
            #rename the columns
            sample_data.columns = ['stars','min_rate','max_rate']
            #sort the data based upon the min_rate (because higher rates are better for adherence measures)
            #and assign label 1-5 stars
            sample_data = sample_data.sort_values('min_rate')
            sample_data['stars'] = [1,2,3,4,5]
            #append thresholds for each sample 
            samples_data.append(sample_data)
        #merge all dataframes for measure and contract type
        samples_data = pd.concat(samples_data)
        #group by stars and calculate the min across the samples
        samples_data = samples_data[['stars','min_rate']].groupby('stars').agg('min').reset_index()
        #add columns for measure and contract type
        samples_data['measure'] = measure
        samples_data['contract_type'] = contract_type
        #append thresholds for each measure and contract_type
        thresholds_data.append(samples_data)
```


```python
#merge dataframes for measures and contract types
thresholds_data = pd.concat(thresholds_data)
#pivot the data
thresholds_data.pivot(index=['contract_type','stars'], columns='measure')['min_rate']
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
      <th>d09_medication_adherence_for_hypertension_ras_antagonists_</th>
      <th>d10_medication_adherence_for_cholesterol_statins_</th>
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
      <td>0.76</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.83</td>
      <td>0.81</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.87</td>
      <td>0.86</td>
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
      <td>0.67</td>
      <td>0.48</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.77</td>
      <td>0.72</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.83</td>
      <td>0.81</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.87</td>
      <td>0.86</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.90</td>
      <td>0.90</td>
      <td>0.90</td>
    </tr>
  </tbody>
</table>
</div>


