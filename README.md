<style>p {font-size: 14px;}</style>
<h1>Star Ratings Cutpoints using Clustering</h1>
<p>This tutorial will guide you thru how CMS calculates the cutpoints used in the Star Ratings program.</p>
<p>CMS publishes the data related to their Star Ratings on their <a href="https://www.cms.gov/Medicare/Prescription-Drug-Coverage/PrescriptionDrugCovGenIn/PerformanceData">Part C and D Performance Data</a> webpage. I have downloaded the <a href="https://www.cms.gov/files/zip/2022-star-ratings-data-table-oct-06-2021.zip">2022 Star Ratings Data Table</a> and unzipped it to a directory on my computer.</p>

<p>Now, I need to read the CSV file into a dataframe. I've also used a converter to remove some spaces after the contract IDs and renamed some of the columns.</p>


```python
import pandas as pd
import numpy as np
import os
import re

#update with your own path; prefix your string for paths with r in windows
file = r'D:\projects\python\cutpoints\2022 Star Ratings Data Table - Measure Data (Oct 06 2021).csv'

#read the CSV file; converter strip an whitespace from contract_id
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
      <th>contract_id</th>
      <th>contract_name</th>
      <th>organization_marketing_name</th>
      <th>organization_type</th>
      <th>parent_organization</th>
      <th>contract_type</th>
      <th>measure</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>345</th>
      <td>H4091</td>
      <td>SIMPRA ADVANTAGE, INC.</td>
      <td>Simpra Advantage</td>
      <td>Local CCP</td>
      <td>Associated Care Ventures, Inc.</td>
      <td>MAPD</td>
      <td>c01_breast_cancer_screening</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>259</th>
      <td>H3291</td>
      <td>PRUITTHEALTH PREMIER, INC.</td>
      <td>PruittHealth Premier</td>
      <td>Local CCP</td>
      <td>UNICO Premier, LLC</td>
      <td>MAPD</td>
      <td>c01_breast_cancer_screening</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>111</th>
      <td>H1587</td>
      <td>ARKANSAS SUPERIOR SELECT, INC.</td>
      <td>Tribute Health Plans</td>
      <td>Local CCP</td>
      <td>Select Founders, LLC</td>
      <td>MAPD</td>
      <td>c01_breast_cancer_screening</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>179</th>
      <td>H2292</td>
      <td>OXFORD HEALTH INSURANCE, INC.</td>
      <td>UnitedHealthcare</td>
      <td>Local CCP</td>
      <td>UnitedHealth Group, Inc.</td>
      <td>MAPD</td>
      <td>c01_breast_cancer_screening</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>758</th>
      <td>H9952</td>
      <td>MEDICA HEALTH PLANS</td>
      <td>Medica</td>
      <td>Local CCP</td>
      <td>Medica Holding Company</td>
      <td>MAPD</td>
      <td>c01_breast_cancer_screening</td>
      <td>0.44</td>
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
    </tr>
    <tr>
      <th>33853</th>
      <td>H9207</td>
      <td>HEALTH PARTNERS PLANS, INC.</td>
      <td>Health Partners Medicare</td>
      <td>Local CCP</td>
      <td>Health Partners Plans, Inc.</td>
      <td>MAPD</td>
      <td>d12_statin_use_in_persons_with_diabetes_supd</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>33897</th>
      <td>H9834</td>
      <td>QUARTZ HEALTH PLAN MN CORPORATION</td>
      <td>Quartz Medicare Advantage</td>
      <td>Local CCP</td>
      <td>University of Wisconsin Hospitals and Clincs A...</td>
      <td>MAPD</td>
      <td>d12_statin_use_in_persons_with_diabetes_supd</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>33436</th>
      <td>H3467</td>
      <td>PROCARE ADVANTAGE, LLC</td>
      <td>ProCare Advantage</td>
      <td>Local CCP</td>
      <td>First Sacramento Capital Funding LLC</td>
      <td>MAPD</td>
      <td>d12_statin_use_in_persons_with_diabetes_supd</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>33278</th>
      <td>H1777</td>
      <td>CATHOLIC SPECIAL NEEDS PLAN, LLC</td>
      <td>ArchCare Advantage</td>
      <td>Local CCP</td>
      <td>Catholic Health Care System</td>
      <td>MAPD</td>
      <td>d12_statin_use_in_persons_with_diabetes_supd</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>33315</th>
      <td>H2185</td>
      <td>LIFEWORKS ADVANTAGE, LLC</td>
      <td>LifeWorks Advantage</td>
      <td>Local CCP</td>
      <td>MFA Lifeworks, LLC</td>
      <td>MAPD</td>
      <td>d12_statin_use_in_persons_with_diabetes_supd</td>
      <td>0.97</td>
    </tr>
  </tbody>
</table>
<p>17962 rows × 8 columns</p>
</div>



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


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-latest.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




<div>                            <div id="a533d2af-1f3d-42de-a021-240e79dbb167" class="plotly-graph-div" style="height:300px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("a533d2af-1f3d-42de-a021-240e79dbb167")) {                    Plotly.newPlot(                        "a533d2af-1f3d-42de-a021-240e79dbb167",                        [{"hovertemplate": "color=2<br>x=%{x}<br>y=%{y}<br>size=%{marker.size}<extra></extra>", "legendgroup": "2", "marker": {"color": "red", "size": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "sizemode": "area", "sizeref": 0.05, "symbol": "circle"}, "mode": "markers", "name": "2", "orientation": "v", "showlegend": true, "type": "scatter", "x": [0.75, 0.78, 0.75, 0.75, 0.77, 0.75, 0.75, 0.75, 0.75, 0.75], "xaxis": "x", "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "yaxis": "y"}, {"hovertemplate": "color=3<br>x=%{x}<br>y=%{y}<br>size=%{marker.size}<extra></extra>", "legendgroup": "3", "marker": {"color": "orange", "size": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "sizemode": "area", "sizeref": 0.05, "symbol": "circle"}, "mode": "markers", "name": "3", "orientation": "v", "showlegend": true, "type": "scatter", "x": [0.82, 0.84, 0.82, 0.82, 0.83, 0.83, 0.82, 0.82, 0.82, 0.83], "xaxis": "x", "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "yaxis": "y"}, {"hovertemplate": "color=4<br>x=%{x}<br>y=%{y}<br>size=%{marker.size}<extra></extra>", "legendgroup": "4", "marker": {"color": "green", "size": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "sizemode": "area", "sizeref": 0.05, "symbol": "circle"}, "mode": "markers", "name": "4", "orientation": "v", "showlegend": true, "type": "scatter", "x": [0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87], "xaxis": "x", "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "yaxis": "y"}, {"hovertemplate": "color=5<br>x=%{x}<br>y=%{y}<br>size=%{marker.size}<extra></extra>", "legendgroup": "5", "marker": {"color": "blue", "size": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "sizemode": "area", "sizeref": 0.05, "symbol": "circle"}, "mode": "markers", "name": "5", "orientation": "v", "showlegend": true, "type": "scatter", "x": [0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91], "xaxis": "x", "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "yaxis": "y"}],                        {"height": 300, "legend": {"itemsizing": "constant", "title": {"text": "color"}, "tracegroupgap": 0}, "margin": {"t": 60}, "plot_bgcolor": "white", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "showgrid": false, "tickformat": ".0%", "title": {"text": "Rate"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "showgrid": false, "showticklabels": false, "title": {"text": ""}, "zeroline": true, "zerolinecolor": "black", "zerolinewidth": 3}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('a533d2af-1f3d-42de-a021-240e79dbb167');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


<iframe id="github-iframe" src="" width="100%" height="300" frameBorder="0"></iframe>
<script>
    fetch('https://api.github.com/repos/joelmontavon/star-ratings-cutpoints/contents/plot.html')
        .then(function(response) {
            return response.json();
        }).then(function(data) {
			console.log(data);
            var iframe = document.getElementById('github-iframe');
            iframe.src = 'data:text/html;base64,' + encodeURIComponent(data['content']);
        });
</script>

<p>To create the final cutpoints, I use the minimum rate across all of the groups for each cluster.</p>


```python
#group by stars and calculate the min across the samples
cutpoints = samples_data.groupby('stars').agg('min').reset_index()
```

<p>Take a look at the output. It may not exactly match CMS' results because they take out some contracts that are impacted by a natural disaster. And, the sampling is random and may not yield the same results.</p>


```python
cutpoints[cutpoints['stars'] != 1]
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
      <th>stars</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.91</td>
    </tr>
  </tbody>
</table>
</div>



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
      <td>0.82</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.87</td>
      <td>0.87</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.90</td>
      <td>0.90</td>
      <td>0.91</td>
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


