# Employee Segmentation Using K-Means Clustering

## Overview

This project applies unsupervised machine learning to employee data to identify groups of employees with similar numerical characteristics.

The project uses **K-Means clustering** to divide employees into three clusters and visualizes the resulting employee segments using a scatter plot.

The main objective is to demonstrate a practical application of clustering for employee segmentation using Python and Scikit-learn.

## Dataset

The project uses the `employee_data.csv` dataset 

### Dataset Details

* Records: 3,000 employees
* Original features: 26
* Dataset type: Employee data
* File used by the notebook: `employee_data.csv`

The dataset contains information related to employee demographics, employment status, business units, job roles, performance, ratings, and other employee attributes.

### Additional Dataset Files


## Objective

The objective of this project is to:

* Prepare employee data for clustering
* Select numerical employee features
* Handle missing values
* Apply K-Means clustering
* Divide employees into three groups
* Visualize the identified clusters

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Google Colab / Jupyter Notebook

## Project Workflow

```text
Employee Dataset
       |
       v
Load CSV Data
       |
       v
Explore Dataset
       |
       v
Select Numerical Features
       |
       v
Remove Missing Values
       |
       v
Apply K-Means Clustering
       |
       v
Create 3 Employee Clusters
       |
       v
Visualize Clusters
```

## Data Preparation

The dataset is first loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv('employee_data.csv')
df.head()
```

The structure of the dataset is inspected using:

```python
df.info()
```

Only numerical columns are retained for clustering:

```python
df = df.select_dtypes(include=['int64','float64'])
```

Rows containing missing values are then removed:

```python
df = df.dropna()
```

The resulting dataset size is checked using:

```python
print(df.shape)
```

## K-Means Clustering

K-Means is used to group employees into three clusters.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0)

df['cluster'] = kmeans.fit_predict(df)
```

The model assigns each employee to one of three clusters:

* Cluster 0
* Cluster 1
* Cluster 2

The cluster assignment is stored in a new `cluster` column.

## Visualization

The employee clusters are visualized using Matplotlib.

```python
import matplotlib.pyplot as plt

plt.scatter(
    df.iloc[:,0],
    df.iloc[:,1],
    c=df['cluster']
)

plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title('Employee Clustering')
plt.show()
```

The visualization uses the first two numerical features of the processed dataset and colors the observations according to their assigned cluster.

## Project Structure

```text
Employee-Segmentation-KMeans/
│
├── employee_data.csv
├── employee_engagement_survey_data.csv
├── recruitment_data.csv
├── training_and_development_data.csv
├── employee_clustering.ipynb
└── README.md
```

If the datasets are not uploaded to GitHub, the dataset files can be excluded from the repository and the notebook can reference the original dataset source.

## Skills Demonstrated

* Data loading with Pandas
* Dataset inspection
* Numerical feature selection
* Missing-value handling
* Unsupervised machine learning
* K-Means clustering
* Cluster assignment
* Data visualization
* Matplotlib
* Scikit-learn

## Limitations

The current notebook performs clustering directly on the selected numerical features without additional feature scaling or detailed cluster interpretation.

The project demonstrates the implementation of K-Means clustering, but the current analysis does not establish what each cluster represents from a business perspective.

## Future Improvements

Possible improvements include:

* Feature scaling before K-Means clustering
* Testing different numbers of clusters
* Elbow method for selecting the number of clusters
* Silhouette score evaluation
* Cluster profiling and interpretation
* Comparing employee characteristics across clusters
* Using the employee engagement, recruitment, and training datasets for a broader HR analytics project

## Conclusion

This project demonstrates how K-Means clustering can be applied to employee data to discover groups based on numerical characteristics.

It provides a practical introduction to unsupervised learning, employee segmentation, preprocessing, clustering, and visualization using Python and Scikit-learn.

## Author

Nayanendhu CU


