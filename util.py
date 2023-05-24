import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from scipy.stats import chi2_contingency
from scipy.stats import ks_2samp, entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import PowerTransformer

def read_file(folder_name:str):
    """
    Args:
        folder_name: Name of the folder where data is stored at
    Returns:
        data_dict: Dictonary of pandas dataframe
        - population, demographics, location, service, status 
    """

    #Defining Data path 
    data_path = os.path.join(os.getcwd(), folder_name)

    data_dict = {}

    #Reading the data
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            name = file_name.split('_')[-1].split('.')[0]
            print(f'Reading file: {file_name}')
            data_dict[name] = pd.read_excel(file_path)

    return data_dict

def data_summary(data):
    #Checking for Null Values

    for k, v in data.items():
        #Counting the number of missinf values
        null_count = data[k].isnull().sum().values

        #Retrieve the Index of the columns having missing values
        cond_idx = np.where(null_count > 0)[0]

        #Any exists, then idenfity which column
        if len(cond_idx) > 0:
            columns = data[k].columns[cond_idx]
            print(f'Missing Values Present in columns {columns}')
            
        else:
            print(f'There are no Missing Values present in Table: {k}')


    for k,v in data.items():
        try:
            isunique = data[k]['Customer ID'].nunique()
            print(f'Total number of rows are {len(data[k])} in Table {k}')
            print(f'Unique CusomterIDs are {isunique} in Table {k}')
        except:
            continue

def plot_binary_corr(binary_data):
    from sklearn.metrics import matthews_corrcoef

    cols = binary_data.columns
    size = len(cols)

    #Generate empty matrix
    corr_matrix = np.zeros((size, size))

    #Computing Matthews Correlation
    for i in range(size):
        for j in range(size):
            if binary_data[cols[i]].std() * binary_data[cols[j]].std() == 0:  # Avoid division by zero
                corr = 0
            else:
                corr = matthews_corrcoef(binary_data[cols[i]], binary_data[cols[j]])
            corr_matrix[i, j] = corr

    # Convert the numpy matrix to a pandas DataFrame for better readability
    corr_matrix = pd.DataFrame(corr_matrix, columns=cols, index=cols)

    
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Matrix Heatmap")
    plt.show()

    return corr_matrix


def calculate_dissimilarity(data, columns, target, transformation=None):
    dissimilarities = {
        "ks": [],
        "js": [],
        "emd": []
    }

    for col in columns:
        if transformation:
            if transformation == 'yeo-johnson':
                pt = PowerTransformer(method='yeo-johnson')
                data[col] = pt.fit_transform(data[col].values.reshape(-1, 1))

            elif transformation == 'log':
                # Adding 1 before applying log transformation to handle zero values
                data[col] = np.log1p(data[col])

        churned = data[data[target] == 1][col]
        not_churned = data[data[target] == 0][col]

        # Convert data into probability distribution for Jensen-Shannon Divergence
        churned_hist = np.histogram(churned, bins=100, density=True)[0]
        not_churned_hist = np.histogram(not_churned, bins=100, density=True)[0]

        # Add a small constant to avoid division by zero errors
        churned_hist += 1e-10
        not_churned_hist += 1e-10

        # K-S Test
        ks_stat, ks_p = ks_2samp(churned, not_churned)
        dissimilarities["ks"].append(ks_stat)

        # Jensen-Shannon Divergence
        js_divergence = jensenshannon(churned_hist, not_churned_hist)
        dissimilarities["js"].append(js_divergence)

        # Earth Mover's Distance (Wasserstein distance)
        emd = wasserstein_distance(churned, not_churned)
        dissimilarities["emd"].append(emd)

    return pd.DataFrame(dissimilarities, index=columns)


def calculate_categorical_association(df, columns, target):
    associations = {"chi2_pvalue": [], "cramer_v": []}
    
    for col in columns:
        cross_tab = pd.crosstab(df[col], df[target])
        chi2, p, dof, ex = chi2_contingency(cross_tab)
        associations["chi2_pvalue"].append(p)

        # Calculating Cramér's V
        n = cross_tab.sum().sum()
        phi2 = chi2/n
        r,k = cross_tab.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        cramers_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        associations["cramer_v"].append(cramers_v)

    return pd.DataFrame(associations, index=columns)
