# Imports
# -----------------------------------------------------------
from scipy.sparse import data
import os
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import numpy as np
# Standardize/scale the dataset and apply PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# Import IsolationForest
from sklearn.ensemble import IsolationForest
import numpy.matlib
from sklearn.manifold import TSNE

sns.set_theme()

#st.set_page_config(layout="wide")
# -----------------------------------------------------------


#col1, col2, col3 = st.columns(3)

# Helper functions
# -----------------------------------------------------------
# Load data from external source

def file_selector(folder_path="./datasets"):
    filenames = os.listdir(folder_path)
    csv_files = list(filter(lambda f: f.endswith('.csv'), filenames))
    selected_filename = st.selectbox("Select a file with data from customer", csv_files)
    return os.path.join(folder_path, selected_filename)


filename = file_selector()
st.info("You Selected {}".format(filename))



@st.cache(allow_output_mutation=True)
def load_data(file):
    # Read sensor data from testing on 01_02_2021
    #df = pd.read_csv(file, decimal=".")

    df_test = pd.read_csv(file,skiprows=1, nrows=1)
    #df_test.columns=['test']
    if "." in str(df_test[df_test.columns[3]].values[0]):
        df = pd.read_csv(file, decimal=".")
    else:
        df = pd.read_csv(file, decimal=",")


    #df = df.replace(',', '.', regex = True).astype(float)
    # Renaming columns
    # df.columns=['time','Test_id','BL_x','BL_y','BL_z','BP_x','BP_y','BP_z','AP_x','AP_y','AP_z','AL_x','AL_y','AL_z']

    return df

def soft_clustering_weights(data, cluster_centres, **kwargs):

    """
    Function to calculate the weights from soft k-means
    data: Array of data. Features arranged across the columns with each row being a different data point
    cluster_centres: array of cluster centres. Input kmeans.cluster_centres_ directly.
    param: m - keyword argument, fuzziness of the clustering. Default 2
    """

    # Fuzziness parameter m>=1. Where m=1 => hard segmentation
    m = 2
    if 'm' in kwargs:
        m = kwargs['m']

    Nclusters = cluster_centres.shape[0]
    Ndp = data.shape[0]
    Nfeatures = data.shape[1]

    # Get distances from the cluster centres for each data point and each cluster
    EuclidDist = np.zeros((Ndp, Nclusters))
    for i in range(Nclusters):
        EuclidDist[:,i] = np.sum((data-np.matlib.repmat(cluster_centres[i], Ndp, 1))**2,axis=1)



    # Denominator of the weight from wikipedia:
    invWeight = EuclidDist**(2/(m-1))*np.matlib.repmat(np.sum((1./EuclidDist)**(2/(m-1)),axis=1).reshape(-1,1),1,Nclusters)
    Weight = 1./invWeight

    return Weight



# Read Data
df = load_data(filename)

def gen_repeating(s):
    """Generator: groups repeated elements in an iterable
    E.g.
        'abbccc' -> [('a', 0, 0), ('b', 1, 2), ('c', 3, 5)]
    """
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        yield (s[i], i, j-1)
        i = j

def plot_data(df, pca_result,selected_columns_names,n_clusters):

    #Interquartile Range anomaly detection

    # Calculate IQR for the 1st principal component (pc1)
    q1_pc1, q3_pc1 = df['pca_one'].quantile([0.25, 0.75])
    iqr_pc1 = q3_pc1 - q1_pc1
    # Calculate upper and lower bounds for outlier for pc1
    lower_pc1 = q1_pc1 - (1.5*iqr_pc1)
    upper_pc1 = q3_pc1 + (1.5*iqr_pc1)
    # Filter out the outliers from the pc1
    df['anomaly_pc1'] = ((df['pca_one']>upper_pc1) | (df['pca_one']<lower_pc1)).astype('int')
    # Calculate IQR for the 2nd principal component (pc2)
    q1_pc2, q3_pc2 = df['pca_two'].quantile([0.25, 0.75])
    iqr_pc2 = q3_pc2 - q1_pc2
    # Calculate upper and lower bounds for outlier for pc2
    lower_pc2 = q1_pc2 - (1.5*iqr_pc2)
    upper_pc2 = q3_pc2 + (1.5*iqr_pc2)
    # Filter out the outliers from the pc2
    df['anomaly_pc2'] = ((df['pca_two']>upper_pc2) | (df['pca_two']<lower_pc2)).astype('int')
    # Let's plot the outliers from pc1 on top of the x_dolny and see where they occured in the time series
    anomaly_1 = df[df['anomaly_pc1'] == 1] #anomaly
    #a = df[((df['anomaly_pc2'] == 1) | (df['anomaly_pc1'] == 1)).astype('int')]
    anomaly_2 = df[df['anomaly_pc2'] == 1] #anomaly



    ##Isolation forest anomaly detection

    #Assume that 13% of the entire data set are anomalies
    outliers_fraction = 0.13
    model = IsolationForest(contamination=outliers_fraction,random_state=42)
    model.fit(pca_result.values)
    df['anomaly_IsoFor'] = pd.Series(model.predict(pca_result.values),index=df.index)
    #df['anomaly_IsoFor'] = pd.Series( df['anomaly_temp'].values, index=df.index)

    anomaly_3 = df.loc[df['anomaly_IsoFor'] == -1] #anomaly
    #df['picked_anomaly']=a



    ## K_means based anomaly detection

    kmeans = KMeans(n_clusters, random_state=42)
    kmeans.fit(pca_result)
    labels = kmeans.predict(pca_result)
    unique_elements, counts_elements = np.unique(labels, return_counts=True)
    clusters = np.asarray((unique_elements, counts_elements))
    # Write a function that calculates distance between each point and the centroid of the closest cluster
    def getDistanceByPoint(data, model):
        """ Function that calculates the distance between a point and centroid of a cluster,
            returns the distances in pandas series"""
        distance = []
        for i in range(0,len(data)):
            Xa = np.array(data.loc[i])
            Xb = model.cluster_centers_[model.labels_[i]-1]
            distance.append(np.linalg.norm(Xa-Xb))
        return pd.Series(distance, index=data.index)
    # Assume that 13% of the entire data set are anomalies
    outliers_fraction = 0.13
    # get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
    distance = getDistanceByPoint(pca_result, kmeans)
    # number of observations that equate to the 13% of the entire data set
    number_of_outliers = int(outliers_fraction*len(distance))
    # Take the minimum of the largest 13% of the distances as the threshold
    threshold = distance.nlargest(number_of_outliers).min()
    # anomaly1 contain the anomaly result of the above method Cluster (0:normal, 1:anomaly)
    df['anomaly_K_Means'] = pd.Series((distance >= threshold).astype(int),index=df.index)

    anomaly_4 = df[df['anomaly_K_Means'] == 1]







    for name in selected_columns_names:
        _ = plt.figure(figsize=(18, 3), facecolor='grey')
        #_ = plt.figure(figsize=(18, 3))
        _ = plt.plot(df[name], color='blue', label=name)
        _ = plt.plot(anomaly_4[name], linestyle='none', marker='X', color='orange', markersize=12, label='K_means Anomaly from pca')
        _ = plt.plot(anomaly_1[name], linestyle='none', marker='X', color='red', markersize=12, label='Interquartile Range Anomaly from pca_two')
        _ = plt.plot(anomaly_3[name], linestyle='none', marker='o', color='green', markersize=14, label='IsoFor Anomaly from pca',alpha=0.4)
        _ = plt.legend(loc='upper left',prop={'size': 14})
        _ = plt.title(name)
        ax = plt.gca()
        ax.set_facecolor('grey')
        #ax.patch.set_alpha(0)
        #plt.axes()

        # Set color

        #ax.set_facecolor(color=None)

        #_ = plt.facecolor(color=None)

        # plt.show()
        st.pyplot(plt)
    # for label, start, end in gen_repeating(df['label']):
    #    if start > 0: # make sure lines connect
    #        start -= 1
    #   idx = df.index[start:end+1]
    #    #df2.loc[idx, 'px_last'].plot(ax=ax, color=color, label='')
    #
    #   #ax.axvspan(start, end+1, color=sns.xkcd_rgb[color], alpha=0.8,ymin=0.5,ymax=1)
    #   _ = plt.axvline(x=start,color=sns.xkcd_rgb['black'])





def run_kmeans(df,pca_result, n_clusters=2):
    # tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=200)
    # tsne_pca_results = tsne.fit_transform(pca_result)

    #principalComponents = pca.fit_transform(x)
    # principalDf = pd.DataFrame(data = principalComponents)
    # pca_result = principalComponents[:, 0:no_of_significant_pca_components]
    #pca = PCA(n_components=2)
    #pca_result = pca.fit_transform(new_df_pca.values)
    df['pca_one'] = pca_result['pc1']
    df['pca_two'] = pca_result['pc2']


    # kmeans = KMeans(n_clusters, random_state=0).fit(df[["pca_one","pca_two"]])

    kmeans = KMeans(n_clusters, random_state=0).fit(df[["pca_one", "pca_two"]])

    fig, ax = plt.subplots(figsize=(16, 9))

    ax.grid(False)
    ax.set_facecolor("#FFF")
    ax.spines[["left", "bottom"]].set_visible(True)
    ax.spines[["left", "bottom"]].set_color("#4a4a4a")
    ax.tick_params(labelcolor="#4a4a4a")
    ax.yaxis.label.set(color="#4a4a4a", fontsize=20)
    ax.xaxis.label.set(color="#4a4a4a", fontsize=20)
    # --------------------------------------------------

    # Create scatterplot
    ax = sns.scatterplot(
        ax=ax,
        x=df.pca_one,
        y=df.pca_two,
        hue=kmeans.labels_,
        palette=sns.color_palette("colorblind", n_colors=n_clusters),
        legend=None,
    )

    # Annotate cluster centroids
    for ix, [pca_one, pca_two] in enumerate(kmeans.cluster_centers_):
        ax.scatter(pca_one, pca_two, s=200, c="#a8323e")
        ax.annotate(
            f"Cluster #{ix + 1}",
            (pca_one, pca_two),
            fontsize=25,
            color="#a8323e",
            xytext=(pca_one + 0.5, pca_two + 0.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#a8323e", lw=2),
            ha="center",
            va="center",
        )

    #st.pyplot(fig)
    df['label'] = kmeans.labels_
    return fig


def run_elbow(pca_result):
    sse = []
    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []
    fig, ax = plt.subplots(2, figsize=(8, 4))

    ax[0].set_title('Elbow test', y=1.0, pad=-14)
    ax[0].grid(False)
    ax[0].set_facecolor("#FFF")
    ax[0].spines[["left", "bottom"]].set_visible(True)
    ax[0].spines[["left", "bottom"]].set_color("#4a4a4a")
    ax[0].tick_params(labelcolor="#4a4a4a")
    ax[0].yaxis.label.set(color="#4a4a4a", fontsize=15)
    ax[0].xaxis.label.set(color="#4a4a4a", fontsize=15)

    #ax[1].set_title('Silhouette coefficient for different number of clusters', y=0.5, pad=-14)
    ax[1].grid(False)
    ax[1].set_facecolor("#FFF")
    ax[1].spines[["left", "bottom"]].set_visible(True)
    ax[1].spines[["left", "bottom"]].set_color("#4a4a4a")
    ax[1].tick_params(labelcolor="#4a4a4a")
    ax[1].yaxis.label.set(color="#4a4a4a", fontsize=15)
    ax[1].xaxis.label.set(color="#4a4a4a", fontsize=15)

    clustering_confidence = []

    #pca_result["clusters"] = 0
    pca_result_temptemp=pca_result.copy()
    for k in range(2, 11):
        # for clustering confidence
        pca_temp=pca_result_temptemp.copy()
        ###
        kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=42).fit(pca_result)

        kmeans_conf = KMeans(n_clusters=k, max_iter=1000, random_state=42).fit(pca_result_temptemp)


        col_names=[]

        for i in range(k):
            pca_temp['p' + str(i)] = 0
            col_names.append('p' + str(i))
        pca_temp[col_names] = soft_clustering_weights(pca_result_temptemp, kmeans_conf.cluster_centers_)

        pca_temp['confidence'] =np.max(pca_temp[col_names].values, axis = 1)

        clustering_confidence.append(pca_temp['confidence'].mean())

        print(k)
        print(col_names)
        print(pca_result_temptemp.shape)
        ###

        pca_result["clusters"] = kmeans.labels_
        # print(data["clusters"])
        # compute inertia
        sse.append(kmeans.inertia_)  # Inertia: Sum of distances of samples to their closest cluster center
        # compute silhouette
        score = silhouette_score(pca_result, kmeans.labels_)
        silhouette_coefficients.append(score)



    ax[0].plot(range(2, 11), sse,label='Elbow test results')
    # ax[1].xticks(range(2, 11))
    ax[0].set_xlabel("Number of Clusters")
    ax[0].set_ylabel("SSE")
    #_ = plt.legend(loc='best',prop={'size': 10})

    # Plot silhouette
    # ax[1].style.use("fivethirtyeight")
    ax[1].plot(range(2, 11), silhouette_coefficients,label='Silhouette score')
    # ax[1].xticks(range(2, 11))
    ax[1].set_xlabel("Number of Clusters")
    ax[1].set_ylabel("Silhouette Coefficient")

    # Plot clustering confidence
    ax[1].plot(range(2, 11), clustering_confidence,color='red',  label='Clustering confidence')

    for i, v in enumerate(clustering_confidence):
        ax[1].text(i+2, v, "%d%%" %(v*100), ha="center")



    _ = plt.legend(loc='best',prop={'size': 10})

    # ax.show()
    return fig


# -----------------------------------------------------------

def run_pca(new_df_pca):
    fig, ax = plt.subplots(2, figsize=(8, 6))

    x = new_df_pca
    scaler = StandardScaler()
    pca = PCA()
    pipeline = make_pipeline(scaler, pca)
    pipeline.fit(x)
    # Plot the principal components against their inertia
    features = range(pca.n_components_)

    ax[0].bar(features, pca.explained_variance_)
    ax[0].set_xlabel('PCA feature')
    ax[0].set_ylabel('Variance')
    ax[0].set_xticks(features)
    ax[0].set_title("Importance of the Principal Components based on inertia", y=1.0, x=0.5, pad=-14)

    y = np.cumsum(pca.explained_variance_ratio_)
    xi = np.arange(1, len(y) + 1, step=1)
    ax[1].set_ylim(0.0, 1.1)
    ax[1].plot(xi, y, marker='o', linestyle='--', color='b')

    ax[1].set_xlabel('Number of Components')
    ax[1].set_xticks(np.arange(0, 11, step=1))  # change from 0-based array index to 1-based human-readable label
    ax[1].set_ylabel('Cumulative variance (%)')
    ax[1].set_title('The number of components needed to explain variance', y=0.5, pad=-14)

    ### How much pca components we need to explian 95% of variance
    no_of_significant_pca_components = list(y < 0.95).count(True)

    ax[1].axhline(y=0.95, color='r', linestyle='-')
    ax[1].text(0.2, 0.1,
               str(no_of_significant_pca_components) + ' PCA components are sufficient to explain \n 95% of the cumulative variance',
               color='red', fontsize=16)
    ax[1].grid(axis='x')

    # pca = PCA(n_components=no_of_significant_pca_components)


    #pca = PCA().fit(scaled_data)
    #pca_result = pd.DataFrame(data = pca, columns = ['pc1', 'pc2'])

    pca = PCA(n_components=2)
    scaler=MinMaxScaler()
    pipeline = make_pipeline(scaler, pca)
    pipeline.fit(x)
    principalComponents = pca.fit_transform(x)
    pca_result = pd.DataFrame(data = principalComponents,columns = ['pc1', 'pc2'])

    #principalComponents = pca.fit_transform(x)
    #principalDf = pd.DataFrame(data = principalComponents)

    # pca_result = principalComponents[:, 0:no_of_significant_pca_components]

    #if no_of_significant_pca_components < 2:
    #pca_result = pd.DataFrame(data=principalComponents)
    #else:
    #    pca_result = pd.DataFrame(data=principalComponents[:, 0:no_of_significant_pca_components])
    #
    # fig.show()
    st.pyplot(fig)
    return pca_result


# SIDEBAR
# -----------------------------------------------------------
sidebar = st.sidebar
df_display = sidebar.checkbox("Display Raw Data", value=True)

n_clusters = sidebar.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=10,
)

sidebar.write(
    """
# Finding the number of clusters
### Elbow test
First method is the elbow method. It is based on the {squared error(SSE){ for some value of K. 
SSE is defined as the sum of the squared distance between centroid and each member of the cluster. 
In the plot of the Number of Clusters against SSE graph we observe that as K increases SSE decreases as disortation will be small. 

The idea of this algorithm 
is to choose the value of K at which the graph decrease abruptly, observed as 
an ???elbow effect??? in the graph. 

### Silhouette coefficient
The silhouette coefficient quantifies how well a data point fits into
its assigned cluster based on two factors:

1. How close the data point is to other points in the cluster
2. How far away the data point is from points in other clusters

Silhouette coefficient values range between -1 and 1. Larger numbers indicate that samples 
closer to their clusters than they are to other clusters.

    """
)
# -----------------------------------------------------------

### Add excluding low confident samples based on clustering significnace
### keep or reject samples, maybe use them to reject outliers

# Main
# -----------------------------------------------------------
# Create a title for your app
st.title("Data exploration, clustering and anomaly detection on production line monitoring datasets")

# Select Columns
if st.checkbox("Select Columns To Show"):
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select", all_columns)
    new_df = df[selected_columns]
    st.dataframe(new_df)

if df_display:
    #st.subheader('Sensor data from Customer B')
    st.info("Sensor data from {}".format(filename))
    #st.write(df)



# if st.checkbox("Select Columns for PCA (hint - select all data columns)"):
#    all_columns_names_pca = df.columns.tolist()
#    selected_columns_pca = st.multiselect("Select data columns for PCA",all_columns_names_pca)
#    new_df_pca = df[selected_columns_pca]

#    pca_result=run_pca(new_df_pca)

if st.checkbox("Run PCA?"):
    "Select Columns for PCA (hint - select all data columns, and exclude time/categorical ones)"
    container = st.container()
    all = st.checkbox("Select all")
    all_columns_names_pca = df.columns.tolist()

    if all:
        selected_columns_pca = container.multiselect("Select one or more options:",
                                                     all_columns_names_pca, all_columns_names_pca)


    else:
        selected_columns_pca = container.multiselect("Select one or more options:",
                                                     all_columns_names_pca)



    new_df_pca = df[selected_columns_pca]
    pca_result = run_pca(new_df_pca)

    "### Measuring optimal number of clusters"
    """
    There???s a point where the SSE curve starts to bend known as the elbow point. 
    Find the x-value of this point to determine the reasonable trade-off between 
    (small-enough) error and (small-enough) number of clusters. 
    """
    # Show silhouette analysis
    st.write(run_elbow(pca_result))

    """
    If the Silhouette score is closer to 1, the cluster is dense and 
    well-separated than other clusters. A value near 0 represents overlapping clusters with samples very 
    close to the decision boundary of the neighboring clusters. A negative score [-1, 0] indicates that the 
    samples might have got assigned to the wrong clusters. 
    
    Suggestion: choose a number of cluster such that Silhouette coefficient is high and corresponds to
    the number of clusters indicated in the Elbow test above. 
    """
    # Show cluster scatter plot
    st.write(run_kmeans(df,pca_result, n_clusters=n_clusters))

    # Customizable Plot

    st.subheader("Plotting raw data and anomaly detection")
    all_columns_names = df.columns.tolist()
    # type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
    #selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

if st.checkbox("Generate Plot"):

    all_2 = st.checkbox("Select all sensors")
    container_2 = st.container()
    all_columns_names_2 = df.columns.tolist()
    if all:
        selected_columns_names_plot = container_2.multiselect("Select one or more sensors:",
                                                        all_columns_names_2, all_columns_names_2)


    else:
        selected_columns_names_plot = container_2.multiselect("Select one or more sensors:",
                                                        all_columns_names_2)
    plot_data(df, pca_result,selected_columns_names_plot,n_clusters)
            # st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

        # Plot By Streamlit
        # if type_of_plot == 'area':
        #    cust_data = df[selected_columns_names]
        #   st.area_chart(cust_data)

        # elif type_of_plot == 'bar':
        #    cust_data = df[selected_columns_names]
        #    st.bar_chart(cust_data)

        # elif type_of_plot == 'line':
        #    cust_data = df[selected_columns_names]
        #    st.line_chart(cust_data)

        # Custom Plot
        # elif type_of_plot:
        #    cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
        #    st.write(cust_plot)
        #   st.pyplot()

# -----------------------------------------------------------
