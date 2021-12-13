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

sns.set_theme()
# -----------------------------------------------------------

# Helper functions
# -----------------------------------------------------------
# Load data from external source

def file_selector(folder_path="./datasets"):
    filenames = os.listdir(folder_path)
    csv_files = list(filter(lambda f: f.endswith('.csv'), filenames))
    selected_filename = st.selectbox("Select a file with data from customer",csv_files)
    return os.path.join(folder_path,selected_filename)

filename = file_selector()
st.info("You Selected {}".format(filename))


@st.cache(allow_output_mutation=True)
def load_data(file):
    #Read sensor data from testing on 01_02_2021
    df = pd.read_csv(file)
    # Renaming columns
    #df.columns=['time','Test_id','BL_x','BL_y','BL_z','BP_x','BP_y','BP_z','AP_x','AP_y','AP_z','AL_x','AL_y','AL_z']

    return df

# Read Data
df = load_data(filename)


def run_kmeans(df, n_clusters=2):
    kmeans = KMeans(n_clusters, random_state=0).fit(df[["BL_x", "BL_y"]])

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
        x=df.BL_x,
        y=df.BL_y,
        hue=kmeans.labels_,
        palette=sns.color_palette("colorblind", n_colors=n_clusters),
        legend=None,
    )

    # Annotate cluster centroids
    for ix, [BL_x, BL_y] in enumerate(kmeans.cluster_centers_):
        ax.scatter(BL_x, BL_y, s=200, c="#a8323e")
        ax.annotate(
            f"Cluster #{ix+1}",
            (BL_x, BL_y),
            fontsize=25,
            color="#a8323e",
            xytext=(BL_x+0.5, BL_y+0.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#a8323e", lw=2),
            ha="center",
            va="center",
        )

    return fig

def run_elbow(df):
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

    ax[1].set_title('Silhouette coefficient for different number of clusters', y=1.0, pad=-14)
    ax[1].grid(False)
    ax[1].set_facecolor("#FFF")
    ax[1].spines[["left", "bottom"]].set_visible(True)
    ax[1].spines[["left", "bottom"]].set_color("#4a4a4a")
    ax[1].tick_params(labelcolor="#4a4a4a")
    ax[1].yaxis.label.set(color="#4a4a4a", fontsize=15)
    ax[1].xaxis.label.set(color="#4a4a4a", fontsize=15)

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df[["BL_x", "BL_y"]])
        df["clusters"] = kmeans.labels_
        #print(data["clusters"])
        #compute inertia
        sse.append(kmeans.inertia_) # Inertia: Sum of distances of samples to their closest cluster center
        #compute silhouette
        score = silhouette_score(df[["BL_x", "BL_y"]], kmeans.labels_)
        silhouette_coefficients.append(score)

    ax[0].plot(range(2, 11), sse)
    #ax[1].xticks(range(2, 11))
    ax[0].set_xlabel("Number of Clusters")
    ax[0].set_ylabel("SSE")

    #ax[1].style.use("fivethirtyeight")
    ax[1].plot(range(2, 11), silhouette_coefficients)
    #ax[1].xticks(range(2, 11))
    ax[1].set_xlabel("Number of Clusters")
    ax[1].set_ylabel("Silhouette Coefficient")

        #ax.show()
    return fig
# -----------------------------------------------------------

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
First method is the elbow method. It is based on the squared error (SSE) for some value of K. 
SSE is defined as the sum of the squared distance between centroid and each member of the cluster. 
In the plot of the Number of Clusters against SSE graph we observe that as K increases SSE decreases as disortation will be small. 

The idea of this algorithm 
is to choose the value of K at which the graph decrease abruptly, observed as 
an “elbow effect” in the graph. 

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


# Main
# -----------------------------------------------------------
# Create a title for your app
st.title("Interactive K-Means Clustering")




# Show cluster scatter plot
st.write(run_kmeans(df, n_clusters=n_clusters))

"### Measuring optimal number of clusters"
"""
There’s a point where the SSE curve starts to bend known as the elbow point. 
Find the x-value of this point to determine the reasonable trade-off between 
(small-enough) error and (small-enough) number of clusters. 
"""
# Show silhouette analysis
st.write(run_elbow(df))

"""
If the Silhouette score is closer to 1, the cluster is dense and 
well-separated than other clusters. A value near 0 represents overlapping clusters with samples very 
close to the decision boundary of the neighboring clusters. A negative score [-1, 0] indicates that the 
samples might have got assigned to the wrong clusters. 

Suggestion: choose a number of cluster such that Silhouette coefficient is high and corresponds to
the number of clusters indicated in the Elbow test above. 
"""

if df_display:
    st.subheader('Sensor data from Customer B')
    st.write(df)

# Select Columns
if st.checkbox("Select Columns To Show"):
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select",all_columns)
    new_df = df[selected_columns]
    st.dataframe(new_df)



    # Customizable Plot

    st.subheader("Customizable Plot")
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
    selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

    if st.button("Generate Plot"):
        st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

        # Plot By Streamlit
        if type_of_plot == 'area':
            cust_data = df[selected_columns_names]
            st.area_chart(cust_data)

        elif type_of_plot == 'bar':
            cust_data = df[selected_columns_names]
            st.bar_chart(cust_data)

        elif type_of_plot == 'line':
            cust_data = df[selected_columns_names]
            st.line_chart(cust_data)

        # Custom Plot
        elif type_of_plot:
            cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()


# -----------------------------------------------------------
