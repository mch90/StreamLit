# Imports
# -----------------------------------------------------------
from scipy.sparse import data
import os
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
# -----------------------------------------------------------

# Helper functions
# -----------------------------------------------------------
# Load data from external source

def file_selector(folder_path="./datasets"):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select A file",filenames)
    return os.path.join(folder_path,selected_filename)

filename = file_selector()
st.info("You Selected {}".format(filename))


@st.cache
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
        """

-----
"""
"""
)
# -----------------------------------------------------------


# Main
# -----------------------------------------------------------
# Create a title for your app
st.title("Interactive K-Means Clustering for selected customers")
"""
Some text
"""


# Show cluster scatter plot
st.write(run_kmeans(df, n_clusters=n_clusters))

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
