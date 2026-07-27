import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("running")
st.title("Neighborhood Business Similarity Explorer")

# load data and clean it once
@st.cache_data
def load_data():
    gdf = gpd.read_file("business-licences.geojson")

    # drop cols without geom
    gdf = gdf.dropna(subset=["geometry"])

    # get latitude and long
    gdf["longitude"] = gdf.geometry.x
    gdf["latitude"] = gdf.geometry.y

    # filter to issued licences
    gdf = gdf[gdf["status"] == "Issued"]

    # consolidate business types
    gdf["businesstype"] = gdf["businesstype"].fillna(gdf["businesssubtype"])
    top_types = gdf["businesstype"].value_counts().head(25).index
    gdf["businesstype"] = gdf["businesstype"].where(gdf["businesstype"].isin(top_types), "Other")

    # keep only necessary cols
    final_df = gdf[["businesstype", "localarea", "latitude", "longitude"]]
    return final_df

# B1, using localarea instead of postal code (as postal code has missing values and formatting issues)
@st.cache_data
def build_feature_vector(df, min_businesses):

    # keep only areas above the min limit
    area_counts = df["localarea"].value_counts()
    valid_areas = area_counts[area_counts >= min_businesses].index
    areas = df[df["localarea"].isin(valid_areas)]

    vector = pd.crosstab(areas["localarea"], areas["businesstype"], normalize="index") * 100

    # get centroids for cluster position
    centroids = areas.groupby("localarea")[["latitude", "longitude"]].mean()

    # get new filtered counts
    counts = areas.groupby("localarea").size().rename("counts")

    return vector, centroids, counts

# running the app -> main code
df = load_data()

with st.expander("Look at Data:"):
    st.dataframe(df.head(20))
    area_counts = df["localarea"].value_counts()
    st.write(f"{len(df)} businesses, {df['localarea'].nunique()} areas.")
    st.write(area_counts)

# B2 - Slider in the side bar
st.sidebar.header("1. Business Count")
# minimum businesses in an area can be 4, as the smallest count in an area in the data is 4
# for the graph, at least 2 areas should exist with a minimum number of businesses, so that number is 6000
min_businesses = st.sidebar.slider("Minimum businesses per area", 4, 6000, 1000)

# get the feature vector
feature_vector, centroids, counts = build_feature_vector(df, min_businesses)

# adding this failsafe just in case
if len(feature_vector) < 2:
    st.warning("Less than 2 areas exist above this number of businesses, lower it to see clustering for more areas.")
    st.stop()

# scaling the features
X_scaled = StandardScaler().fit_transform(feature_vector.to_numpy())

# Part B2, interactive K-means
st.sidebar.header("2. Clustering")
max_k = min(10, len(feature_vector))
k = st.sidebar.slider("Number of Clusters (K)", 2, max_k, min(4, max_k))

model = KMeans(n_clusters=k, n_init=10, random_state=42)
labels = model.fit_predict(X_scaled)

# check label count
if len(labels) == 0:
    st.warning("there are no clustering labels")
    st.stop()

feature_vector["cluster"] = pd.Categorical(labels.astype(str))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Areas", len(feature_vector))
with col2:
    st.metric("Clusters", feature_vector["cluster"].nunique())
with col3:
    st.metric("Businesses (filtered)", int(counts.sum()))

# B2 & B3 - Map for K-Means cluster
st.subheader("MAP")

geo = centroids.join(counts).join(feature_vector["cluster"])
geo["area"] = geo.index

fig = px.scatter_map(
    geo, lat="latitude", lon="longitude", color="cluster", size="counts",
    hover_name="area", zoom=10, height=550, map_style="carto-darkmatter"
)
st.plotly_chart(fig, width="stretch")

# B4 - Cluster membership
st.subheader("Cluster membership")

for c, group in feature_vector.groupby("cluster"):
    st.write(f"**Cluster {c}:** {', '.join(group.index)}")

    top_types = (
        group.select_dtypes(include="number")
             .mean()
             .sort_values(ascending=False)
             .head(5)
    )

    st.write(top_types)
    st.write()

st.write("B4 - Explanation: Given what I know about the areas in vancouver, we can " \
"see that the most dense area containing businesses is Downtown Vancouver with the " \
"biggest cluster, with businesses like legal, health, business and retail." \
"Other clusters like Cluster 0: Fairview, Kerrisdale, Kitsilano, Renfrew-Collingwood, Riley Park, West End" \
"are grouped together based on their business types: health, rental, retail and restaurants." \
"Hence, all clusters have been grouped together here based on the likeliness of their business type," \
"while cluster sizes depends on how dense / the count of the businesses are in those areas grouped by type.")