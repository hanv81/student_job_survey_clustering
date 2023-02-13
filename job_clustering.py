from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import umap
from hdbscan import HDBSCAN

@st.cache_data
def load_model():
    print('load model')
    # return SentenceTransformer('distiluse-base-multilingual-cased-v2')
    return SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')

@st.cache_data
def load_data():
    print('load_data')
    return pd.read_csv('AI4ALL_Survey_clean.csv')

@st.cache_resource
def encode(_model, df):
    print('encode')
    jobs = [tokenize(job) for job in df['Job']]
    return _model.encode(jobs)

@st.cache_data
def kmeans_clustering(X: np.ndarray, n_cluster: int) -> KMeans:
	print("kmeans_clustering")
	kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(X)
	return kmeans

@st.cache_data
def hdbscan_clustering(X: np.ndarray, min_cluster_size: int) -> HDBSCAN:
    print('hdbscan_cluster')
    return HDBSCAN(min_cluster_size=min_cluster_size,
                           gen_min_span_tree=True).fit(X)

@st.cache_data
def create_knn(n_neighbors):
    print("create_knn")
    return KNeighborsClassifier(n_neighbors=n_neighbors)

@st.cache_data
def fit_pca(X, n_components):
    print("fit_pca", n_components)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

@st.cache_data
def fit_umap(X, n_components):
    print("fit_umap", n_components)
    reducer = umap.UMAP(n_components=n_components)
    return reducer.fit_transform(X)

@st.cache_data
def fit_tsne(X, n_components):
    print("fit_tsne", n_components)
    tsne = TSNE(n_components=n_components, learning_rate='auto', init='random')
    return tsne.fit_transform(X)

@st.cache_data
def draw_3d(X, y):
    print("draw_3d")
    return px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y)

@st.cache_data
def draw_2d(X, y):
    print('draw_2d')
    fig, ax = plt.subplots()
    for i in range(y.max()):
        ax.scatter(X[y==i, 0], X[y==i, 1])
    return fig

def main():
    model = load_model()
    df = load_data()
    X = encode(model, df)

    col1, col2 = st.columns(2)
    with col1:
        reduction = st.radio('Dimension Reduction', ('PCA', 'TSNE', 'UMAP'), horizontal=True)
        n_components = st.slider('Number of components for clustering', 3, min(X.shape[0], X.shape[1]), 100)
        if reduction == 'PCA':
            X_red = fit_pca(X, n_components)
        elif reduction == 'TSNE':
            X_red = fit_tsne(X, n_components)
        else:
            X_red = fit_umap(X, n_components)
    with col2:
        clustering_alg = st.radio('Clustering Algorithm', ('Kmeans', 'HDBSCAN'), horizontal=True)
        if clustering_alg == 'Kmeans':
            n_cluster = st.slider('Number of clusters', 3, 20, 10)
            cluster = kmeans_clustering(X_red, n_cluster)
        else:
            min_cluster_size = st.slider('Minimum cluster size', 3, 20, 8)
            cluster = hdbscan_clustering(X_red, min_cluster_size)
    print('max cluster:', max(cluster.labels_))

    dimension_plot = st.radio('Plotting Dimension', ('2D', '3D'), horizontal=True)
    n_dimension_draw = 2 if dimension_plot == '2D' else 3
    if reduction == 'PCA':
        X_draw = fit_pca(X, n_dimension_draw)
    elif reduction == 'TSNE':
        X_draw = fit_tsne(X, n_dimension_draw)
    else:
        X_draw = fit_umap(X, n_dimension_draw)

    if dimension_plot == '2D':
        fig = draw_2d(X_draw, cluster.labels_)
        st.pyplot(fig)
    else:
        fig = draw_3d(X_draw, cluster.labels_)
        st.plotly_chart(fig, use_container_width=True)

    cluster_id = st.selectbox('Cluster ID', tuple(range(max(cluster.labels_)+1)))
    
    st.dataframe(df[cluster.labels_ == cluster_id])
    with st.sidebar:
        job = st.text_input('Your dream job')
        if st.button('Search') or job:
            print('search')
            knn = create_knn(5)
            knn.fit(X, cluster.labels_)
            emb = model.encode(job).reshape(1,-1)
            id = knn.predict(emb)
            st.write('Your job cluster: ', id[0])
            r = knn.kneighbors(emb, n_neighbors=5)
            r = df.loc[r[1][0]]
            for i in r.index:
                st.write(r.loc[i, 'Job'])

main()