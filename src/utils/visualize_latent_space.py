import os

import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import cv2

from src.utils.utils import batch2img_list
from src.utils.visualization import image_scatter_plot, plot_image_pairs, plot_images, plot_image_grid


def lle(latent_features):
    lle = LocallyLinearEmbedding(n_components=2)
    reduced_data = lle.fit_transform(latent_features)
    # Apply K-Means clustering on the high-dimensional data
    num_clusters = 30  # You can choose the number of clusters based on your data
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(latent_features)

    # Plot the reduced data with colors representing clusters
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', marker='o')
    plt.title('K-Means Clustering of High-Dimensional Data')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Show the cluster centers if needed
    cluster_labels_ = kmeans.predict(kmeans.cluster_centers_)
    cluster_centers = lle.transform(kmeans.cluster_centers_)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=cluster_labels_, marker='x', s=100, label='Cluster Centers')
    plt.legend()
    plt.show()

def tSNE(latent_features, dataset, directory=None):
    """
    Use sklearn to perform tSNE on the latent features
    """
    #  use PCA to reduce to 50 dimensions
    if latent_features.shape[1] > 1000:
        pca = PCA(n_components=1000)
        pca.fit(latent_features)
        latent_features = pca.transform(latent_features)

    for perpexity in [5,20,30,50,100,200,500]:
        # apply tnse from sklearn
        tsne = TSNE(n_components=2, verbose=1, perplexity=perpexity, n_iter=300)
        tsne_results = tsne.fit_transform(latent_features)

        # scatter plot
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], marker='o', alpha=0.5)
        if directory:
            plt.savefig(os.path.join(directory, f"tsne_{perpexity}.png"), dpi=600, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        image_scatter(tsne_results, dataset, np.arange(min(1000, len(dataset))), zoom=0.25, filename=os.path.join(directory, f"tsne_{perpexity}.png"))



def tSNE_tiles(latent_features, dataset, filename):
    """"""
    img_size = dataset[0][0].shape[1]
    #  use PCA to reduce to 50 dimensions
    if latent_features.shape[1] > 50:
        pca = PCA(n_components=50)
        pca.fit(latent_features)
        latent_features = pca.transform(latent_features)

    # apply tnse from sklearn
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latent_features)

    # nearest neigbor
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(tsne_results)
    # create grid
    grid_size = 100
    x = np.linspace(tsne_results[:, 0].min(), tsne_results[:, 0].max(), grid_size)
    y = np.linspace(tsne_results[:, 1].min(), tsne_results[:, 1].max(), grid_size)
    xx, yy = np.meshgrid(x, y)
    # find nearest neighbor
    _, indices = nn.kneighbors(np.stack([xx.ravel(), yy.ravel()], axis=1))
    indices = indices.reshape(xx.shape)
    # create image grid
    grid = np.zeros((grid_size*img_size, grid_size*img_size, 3))
    for i in range(0, grid_size, img_size):
        for j in range(0, grid_size, img_size):
            grid[i:i+img_size, j:j+img_size] = dataset[indices[i, j]][0].numpy().transpose(1,2,0)
    cv2.imwrite(filename, filename)




def pca(latent_features, filename=None):
    # Apply K-Means clustering on the high-dimensional data
    num_clusters = 30  # You can choose the number of clusters based on your data
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(latent_features)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(latent_features)

    # Generate cluster labels (for example, from K-Means clustering)

    # Create a colormap for clusters
    cmap = plt.get_cmap('viridis', len(np.unique(cluster_labels)))

    # Plot the two principal components with colors representing clusters
    plt.scatter(
        principal_components[:, 0],
        principal_components[:, 1],
        c=cluster_labels,
        cmap=cmap,
        marker='o',
        alpha=0.5
    )
    plt.title('PCA: Two Principal Components with Cluster Colors')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Create a colorbar legend
    cbar = plt.colorbar()
    cbar.set_label('Cluster Labels')

    if filename:
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def kmeans_cluster(latent_features, num_clusters=30):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(latent_features)

    # Plot the two principal components with colors representing clusters
    plt.scatter(
        latent_features[:, 0],
        latent_features[:, 1],
        c=cluster_labels,
        marker='o',
        alpha=0.5
    )
    plt.title('K-Means Clustering of Latent Features')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Create a colorbar legend
    cbar = plt.colorbar()
    cbar.set_label('Cluster Labels')

    plt.show()

def single_cluster(latent_features, dataset, num_clusters=5, n_neighbors=100, figure_dir=None):
    if figure_dir:
        os.makedirs(figure_dir, exist_ok=True)
    # Apply K-Means clustering on the high-dimensional data
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(latent_features)
    # Find cluster centers
    cluster_centers = kmeans.cluster_centers_
    # Find cluster label for each sample
    cluster_labels = kmeans.predict(latent_features)


    for i, center in enumerate(cluster_centers):
        _, indices = np.where(cluster_labels == i)
        # shuffle indices
        indices = indices.ravel()[torch.randperm(len(indices))]
        pca = PCA(n_components=2)
        pca.fit(latent_features[indices.ravel()])
        latent = pca.transform(latent_features)

        # save figure or show
        filename = None
        if figure_dir:
            filename = os.path.join(figure_dir, f"single_cluster{i}.png")

        image_scatter(latent, dataset, indices.ravel(), filename=filename)

        # create image grid
        grid_size = 8
        ll = [None] * grid_size
        for j in range(grid_size):
            ll[j] = batch2img_list(torch.stack([dataset[ind][0] for ind in indices[0, j*grid_size:(j+1)*grid_size]]), grid_size)

        if figure_dir:
            filename = os.path.join(figure_dir, f"single_cluster_grid{i}.png")
        plot_image_grid(ll, filename=filename, adaptive=True)



def knn(latent_features, dataset, num_clusters=30, n_neighbors=7):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(latent_features)

    # Apply K-Means clustering on the high-dimensional data
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(latent_features)

    # Find cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Create a NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(latent_features)

    # Find the indices of the nearest neighbors for each cluster center
    nearest_neighbor_indices = []
    for center in cluster_centers:
        _, indices = nn.kneighbors([center])
        nearest_neighbor_indices.append(indices)

    # Convert the indices to a more usable format
    nearest_neighbor_indices = np.array(nearest_neighbor_indices).squeeze()

    # Print the indices of the 5 nearest neighbors for each cluster center
    for i, indices in enumerate(nearest_neighbor_indices):

        batch = torch.stack([dataset[ind] for ind in indices])
        plot_images(batch2img_list(batch, n_neighbors))
        plt.show()
        print(f"Cluster {i} Center Nearest Neighbors Indices: {indices}")


def latent_interpolation(encode, decode, dataset, n=9, i=0, j=10, filename=None):
    image1, _ = dataset[i]
    image2, _ = dataset[j]
    # encode them to get their latent representations
    feat1 = encode(image1.unsqueeze(0))
    feat2 = encode(image2.unsqueeze(0))
    # interpolate between the two images in latent space
    t = torch.linspace(0, 1, n)
    interps = torch.stack([feat1*(1-i) + feat2*i for i in t])
    # decode the interpolations
    decoded = decode(interps)
    pixel_interp = batch2img_list(torch.stack([image1*(1-i) + image2*i for i in t]), n)
    latent_interp = batch2img_list(decoded, n)
    # plot the interpolations
    plot_image_pairs(pixel_interp, latent_interp)
    if filename:
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def mean_shift(latent_features, dataset, figure_dir=None):
    """Unsupervised clustering using mean shift"""
    #  use PCA to reduce to 32 dimensions
    if latent_features.shape[1] > 50:
        pca = PCA(n_components=50)
        pca.fit(latent_features)
        latent_features = pca.transform(latent_features)

    if figure_dir:
        os.makedirs(figure_dir, exist_ok=True)
    # Apply mean shift clustering on the high-dimensional latent features
    bandwidth = estimate_bandwidth(latent_features, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(latent_features)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    for i, cluster_center in enumerate(tqdm(cluster_centers)):
        # get indices from labels
        indices = np.where(labels == i)[0]
        pca = PCA(n_components=2)
        pca.fit(latent_features[indices])
        latent = pca.transform(latent_features)

        # save figure or show
        filename = None
        if figure_dir:
            filename = os.path.join(figure_dir, f"single_cluster{i}.png")

        indices = indices.ravel()[torch.randperm(len(indices))[:400]]
        if len(indices) > latent_features.shape[1]:
            image_scatter(latent, dataset, indices, filename=filename)


        # create image grid
        if figure_dir:
            filename = os.path.join(figure_dir, f"single_cluster_grid{i}.png")
        grid_size = 8
        rows = [None] * min(grid_size, len(indices) // grid_size)
        if len(rows) <= 1:
            plot_images(batch2img_list(torch.stack([dataset[ind][0] for ind in indices]), grid_size))
            plt.savefig(filename, dpi=600, bbox_inches="tight")
            continue
        for j in range(len(rows)):
            rows[j] = batch2img_list(torch.stack([dataset[ind][0] for ind in indices[j*grid_size:(j+1)*grid_size]]), grid_size)

        plot_image_grid(rows, filename=filename, adaptive=True)


def image_scatter(latent_features, dataset, indices, filename=None, zoom=1):
    """Plot the latent features on a 2D plane with images as the points"""
    latent_features = latent_features[indices]
    batch = torch.stack([dataset[ind][0] for ind in indices])

    img_list = batch2img_list(batch, len(indices))
    image_scatter_plot(img_list, latent_features[:, 0], latent_features[:, 1], filename=filename, zoom=zoom)
