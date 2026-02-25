import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_elbow(K, inertia, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertia, marker='o')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")

    if save_path:
        plt.savefig(save_path)

    plt.close()


def plot_correlation_heatmap(df, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")

    if save_path:
        plt.savefig(save_path)

    plt.close()


def plot_pca(data, labels, save_path=None):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', s=10)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Customer Segments (PCA Projection)")
    plt.colorbar(label="Cluster")

    if save_path:
        plt.savefig(save_path)

    plt.close()

    return pca.explained_variance_ratio_