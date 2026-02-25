import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import load_data, clean_data, create_features, transform_and_scale
from clustering import run_kmeans, run_hierarchical, run_dbscan, run_gmm
from evaluation import evaluate_clustering
from visualization import plot_elbow, plot_correlation_heatmap, plot_pca
from sklearn.cluster import KMeans
import pandas as pd


def main():

    raw_data_path = "data/raw/online_retail.csv"

    processed_data_path = "data/processed"
    results_metrics_path = "results/metrics"
    cluster_plots_path = "results/cluster_plots"
    pca_outputs_path = "results/pca_outputs"

    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(results_metrics_path, exist_ok=True)
    os.makedirs(cluster_plots_path, exist_ok=True)
    os.makedirs(pca_outputs_path, exist_ok=True)

    df = load_data(raw_data_path)
    df = clean_data(df)

    df.to_csv(os.path.join(processed_data_path, "cleaned_data.csv"), index=False)

    # Feature Engineering 
    rfm = create_features(df)

    rfm.to_csv(os.path.join(processed_data_path, "rfm_features.csv"))

    plot_correlation_heatmap(
        rfm,
        save_path=os.path.join(cluster_plots_path, "correlation_heatmap.png")
    )

    # Transform & Scale 
    rfm_scaled, rfm_log = transform_and_scale(rfm)

    # Elbow Method 
    inertia = []
    K = range(1, 11)

    for k in K:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(rfm_scaled)
        inertia.append(model.inertia_)

    plot_elbow(
        K,
        inertia,
        save_path=os.path.join(cluster_plots_path, "elbow_plot.png")
    )

    # Clustering 
    kmeans_model, kmeans_labels = run_kmeans(rfm_scaled, n_clusters=3)
    hier_model, hier_labels = run_hierarchical(rfm_scaled, n_clusters=3)
    dbscan_model, dbscan_labels = run_dbscan(rfm_scaled)
    gmm_model, gmm_labels = run_gmm(rfm_scaled, n_clusters=3)

    # Evaluation 
    results = []

    for name, labels in [
        ("KMeans", kmeans_labels),
        ("Hierarchical", hier_labels),
        ("DBSCAN", dbscan_labels),
        ("GMM", gmm_labels),
    ]:
        silhouette, db_index = evaluate_clustering(rfm_scaled, labels)
        results.append([name, silhouette, db_index])

    metrics_df = pd.DataFrame(
        results,
        columns=["Algorithm", "Silhouette Score", "Davies-Bouldin Index"]
    )

    metrics_df.to_csv(
        os.path.join(results_metrics_path, "model_comparison_metrics.csv"),
        index=False
    )

    # Final KMeans Output 
    rfm["Cluster"] = kmeans_labels

    # Save clustered customer dataset
    rfm.to_csv(
        os.path.join(results_metrics_path, "final_clustered_customers.csv")
    )

    #  PCA (Final Visualization) 
    variance_ratio = plot_pca(
        rfm_scaled,
        kmeans_labels,
        save_path=os.path.join(pca_outputs_path, "pca_clusters.png")
    )

    print("\nExplained Variance Ratio (PCA):", variance_ratio)
    print("\nModel Comparison:")
    print(metrics_df)


if __name__ == "__main__":
    main()
