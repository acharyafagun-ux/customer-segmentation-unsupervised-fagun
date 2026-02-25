from sklearn.metrics import silhouette_score, davies_bouldin_score


def evaluate_clustering(data, labels):
    # If DBSCAN produces noise (-1), ignore it for evaluation
    if len(set(labels)) <= 1:
        return None, None

    silhouette = silhouette_score(data, labels)
    db_index = davies_bouldin_score(data, labels)

    return silhouette, db_index