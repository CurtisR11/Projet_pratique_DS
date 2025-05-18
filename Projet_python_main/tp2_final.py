import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

def preprocess_for_financial_clustering(filepath, selected_columns):
    df = pd.read_csv(filepath, index_col=0)
    df_clean = df[selected_columns].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_clean)
    return pd.DataFrame(data_scaled, index=df_clean.index, columns=df_clean.columns), df_clean

def prepare_daily_returns(folder_path):
    all_returns = {}
    for filepath in glob.glob(os.path.join(folder_path, "*.csv")):
        company = os.path.basename(filepath).replace(".csv", "").replace("_", " ")
        df = pd.read_csv(filepath)
        if "Daily_Return" in df.columns:
            all_returns[company] = df["Daily_Return"]
    df_returns = pd.DataFrame(all_returns).fillna(method='ffill').fillna(method='bfill')
    return df_returns.corr()

def do_kmeans(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return labels, silhouette_score(data, labels)

def do_hierarchical(data, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(data)
    return labels, silhouette_score(data, labels)

def do_dbscan(data, eps=1, min_samples=2):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    try:
        score = silhouette_score(data, labels) if len(set(labels)) > 1 else -1
    except:
        score = -1
    return labels, score

def save_labels_with_data(df_data, labels_dict, dataset_name, output_path):
    df_out = df_data.copy()
    for algo, labels in labels_dict.items():
        df_out[algo + "_Cluster"] = labels
    filename = dataset_name.lower().replace(" ", "_") + "_labels.csv"
    full_path = os.path.join(output_path, filename)
    df_out.to_csv(full_path)
    print(f"✅ Données + labels sauvegardés : {full_path}")

def plot_tsne(data_scaled, labels_dict, dataset_name, output_path):
    tsne = TSNE(n_components=2, random_state=42, init="random", perplexity=5)
    tsne_results = tsne.fit_transform(data_scaled)

    for algo, clusters in labels_dict.items():
        df_tsne = pd.DataFrame(tsne_results, columns=["TSNE1", "TSNE2"])
        df_tsne["Cluster"] = clusters

        plt.figure(figsize=(8, 6))
        unique_clusters = np.unique(clusters)
        colors = plt.colormaps.get_cmap("tab10")

        for i, cluster in enumerate(unique_clusters):
            subset = df_tsne[df_tsne["Cluster"] == cluster]
            plt.scatter(subset["TSNE1"], subset["TSNE2"],
                        label=f"Cluster {cluster}",
                        color=colors(i % 10),
                        alpha=0.7)

        plt.xlabel("TSNE 1")
        plt.ylabel("TSNE 2")
        plt.title(f"TSNE - {dataset_name} - {algo}")
        plt.legend()
        filename = f"{dataset_name.lower().replace(' ', '_')}_{algo.lower()}_tsne.png"
        plt.savefig(os.path.join(output_path, filename))
        plt.close()
        print(f"📸 TSNE sauvegardé : {filename}")

def main():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    project_path = os.path.join(desktop, "Projet_python")
    ratios_path = os.path.join(project_path, "data_tp1", "ratios.csv")
    returns_folder = os.path.join(project_path, "data_tp1", "companies")
    output_path = os.path.join(project_path, "output", "clustering_tp2")
    os.makedirs(output_path, exist_ok=True)

    selected_columns_finance = ["forwardPE", "beta", "priceToBook", "returnOnEquity"]
    selected_columns_risk = ["debtToEquity", "currentRatio", "quickRatio"]

    data_finance_scaled, data_finance_raw = preprocess_for_financial_clustering(ratios_path, selected_columns_finance)
    data_risk_scaled, data_risk_raw = preprocess_for_financial_clustering(ratios_path, selected_columns_risk)
    data_returns = prepare_daily_returns(returns_folder)
    data_returns_dist = 1 - data_returns

    datasets = {
        "Financial Profile": (data_finance_scaled, data_finance_raw),
        "Risk Profile": (data_risk_scaled, data_risk_raw),
        "Daily Returns Correlation": (data_returns_dist, data_returns)
    }

    results = []

    for name, (data_scaled, data_original) in datasets.items():
        print(f"\n📊 Résultats pour : {name}")
        try:
            labels_km, score_km = do_kmeans(data_scaled, n_clusters=5)
            labels_hc, score_hc = do_hierarchical(data_scaled, n_clusters=5)
            labels_db, score_db = do_dbscan(data_scaled, eps=1, min_samples=2)

            all_scores = {
                "KMeans": score_km,
                "Hierarchical": score_hc,
                "DBSCAN": score_db
            }
            best_algo = max(all_scores, key=all_scores.get)

            print(f"KMeans Silhouette Score      : {score_km:.3f}")
            print(f"Hierarchical Silhouette Score: {score_hc:.3f}")
            print(f"DBSCAN Silhouette Score      : {score_db:.3f}")
            print(f"✅ Meilleur algo : {best_algo}")

            results.append({
                "Dataset": name,
                "KMeans Silhouette": score_km,
                "Hierarchical Silhouette": score_hc,
                "DBSCAN Silhouette": score_db,
                "Best Algorithm": best_algo
            })

            labels_dict = {
                "KMeans": labels_km,
                "Hierarchical": labels_hc,
                "DBSCAN": labels_db
            }

            save_labels_with_data(data_original, labels_dict, name, output_path)
            plot_tsne(data_scaled, labels_dict, name, output_path)

        except Exception as e:
            print(f"❌ Erreur sur {name} : {e}")
            results.append({
                "Dataset": name,
                "KMeans Silhouette": None,
                "Hierarchical Silhouette": None,
                "DBSCAN Silhouette": None,
                "Best Algorithm": f"Erreur: {e}"
            })

    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_path, "clustering_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"📄 Résumé des scores sauvegardé dans : {results_csv_path}")

if __name__ == "__main__":
    main()
