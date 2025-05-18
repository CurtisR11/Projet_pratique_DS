import subprocess
import os
import pandas as pd

def run_tp(file_path):
    print(f"\n🚀 Exécution de : {file_path}")
    try:
        result = subprocess.run(["python", file_path], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur dans {file_path} :\n{e.stderr}")

if __name__ == "__main__":
    print(f"\n🚀 Exécution du fichier main")
    user_desktop_main = os.path.join(os.path.expanduser("~"), "Desktop", "Projet_python_main")
    tp_files = [
        "tp1_final.py",
        "tp2_final.py",
        "tp3_final.py",
        "tp4_final.py",
        "tp5_final.py",
        "tp6_final.py",
        # "tp7_final.py",
        "tp8_final.py"
    ]

    for tp in tp_files:
        tp_path = os.path.join(user_desktop_main, tp)
        run_tp(tp_path)


# summary

# Dossiers
base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Projet_python", "output")
tp2_path = os.path.join(base_path, "clustering_tp2")
tp4_path = os.path.join(base_path, "prediction_tp4")

summary_path = os.path.join(base_path, "summary")
os.makedirs(summary_path, exist_ok=True)

# Chargement des fichiers
df_fin = pd.read_csv(os.path.join(tp2_path, "financial_profile_labels.csv"))
df_risk = pd.read_csv(os.path.join(tp2_path, "risk_profile_labels.csv"))
df_corr = pd.read_csv(os.path.join(tp2_path, "daily_returns_correlation_labels.csv"))
df_preds = pd.read_csv(os.path.join(tp4_path, "j+1_predictions_models_tp4.csv"))

# Renommage colonnes
for df in [df_fin, df_risk, df_corr]:
    df.rename(columns={df.columns[0]: "Company"}, inplace=True)

# Liste des compagnies à analyser
companies = [
    "Tesla", "Samsung", "Tencent", "Alibaba", "Sony",
    "Adobe", "Johnson & Johnson", "Pfizer", "Louis Vuitton (LVMH)"
]

# Fonction pour trouver les pairs dans le même cluster
def find_cluster_peers(df, company_name, cluster_column="Hierarchical_Cluster"):
    if company_name not in df["Company"].values:
        return []
    cluster_id = df[df["Company"] == company_name][cluster_column].values[0]
    same_cluster = df[df[cluster_column] == cluster_id]["Company"].tolist()
    if company_name in same_cluster:
        same_cluster.remove(company_name)
    return same_cluster

# Compilation des résultats
results = []

for company in companies:
    peers_financial = find_cluster_peers(df_fin, company, "Hierarchical_Cluster")
    peers_risk = find_cluster_peers(df_risk, company, "Hierarchical_Cluster")
    peers_corr = find_cluster_peers(df_corr, company, "Hierarchical_Cluster")

    if company in df_preds["Company"].values:
        row = df_preds[df_preds["Company"] == company].iloc[0]
        pred_xgb = row["XGBoost"]
        pred_rf = row["RandomForest"]
        pred_knn = row["KNN"]
        pred_lr = row["Linear"]
    else:
        last_price = pred_xgb = pred_rf = pred_knn = pred_lr = None

    results.append({
        "Company": company,
        "Peers_Financial": peers_financial,
        "Peers_Risk": peers_risk,
        "Peers_Correlation": peers_corr,
        "Pred_XGBoost": pred_xgb,
        "Pred_RF": pred_rf,
        "Pred_KNN": pred_knn,
        "Pred_Linear": pred_lr
    })

# Création du DataFrame final
df_final = pd.DataFrame(results)

print(df_final)

file_path = os.path.join(summary_path, "summary.csv")
df_final.to_csv(file_path, index=False)




