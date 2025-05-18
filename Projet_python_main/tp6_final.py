import requests
import json
from datetime import datetime, timedelta
import os
import pandas as pd

# 📁 Dossier de sortie configuré ici
user_path = os.path.expanduser("~")
OUTPUT_FOLDER_path = os.path.join(user_path, "Desktop", "Projet_python", "output", "news_tp6")
os.makedirs(OUTPUT_FOLDER_path, exist_ok=True)    

def load_existing_news(company_name):
    """Charge les news existantes depuis un fichier JSON"""
    filename = os.path.join(OUTPUT_FOLDER_path, f"{company_name.lower().replace(' ', '_')}_news.json")
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    return {}

def save_news(company_name, news_dict):
    """Sauvegarde les news dans un fichier JSON"""
    filename = os.path.join(OUTPUT_FOLDER_path, f"{company_name.lower().replace(' ', '_')}_news.json")
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(news_dict, file, indent=4, ensure_ascii=False)

def get_news_by_date(company_name, api_key):
    """Scrape les news récentes et sauvegarde les nouvelles"""
    url = 'https://newsapi.org/v2/everything'
    last_day = datetime.today().strftime('%Y-%m-%d')
    first_day = (datetime.today() - timedelta(days=10)).strftime('%Y-%m-%d')

    params = {
        "q": company_name,
        "apiKey": api_key,
        "language": "en",
        "pageSize": 100,
        "from": first_day,
        "to": last_day,
        "sources": 'financial-post,the-wall-street-journal,bloomberg,the-washington-post,australian-financial-review'
    }

    response = requests.get(url, params=params)
    news_dict = load_existing_news(company_name)

    if response.status_code == 200:
        articles = response.json().get('articles', [])
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            published_at = article.get('publishedAt', '')
            source_name = article.get('source', {}).get('name', '')

            if company_name.lower() in title.lower() or company_name.lower() in description.lower():
                date = published_at.split("T")[0]
                article_data = {
                    'title': title,
                    'description': description,
                    'source': source_name,
                    'date': date
                }

                # Vérifie si l'article existe déjà
                if date not in news_dict:
                    news_dict[date] = []

                if not any(existing['title'] == title for existing in news_dict[date]):
                    news_dict[date].append(article_data)

        # Sauvegarde des nouvelles données
        save_news(company_name, news_dict)
        print(f"✔️ Articles mis à jour pour {company_name}")
    else:
        print(f"❌ Erreur pour {company_name} : {response.status_code} - {response.text}")

# 🔐 Clé API NewsAPI
api_key = "631fa67f0ce74c7ebc857bb00a2d1df4"

# 📦 Liste des entreprises
all_companies = [
    "Apple",
    "Microsoft",
    "Amazon",
    "Alphabet",
    "Meta",
    "Tesla",
    "NVIDIA",
    "Samsung",
    "Tencent",
    "Alibaba",
    "IBM",
    "Intel",
    "Oracle",
    "Sony",
    "Adobe",
    "Netflix",
    "AMD",
    "Qualcomm",
    "Cisco",
    "JP Morgan",
    "Goldman Sachs",
    "Visa",
    "Johnson & Johnson",
    "Pfizer",
    "ExxonMobil",
    "ASML",
    "SAP",
    "Siemens",
    "Louis Vuitton (LVMH)",
    "TotalEnergies",
    "Shell",
    "Baidu",
    "JD.com",
    "BYD",
    "ICBC",
    "Toyota",
    "SoftBank",
    "Nintendo",
    "Hyundai",
    "Reliance Industries",
    "Tata Consultancy Services"
]

# ▶️ Lancement de la collecte
for company_name in all_companies:
    get_news_by_date(company_name, api_key)
