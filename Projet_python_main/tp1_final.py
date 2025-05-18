from datetime import datetime, timedelta
import os
import time
import yfinance as yf
import pandas as pd

companies = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Alphabet": "GOOGL",
    "Meta": "META",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Samsung": "005930.KS",
    "Tencent": "TCEHY",
    "Alibaba": "BABA",
    "IBM": "IBM",
    "Intel": "INTC",
    "Oracle": "ORCL",
    "Sony": "SONY",
    "Adobe": "ADBE",
    "Netflix": "NFLX",
    "AMD": "AMD",
    "Qualcomm": "QCOM",
    "Cisco": "CSCO",
    "JP Morgan": "JPM",
    "Goldman Sachs": "GS",
    "Visa": "V",
    "Johnson & Johnson": "JNJ",
    "Pfizer": "PFE",
    "ExxonMobil": "XOM",
    "ASML": "ASML.AS",
    "SAP": "SAP.DE",
    "Siemens": "SIE.DE",
    "Louis Vuitton (LVMH)": "MC.PA",
    "TotalEnergies": "TTE.PA",
    "Shell": "SHEL.L",
    "Baidu": "BIDU",
    "JD.com": "JD",
    "BYD": "BYDDY",
    "ICBC": "1398.HK",
    "Toyota": "TM",
    "SoftBank": "9984.T",
    "Nintendo": "NTDOY",
    "Hyundai": "HYMTF",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS"
}

def safe_get_ticker_info(symbol, retries=1, delay=3):
    for attempt in range(retries):
        try:
            return yf.Ticker(symbol).info
        except Exception as e:
            if "Rate limited" in str(e):
                print(f"⏳ Rate limited pour {symbol}. Nouvelle tentative dans {delay} secondes...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e
    raise Exception(f"❌ Échec après {retries} tentatives pour {symbol}")

def scrape_financial_ratios(output_path):
    print("▶ Scraping des ratios financiers...")
    ratios = {
        "forwardPE": [], "beta": [], "priceToBook": [], "priceToSales": [],
        "dividendYield": [], "trailingEps": [], "debtToEquity": [],
        "currentRatio": [], "quickRatio": [], "returnOnEquity": [],
        "returnOnAssets": [], "operatingMargins": [], "profitMargins": []
    }

    company_names = []

    for company_name, symbol in companies.items():
        try:
            info = safe_get_ticker_info(symbol)
            company_names.append(company_name)
            for ratio_key in ratios.keys():
                value = info.get(ratio_key, None)
                ratios[ratio_key].append(value)
            time.sleep(1)
        except Exception as e:
            print(f"❌ Erreur pour {company_name}: {e}")
            for ratio_key in ratios:
                ratios[ratio_key].append(None)
            company_names.append(company_name)

    df_ratios = pd.DataFrame(ratios, index=company_names)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_ratios.to_csv(output_path, encoding='utf-8')
    print(f"✅ Ratios financiers sauvegardés dans {output_path}")

def fetch_stock_variations(company_name, symbol, start_date, end_date, output_folder):
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        df.columns = df.columns.droplevel(1)
        df.columns.name = None
        df.index.name = None
        df = df[["Close"]]
        df = df.reset_index()
        df.rename(columns={"index": "Date"}, inplace=True)

        if df.empty:
            raise ValueError("DataFrame vide – possible rate limit ou symbole invalide.")

        # Conversion sécurisée de 'Close' en float
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)

        df['Next_Day_Close'] = df['Close'].shift(-1)
        df['Daily_Return'] = df['Close'].pct_change()
        df.dropna(inplace=True)

        os.makedirs(output_folder, exist_ok=True)
        file_path = os.path.join(output_folder, f"{company_name.replace(' ', '_')}.csv")
        df.to_csv(file_path, index=False)
        time.sleep(1)
        return True, company_name, file_path
    except Exception as e:
        return False, company_name, str(e)

def main():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    project_path = os.path.join(desktop, "Projet_python")
    os.makedirs(project_path, exist_ok=True)

    start_date = (datetime.today() - timedelta(days=365*5)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    output_folder = os.path.join(project_path, "data_tp1", "companies")
    ratios_path = os.path.join(project_path, "data_tp1", "ratios.csv")

    # Partie 1 : Ratios financiers
    scrape_financial_ratios(ratios_path)

    # Partie 2 : Variations journalières
    success_count = 0
    failed_companies = []

    print("▶ Début du scraping des variations boursières...")
    for company_name, symbol in companies.items():
        success, name, result = fetch_stock_variations(company_name, symbol, start_date, end_date, output_folder)
        if success:
            success_count += 1
            print(f"✅ Données exportées pour {name} -> {result}")
        else:
            print(f"❌ Erreur pour {name} : {result}")
            failed_companies.append(name)

    print("\n📢 **Récapitulatif final**")
    print(f"✅ Nombre de fichiers exportés avec succès : {success_count}/{len(companies)}")

    if failed_companies:
        print("\n❗ Entreprises non récupérées :")
        for failed in failed_companies:
            print(f"  - {failed}")
    else:
        print("\n🎉 Toutes les entreprises ont été récupérées avec succès !")

if __name__ == "__main__":
    main()