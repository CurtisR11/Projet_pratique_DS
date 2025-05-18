import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import pytz
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from collections import defaultdict
from matplotlib.lines import Line2D
import numpy as np

# Configuration
model_name = "ProsusAI_finbert"
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
project_path = os.path.join(desktop, "Projet_python_main")
os.makedirs(project_path, exist_ok=True)

# Paths for models and outputs
model_save_path = os.path.join(project_path, "ProsusAI_finbert_results")
os.makedirs(model_save_path, exist_ok=True)

# New directory for TP8 plots
tp8_output_dir = os.path.join(os.path.expanduser("~"), "Desktop","Projet_python", "output", "tp8_plots")
os.makedirs(tp8_output_dir, exist_ok=True)

def get_text_timestamps(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        news_data = json.load(file)

    texts = []
    timestamps = []

    for date, articles in news_data.items():
        for item in articles:
            try:
                utc_time = datetime.strptime(item['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                ny_time = utc_time.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('America/New_York'))
                rounded_time = ny_time.replace(minute=0, second=0, microsecond=0)

                title = item.get('title', '')
                description = item.get('description', '')
                full_text = f"{title} {description}".strip()

                if full_text:
                    texts.append(full_text)
                    timestamps.append(rounded_time)
            except Exception as e:
                print(f"Erreur sur un article : {e}")
                continue

    return texts, timestamps


def get_sentiment(model_path, texts):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    sentiments = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits, dim=1).item()
            sentiments.append(sentiment)
    return sentiments


def align_timestamps(timestamps):
    aligned = []
    for ts in timestamps:
        ts = pd.Timestamp(ts).tz_localize(None)
        market_open = ts.replace(hour=9, minute=30)
        market_close = ts.replace(hour=15, minute=0)
        if market_open <= ts < market_close:
            aligned.append(ts)
        elif market_close <= ts:
            aligned.append(market_close)
        else:
            prev_day = (ts - pd.Timedelta(days=1)).replace(hour=15, minute=0)
            aligned.append(prev_day)
    return aligned


def get_stock_data(ticker, start_date="2025-01-01"):  
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, interval="60m").reset_index()
    return df[["Datetime", "Close"]]


def plot_comparison(df, sentiment_a, sentiment_b, timestamps, title_a, title_b, company_name, output_dir):
    aligned_ts = align_timestamps(timestamps)
    # Ensure tz-naive
    df['Datetime'] = df['Datetime'].dt.tz_localize(None)
    aligned_naive = [ts.replace(tzinfo=None) for ts in aligned_ts]

    def group_sentiments(ts_list, sents):
        grouped = defaultdict(list)
        for ts, sent in zip(ts_list, sents):
            grouped[ts].append(sent)
        return grouped

    grouped_a = group_sentiments(aligned_naive, sentiment_a)
    grouped_b = group_sentiments(aligned_naive, sentiment_b)

    def get_points(grouped):
        pts = []
        for ts, s_list in grouped.items():
            for idx, sent in enumerate(s_list):
                pts.append((ts, sent, idx))
        return pts

    points_a = get_points(grouped_a)
    points_b = get_points(grouped_b)

    sentiment_colors = {2:'green',1:'gold',0:'red'}

    fig, axes = plt.subplots(1, 2, figsize=(18,6), sharey=True)
    for ax, pts, title in zip(axes, [points_a, points_b],[title_a,title_b]):
        ax.plot(df['Datetime'], df['Close'], label='Prix action')
        for ts, sent, idx in pts:
            nearest = df['Datetime'].iloc[(df['Datetime']-ts).abs().argsort()[0]]
            y = df.loc[df['Datetime']==nearest,'Close'].values
            if y.size:
                y_val = y[0] + (idx-0.5)*0.5
                ax.scatter(nearest, y_val, color=sentiment_colors[sent], s=80, edgecolor='black', zorder=5)
        ax.set_title(title)
    axes[0].set_ylabel('Prix de clôture')

    legend_elements = [
        Line2D([0],[0],color='blue',lw=2,label='Prix action'),
        Line2D([0],[0],marker='o',color='w',label='Positif',markerfacecolor='green',markersize=10,markeredgecolor='black'),
        Line2D([0],[0],marker='o',color='w',label='Neutre',markerfacecolor='gold',markersize=10,markeredgecolor='black'),
        Line2D([0],[0],marker='o',color='w',label='Négatif',markerfacecolor='red',markersize=10,markeredgecolor='black')
    ]
    axes[1].legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(output_dir, f"{company_name.replace(' ','_')}_tp8.png")
    fig.savefig(save_path)
    print(f"📸 Plot saved to {save_path}")
    plt.close(fig)


# Main loop for TP8
companies = {
    "Amazon": "AMZN",
    "Louis Vuitton (LVMH)": "MC.PA",
    "Tesla": "TSLA",
    "Samsung": "005930.KS",
    "Tencent": "TCEHY",
    "Alibaba": "BABA",
    "Sony": "SONY",
    "Adobe": "ADBE",
    "Johnson & Johnson": "JNJ",
    "Pfizer": "PFE",
}

for company_name, ticker in companies.items():
    adjusted = company_name.replace(' ', '_')
    news_path = os.path.join(os.path.expanduser("~"), "Desktop","Projet_python", "output", "news_tp6", f"{adjusted}_news.json")
    texts, timestamps = get_text_timestamps(news_path)
    sents_a = get_sentiment('ProsusAI/finbert', texts)
    sents_b = get_sentiment(model_save_path, texts)
    df_stock = get_stock_data(ticker)

    plot_comparison(
        df_stock, sents_a, sents_b, timestamps,
        "FinBERT (original)", "FinBERT (fine-tuné)",
        company_name, tp8_output_dir
    )
    print(f"Traitement de {company_name} terminé.")
