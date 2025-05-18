from datasets import load_dataset, concatenate_datasets, DatasetDict,ClassLabel
import numpy as np
import time
import torch
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from zipfile import ZipFile
import shutil
import os
#!pip install --upgrade datasets

def load_financial_dataset(test_size=0.2, seed=42):
    ds1 = load_dataset("zeroshot/twitter-financial-news-sentiment")
    ds2 = load_dataset("nickmuchi/financial-classification")

    train_ds1 = ds1["train"]
    train_ds2 = ds2["train"]

    train_ds2 = train_ds2.rename_column("labels", "label")

    class_labels = ClassLabel(num_classes=3)

    train_ds1 = train_ds1.cast_column("label", class_labels)
    train_ds2 = train_ds2.cast_column("label", class_labels)

    full_train = concatenate_datasets([train_ds1, train_ds2])

    split = full_train.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")

    return DatasetDict({
        "train": split["train"],
        "test": split["test"]
    })

dataset = load_financial_dataset()
print(dataset)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

def train_model(model_name, dataset, batch_size, num_epochs):

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    project_path = os.path.join(desktop, "Projet_python")
    os.makedirs(project_path, exist_ok=True)
    model_save_path = os.path.join(project_path, "ProsusAI_finbert_results")
    os.makedirs(model_save_path, exist_ok=True)
    dataset = load_financial_dataset()
    # 1. Charger le tokenizer et le modèle pré-entraînés
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # 2. Tokenizer le dataset
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    tokenized_train = dataset["train"].map(tokenize_function, batched=True)
    tokenized_test = dataset["test"].map(tokenize_function, batched=True)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 3. Définir les arguments d'entraînement
    output_dir = os.path.join(project_path, f"{model_name.replace('/', '_')}_results")
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        report_to="none",
        disable_tqdm=False,
        logging_first_step=True
    )

    # 4. Créer le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    # 5. Entraînement et évaluation
    trainer.train()
    eval_result = trainer.evaluate()
    print("Évaluation finale :", eval_result)

    #6. Saving the Model
    tokenizer.save_pretrained(model_save_path)
    model.save_pretrained(model_save_path)
    print(f"Modèle sauvegardé")




start_time = time.time()
train_model("ProsusAI/finbert", dataset, batch_size=32, num_epochs=3)
print (f"training time : {time.time() - start_time}")