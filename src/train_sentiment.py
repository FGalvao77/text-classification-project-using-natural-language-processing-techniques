# src/train_sentiment.py
import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

_NEGATIVE_KEYWORDS = ['reclamação', 'defeito', 'incorreto', 'problema', 
                      'cancelamento', 'promoção não aplicada', 'ruim', 
                      'insatisfeito', 'devolução', 'atraso', 'cobrança indevida', 
                      'erro', 'demora', 'atendimento ruim']
_POSITIVE_KEYWORDS = ['parabenizo', 'positiva', 'agradável', 'gostei', 'elogio', 
                      'satisfação', 'correto', 'resolvido', 'promoção aplicada', 
                      'bom', 'satisfeito']

def _infer_label(subject: str) -> str:
    s = str(subject).lower()
    if any(k in s for k in _NEGATIVE_KEYWORDS):
        return 'negativo'
    if any(k in s for k in _POSITIVE_KEYWORDS):
        return 'positivo'
    return 'neutro'

def load_data(path: str):
    df = pd.read_csv(path)

    if 'label' not in df.columns and 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})

    # Garante que a coluna de texto principal existe e é do tipo string
    if 'text_clean' not in df.columns:
        if 'text' in df.columns:
            df['text_clean'] = df['text'].astype(str)
        else:
            raise ValueError("O CSV deve conter pelo menos a coluna 'text' ou 'text_clean'.")
    else:
        df['text_clean'] = df['text_clean'].astype(str)

    # Limpeza básica de nulos antes de inferir ou treinar
    df = df.dropna(subset=['text_clean'])
    df = df[df['text_clean'].str.strip() != '']
    df = df[df['text_clean'].str.lower() != 'nan']

    if 'label' not in df.columns:
        print("[INFO] Coluna 'label' não encontrada. Inferindo a partir de text_clean...")
        df['label'] = df['text_clean'].apply(_infer_label)
        # Salva o dataframe limpo e com labels
        df.to_csv(path, index=False)
        print(f"[INFO] CSV atualizado com coluna 'label' salvo em {path}")
        print(df['label'].value_counts().to_string())

    return df

def main(data_path, out_model):
    df = load_data(data_path)
    
    # Limpeza final e rigorosa
    df = df.dropna(subset=['label', 'text_clean'])
    
    # Garante que text_clean é string e não contém nulos disfarçados
    df['text_clean'] = df['text_clean'].astype(str).fillna('')
    df = df[df['text_clean'].str.strip() != '']
    df = df[df['text_clean'].str.lower() != 'nan']

    X = df['text_clean'].tolist()
    y = df['label'].astype(str).tolist()

    if not X:
        print("[ERROR] Nenhum dado válido para treinamento após a limpeza.")
        return

    if len(X) < 10:
        print(f"[WARNING] Poucos dados para treinamento ({len(X)} amostras).")
        # Se houver pouquíssimos dados, o split estratificado pode falhar.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            stratify=y, random_state=42)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=1)), # min_df=1 para datasets pequenos
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, digits=4))
    
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump(pipe, out_model)
    print(f'Modelo salvo em {out_model}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, 
                        help='CSV com colunas text, label')
    parser.add_argument('--out', default='models/sentiment.joblib', 
                        help='path para salvar modelo')
    args = parser.parse_args()
    main(args.data, args.out)












