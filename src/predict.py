# src/predict_pipeline.py
import joblib
import pandas as pd
from pathlib import Path

SENT_MODEL = 'models/sentiment.joblib'
OUT = Path('data/final_triage.csv')

def load_models():
    sentiment = joblib.load(SENT_MODEL)
    return sentiment

def extract_entities(nlp, text):
    if not nlp:
        return []
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append({'text': ent.text, 
                     'label': ent.label_, 
                     'start': ent.start_char, 
                     'end': ent.end_char})
    return ents

def main(in_csv='data/unified_clean.csv', out_csv=OUT):
    if not Path(in_csv).exists():
        print(f"[WARNING] Arquivo de entrada {in_csv} não encontrado.")
        return

    df = pd.read_csv(in_csv)
    
    if df.empty:
        print(f"[INFO] O dataframe de entrada {in_csv} está vazio. Nenhuma predição será feita.")
        # Opcional: salva um CSV vazio com as colunas esperadas se necessário
        df.to_csv(out_csv, index=False)
        return

    sentiment = load_models()
    # Garante que text_clean existe e não tem nulos para a predição
    df['text_clean'] = df.get('text_clean', df.get('text', '')).astype(str).fillna('')
    
    # Filtra novamente para garantir que não enviamos nulos/vazios para o modelo
    # mas mantemos o índice original se possível para não perder linhas no df final
    texts = df['text_clean'].tolist()
    
    if not texts:
        print("[INFO] Nenhuma linha de texto válida para predição.")
        df.to_csv(out_csv, index=False)
        return

    preds_sent = sentiment.predict(texts)
    probs_sent = sentiment.predict_proba(texts)
    
    df['sentiment'] = preds_sent
    df['sentiment_scores'] = [dict(zip(sentiment.classes_, p)) for p in probs_sent]
    df.to_csv(out_csv, index=False)
    print('predictions saved to', out_csv)

if __name__ == '__main__':
    main()



