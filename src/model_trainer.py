import pandas as pd
import logging
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelTrainer")

# Caminhos
TRAIN_DATA = Path('data/processed/train.csv')
TEST_DATA = Path('data/processed/test.csv')
MODEL_DIR = Path('model')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_model():
    logger.info("--- Iniciando Treinamento do Modelo NLP ---")
    
    # 1. Carga de Dados
    if not TRAIN_DATA.exists() or not TEST_DATA.exists():
        logger.error("Arquivos de treino/teste não encontrados em data/processed/. Execute o pipeline primeiro.")
        return None

    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)

    # Garantir que não temos nulos no texto
    train_df['text'] = train_df['text'].fillna('')
    test_df['text'] = test_df['text'].fillna('')

    X_train, y_train = train_df['text'], train_df['subject']
    X_test, y_test = test_df['text'], test_df['subject']

    logger.info(f"Dados carregados: {len(X_train)} exemplos de treino, {len(X_test)} de teste.")

    # 2. Definição do Pipeline (Feature Engineering + Estimador)
    # Usamos n-grams (1,2) para capturar contextos como "não recebi" ou "dúvida técnica"
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    # 3. Treinamento
    logger.info("Treinando Logistic Regression com pesos balanceados...")
    pipeline.fit(X_train, y_train)

    # 4. Avaliação
    logger.info("Avaliando modelo no set de teste...")
    y_pred = pipeline.predict(X_test)
    
    report = classification_report(y_test, y_pred)
    logger.info(f"\nResultado da Classificação:\n{report}")

    # 5. Salvando Artefatos
    model_path = MODEL_DIR / 'text_classifier_v1.joblib'
    joblib.dump(pipeline, model_path)
    
    logger.info(f"✅ Modelo salvo com sucesso em: {model_path}")
    return model_path

if __name__ == '__main__':
    train_model()
