import joblib
import logging
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Predictor")

# Caminhos
MODEL_PATH = Path('model/text_classifier_v1.joblib')

class UnifiedPredictor:
    def __init__(self):
        self.model = None
        self.openai_client = None
        
        # 1. Carregar Modelo Local
        if MODEL_PATH.exists():
            try:
                self.model = joblib.load(MODEL_PATH)
                logger.info("✅ Modelo Local (v1) carregado com sucesso.")
            except Exception as e:
                logger.error(f"Erro ao carregar modelo local: {e}")
        
        # 2. Configurar OpenAI (se disponível)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("✅ Integração OpenAI configurada como fallback de alta precisão.")

    def _predict_local(self, text: str):
        if not self.model: return None, 0.0
        
        # Predição e Probabilidades
        probas = self.model.predict_proba([text])[0]
        max_idx = probas.argmax()
        label = self.model.classes_[max_idx]
        confidence = probas[max_idx]
        
        return label, confidence

    def _predict_llm(self, text: str):
        if not self.openai_client: return None
        
        prompt = f"""
        Classifique a seguinte mensagem de atendimento ao cliente em uma das categorias:
        - suporte (dúvidas técnicas, bugs)
        - financeiro (pagamentos, notas fiscais)
        - cancelamento (pedidos de encerramento)
        - reclamação (críticas ao serviço/atrasos)
        - outro (assuntos gerais)

        Mensagem: "{text}"
        Retorne APENAS o nome da categoria.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            logger.error(f"Erro na API OpenAI: {e}")
            return None

    def classify(self, text: str, force_llm: bool = False):
        '''
        Lógica Híbrida Profissional:
        - Se force_llm for True ou o modelo local tiver baixa confiança (< 0.6), usa LLM.
        - Caso contrário, usa o modelo local para performance e baixo custo.
        '''
        
        # 1. Predição Local
        label_local, conf = self._predict_local(text)
        
        # 2. Decisão de Escalonamento (Threshold de 60% de confiança)
        if (force_llm or conf < 0.6) and self.openai_client:
            logger.info(f"Escalonando para LLM (Confiança Local Baixa: {conf:.2f})")
            llm_label = self._predict_llm(text)
            if llm_label: return llm_label, 1.0, "LLM (Inference)"
            
        return label_local, conf, "Local Model"

if __name__ == "__main__":
    # Teste Rápido
    p = UnifiedPredictor()
    msg = "Não estou conseguindo logar no sistema, dá erro 404."
    cat, score, origin = p.classify(msg)
    print(f"\n--- Resultado de Teste ---")
    print(f"Texto: {msg}")
    print(f"Categoria: {cat} | Confiança: {score:.2f} | Origem: {origin}")
