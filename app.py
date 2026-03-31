import streamlit as st
import pandas as pd
from src.predictor import UnifiedPredictor
from pathlib import Path

# Configuração da Página
st.set_page_config(
    page_title="NLP AI Classifier",
    page_icon="🤖",
    layout="wide"
)

# Inicialização do Predictor (Cache para performance)
@st.cache_resource
def load_predictor():
    return UnifiedPredictor()

predictor = load_predictor()

# --- BARRA LATERAL (Insights de Dados) ---
st.sidebar.title("📊 Painel de Controle")
st.sidebar.info("Projeto: NLP Text Classification Pipeline")

# Estatísticas Rápidas
gold_path = Path('data/processed/master_dataset.csv')
if gold_path.exists():
    df = pd.read_csv(gold_path)
    st.sidebar.subheader("Saúde dos Dados (Gold)")
    st.sidebar.metric("Exemplos Únicos", len(df))
    st.sidebar.write("Distribuição por Categoria:")
    st.sidebar.bar_chart(df['subject'].value_counts())
else:
    st.sidebar.warning("Pipeline de dados não executado.")

# --- CORPO PRINCIPAL ---
st.title("🤖 Classificador de Atendimento Inteligente")
st.markdown("""
    Esta interface utiliza um **Pipeline Híbrido** (Modelo Local Scikit-Learn + Fallback LLM) 
    para classificar mensagens de clientes com precisão industrial.
""")

# Área de Entrada
st.subheader("📝 Entrada de Mensagem")
user_input = st.text_area("Cole aqui o texto da mensagem ou dúvida do cliente:", 
                         placeholder="Ex: Não consigo acessar minha fatura, o boleto está com erro...",
                         height=150)

# Opções de Inferência
col1, col2 = st.columns(2)
with col1:
    force_llm = st.toggle("Forçar IA Generativa (GPT-4o-mini)", value=False, help="Ignora o modelo local e usa LLM para máxima precisão.")

with col2:
    threshold = st.slider("Confiança Mínima para Modelo Local", 0.0, 1.0, 0.60, 0.05)

if st.button("🚀 Classificar Agora", type="primary"):
    if user_input.strip() == "":
        st.warning("Por favor, insira um texto para análise.")
    else:
        with st.spinner("Analisando semântica..."):
            # Executa Classificação
            label, confidence, origin = predictor.classify(user_input, force_llm=force_llm)
            
            # Exibição do Resultado
            st.divider()
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric("Categoria Detectada", label.upper())
            
            with res_col2:
                # Cor do progresso baseada na confiança
                color = "green" if confidence > threshold else "orange" if confidence > 0.3 else "red"
                st.write(f"**Confiança:** {confidence:.2%}")
                st.progress(confidence)
            
            with res_col3:
                st.metric("Método de Origem", origin)

            # Feedback Visual
            if confidence < threshold and origin == "Local Model":
                st.warning("⚠️ Confiança abaixo do limite configurado. Recomenda-se revisão humana ou uso de LLM.")
            elif confidence > 0.8:
                st.success("✅ Classificação com alta confiabilidade.")

# Footer
st.divider()
st.caption("Desenvolvido sob mentoria de Data Engineering & AI Specialists.")
