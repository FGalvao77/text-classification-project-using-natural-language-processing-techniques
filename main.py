# main.py

import logging
import argparse
from src.form_ingest import ingest
from src.data_processor import run_silver_processing
from src.merge_messages import merge_and_prepare_dataset
from src.model_trainer import train_model

# Configuração Global de Logging (Padrão de Elite)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NLP-Pipeline")

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Classificação de Texto NLP")
    parser.add_argument('--reset-cursor', action='store_true', help="Reinicia a ingestão do início.")
    args = parser.parse_args()

    logger.info("=== 🚀 INICIANDO PIPELINE DE DADOS (END-TO-END) ===")

    try:
        # FASE 1: INGESTÃO (CAMADA BRONZE)
        logger.info(f"--- Fase 1: Ingestão (Bronze) [Reset: {args.reset_cursor}] ---")
        ingest(max_responses=100, reset_cursor=args.reset_cursor)


        # FASE 2: PROCESSAMENTO TÉCNICO (CAMADA SILVER)
        # Consolida, limpa e salva em Parquet para performance
        logger.info('--- Fase 2: Processamento Técnico (Silver) ---')
        silver_path = run_silver_processing()

        # FASE 3: PREPARAÇÃO PARA MACHINE LEARNING (CAMADA GOLD)
        # Cria o dataset mestre e os arquivos de Treino/Teste (Train/Test Split)
        logger.info('--- Fase 3: Preparação para ML (Gold) ---')
        master_path = merge_and_prepare_dataset(test_size=0.2)

        if master_path:
            # FASE 4: TREINAMENTO DO MODELO (MACHINE LEARNING)
            # Treina, avalia e salva o classificador de texto
            logger.info('--- Fase 4: Treinamento e Avaliação do Modelo ---')
            model_path = train_model()
            
            if model_path:
                logger.info(f'✅ Pipeline concluído com sucesso!')
                logger.info(f'📍 Dataset Final: {master_path}')
                logger.info(f'📍 Modelo Salvo: {model_path}')
                logger.info(f'📍 Pasta de Modelagem: data/processed/')
            else:
                logger.warning('Pipeline de dados concluído, mas o treinamento do modelo falhou.')
        else:
            logger.warning('Pipeline finalizado, mas não houve dados suficientes para gerar o dataset de ML.')

    except Exception as e:
        logger.error(f'❌ Erro crítico no pipeline: {e}', exc_info=True)

    logger.info('=== ✅ PROCESSO ENCERRADO ===')

if __name__ == '__main__':
    main()
