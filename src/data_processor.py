# src/data_processor.py

'''
Processador de Dados (Camada Silver):
- Consolida arquivos da Camada Bronze (Staging)
- Realiza limpeza, deduplicação e normalização.
- Salva o dataset limpo para o Pipeline de NLP.
'''

import pandas as pd
import logging
from pathlib import Path

# Configuração de Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataProcessor')

# Caminhos
BRONZE_DIR = Path('data/staging/email')
SILVER_DIR = Path('data/silver')
SILVER_DIR.mkdir(parents=True, exist_ok=True)

def run_silver_processing():
    logger.info(f'Iniciando processamento da Camada Silver a partir de {BRONZE_DIR}...')
    
    # 1. Coleta todos os CSVs gerados pela ingestão
    all_files = list(BRONZE_DIR.glob('*.csv'))
    
    if not all_files:
        logger.warning('Nenhum arquivo encontrado para processar na Camada Bronze.')
        return None

    logger.info(f'Encontrados {len(all_files)} arquivos para processar.')

    # 2. Consolidação (Lendo todos os arquivos e unindo em um único DataFrame)
    dfs = []
    for f in all_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            logger.error(f'Erro ao ler arquivo {f}: {e}')

    df_combined = pd.concat(dfs, ignore_index=True)

    # 3. Limpeza e Deduplicação (Engenharia de Dados Elite)
    initial_count = len(df_combined)
    
    # Primeiro: Remove duplicatas baseadas no ID (response_id do Typeform)
    df_combined = df_combined.drop_duplicates(subset=['id'], keep='last')
    
    # Segundo: Limpeza de texto: remove espaços em branco extras e garante strings
    df_combined['text'] = df_combined['text'].fillna('').astype(str).str.strip()
    df_combined['subject'] = df_combined['subject'].fillna('Sem Assunto').str.strip()

    # Terceiro: Deduplicação Semântica (Crítico para evitar Data Leakage)
    # Remove textos idênticos, mantendo apenas a primeira ocorrência
    df_combined = df_combined.drop_duplicates(subset=['text'], keep='first')
    
    final_count = len(df_combined)
    logger.info(f'Consolidação concluída. Registros: {initial_count} -> {final_count} (Removidas {initial_count - final_count} duplicatas/IDs)')

    # 4. Salvando na Camada Silver (Parquet - Melhor para performance e tipos)
    output_path = SILVER_DIR / 'processed_data.parquet'
    
    # Nota: Parquet preserva os tipos de dados melhor que CSV
    df_combined.to_parquet(output_path, index=False)
    
    # Também salvamos um CSV para fácil visualização humana se necessário
    df_combined.to_csv(SILVER_DIR / 'processed_data.csv', index=False)
    
    logger.info(f'Dados limpos salvos com sucesso em: {output_path}')
    return output_path

if __name__ == '__main__':
    run_silver_processing()
