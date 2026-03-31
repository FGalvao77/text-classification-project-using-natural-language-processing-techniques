# src/merge_messages.py
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MergeMessages")

# Caminhos de Dados (Seguindo a estrutura do projeto)
INPUT_DIR = Path('data/staging/email')
OUTPUT_DIR = Path('data/processed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def merge_and_prepare_dataset(test_size=0.2, random_state=42):
    '''
    Unifica CSVs individuais em um dataset mestre para ML.
    Realiza limpeza básica e split de treino/teste.
    '''
    logger.info(f'Buscando arquivos CSV em {INPUT_DIR}...')
    
    csv_files = list(INPUT_DIR.glob('*.csv'))
    
    if not csv_files:
        logger.warning('Nenhum arquivo CSV encontrado para unificar.')
        return None

    logger.info(f'Encontrados {len(csv_files)} arquivos. Iniciando unificação...')

    # 1. Leitura e Consolidação
    dfs = []
    for f in csv_files:
        try:
            df_tmp = pd.read_csv(f)
            dfs.append(df_tmp)
        except Exception as e:
            logger.error(f'Erro ao ler {f.name}: {e}')

    df_full = pd.concat(dfs, ignore_index=True)

    # 2. Limpeza de Engenharia de Dados (Silver Layer)
    initial_len = len(df_full)
    
    # Deduplicação: Garante IDs únicos e TEXTOS únicos (crucial para ML de alta qualidade)
    df_full = df_full.drop_duplicates(subset=['id'])
    df_full = df_full.drop_duplicates(subset=['text'])
    
    # Tratamento de Nulos
    df_full['text'] = df_full['text'].fillna('').astype(str).str.strip()
    df_full['subject'] = df_full['subject'].fillna('Sem Assunto').str.strip()
    
    # Filtro de Qualidade: Remove registros onde o texto é muito curto/vazio
    df_full = df_full[df_full['text'].str.len() > 5]
    
    logger.info(f'Unificação concluída. Registros: {initial_len} -> {len(df_full)} (Deduplicados/Limpos)')

    # 3. Exportação do Dataset Mestre (Parquet é melhor para tipos, mas CSV é bom para debug)
    master_path = OUTPUT_DIR / 'master_dataset.csv'
    df_full.to_csv(master_path, index=False)
    logger.info(f'Dataset mestre salvo em: {master_path}')

    # 4. Split Treino/Teste (Opcional, mas útil para o Cientista de Dados)
    if len(df_full) > 5:  # Só faz split se tiver dados suficientes
        train, test = train_test_split(df_full, test_size=test_size, random_state=random_state)
        
        train_path = OUTPUT_DIR / 'train.csv'
        test_path = OUTPUT_DIR / 'test.csv'
        
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        
        logger.info(f'Split de ML criado: Train ({len(train)}) | Test ({len(test)})')
        logger.info(f'Arquivos salvos em: {OUTPUT_DIR}')
    else:
        logger.warning('Dados insuficientes para criar split de Treino/Teste.')

    return master_path

if __name__ == '__main__':
    merge_and_prepare_dataset()
