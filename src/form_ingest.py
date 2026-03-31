# src/form_ingest.py
import os
import uuid
import argparse
import requests
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timezone
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

from src.schemas import RawMessage

load_dotenv()

# --- Configurações de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TypeformIngest")

# --- Configurações de Ambiente ---
STAGING = Path('data/staging/email')
CURSOR_FILE = Path('data/staging/.typeform_cursor')
STAGING.mkdir(parents=True, exist_ok=True)

TOKEN = os.getenv('TYPEFORM_TOKEN')
FORM_ID = os.getenv('TYPEFORM_FORM_ID')
BASE_URL = 'https://api.typeform.com'

# --- Keywords para Auto-Mapeamento (Heurística de Engenharia de Dados) ---
KEYWORDS = {
    'name': ['nome', 'name', 'quem é você', 'contato'],
    'email': ['email', 'e-mail', 'seu melhor email'],
    'subject': ['assunto', 'subject', 'o que se trata'],
    'message': ['mensagem', 'message', 'texto', 'comentário', 'descreva']
}

def get_answer_value(ans: dict) -> str:
    ans_type = ans.get('type', '')
    if ans_type in ['text', 'email', 'number']:
        return str(ans.get(ans_type, ''))
    elif ans_type == 'choice':
        return str(ans.get('choice', {}).get('label', ''))
    elif ans_type == 'choices':
        return ', '.join(ans.get('choices', {}).get('labels', []))
    return str(ans.get(ans_type, ''))

def get_field_mapping(fields: list) -> dict:
    '''Cruza as definições do Typeform com as Keywords e Refs.'''
    mapping = {}
    logger.info("Detectando campos no formulário...")
    for f in fields:
        f_id = f.get('id')
        f_ref = f.get('ref', '').lower()
        f_title = f.get('title', '').lower()
        
        # Tenta mapear o papel do campo (ex: se é o campo de email)
        detected_role = None
        for role, keywords in KEYWORDS.items():
            if f_ref in keywords or any(k in f_title for k in keywords):
                detected_role = role
                break
        
        if detected_role:
            mapping[f_id] = detected_role
            logger.info(f"-> Mapeado ID {f_id} como '{detected_role}' (Título: '{f_title}')")
        else:
            logger.warning(f"-> Campo ignorado ou não mapeado: {f_title}")
    
    return mapping

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_data(url, params=None):
    headers = {'Authorization': f'Bearer {TOKEN}'}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def ingest(max_responses: int = 100, reset_cursor: bool = False):
    try:
        # 1. Carrega Mapa de Definição
        form_def = fetch_data(f'{BASE_URL}/forms/{FORM_ID}')
        field_map = get_field_mapping(form_def.get('fields', []))
        
        # 2. Busca Respostas
        after = None if reset_cursor else (CURSOR_FILE.read_text().strip() if CURSOR_FILE.exists() else None)
        params = {'page_size': max_responses, 'sort': 'submitted_at,asc'}
        if after: params['after'] = after
        
        data = fetch_data(f'{BASE_URL}/forms/{FORM_ID}/responses', params=params)
        items = data.get('items', [])

        if not items:
            logger.info('Sem novas respostas.')
            return

        ingested = 0
        last_token = None

        for item in items:
            response_id = item.get('response_id')
            answers = item.get('answers', [])
            
            # Dicionário temporário para extração
            extracted = {'id': response_id, 'received_at': item.get('submitted_at')}
            
            for ans in answers:
                f_id = ans.get('field', {}).get('id')
                role = field_map.get(f_id)
                if role:
                    extracted[role] = get_answer_value(ans)

            # Prepara texto para NLP
            subject = extracted.get('subject', 'Sem Assunto')
            message = extracted.get('message', '')
            text = f'{subject}: {message}'.strip() if subject and message else (message or subject or '')

            # Valida com Pydantic (Garante Qualidade Bronze)
            msg = RawMessage(
                id=f"form-{response_id}",
                from_name=extracted.get('name'),
                from_email=extracted.get('email'),
                subject=subject,
                text=text,
                received_at=extracted.get('received_at'),
                message_id=extracted.get('email') or response_id
            )

            # Salva
            fname = STAGING / f'form-{response_id}.csv'
            pd.DataFrame([msg.to_row()]).to_csv(fname, index=False)
            ingested += 1
            last_token = response_id

        if last_token:
            CURSOR_FILE.write_text(last_token)

        logger.info(f'Ingestão concluída. {ingested} arquivos salvos.')

    except Exception as e:
        logger.error(f'Erro fatal na ingestão: {e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=100)
    parser.add_argument('--reset-cursor', action='store_true')
    args = parser.parse_args()
    ingest(max_responses=args.max, reset_cursor=args.reset_cursor)
