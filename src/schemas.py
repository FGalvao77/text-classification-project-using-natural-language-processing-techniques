# src/schemas.py
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime

class RawMessage(BaseModel):
    '''Contrato de dados para a Camada Bronze (Staging)'''
    id: str
    channel: str = "form"
    sender_name: Optional[str] = Field(None, alias="from_name")
    sender_email: Optional[str] = Field(None, alias="from_email")
    subject: str = "Sem Assunto"
    text: str
    received_at: datetime
    message_id: str

    def to_row(self):
        '''Converte para o formato de linha do CSV/Parquet esperado pelo pipeline'''
        return {
            'id': self.id,
            'channel': self.channel,
            'from': f"{self.sender_name} <{self.sender_email}>" if self.sender_email else self.sender_name,
            'to': 'pipeline@nlp.ai',
            'subject': self.subject,
            'text': self.text,
            'received_at': self.received_at.isoformat(),
            'message_id': self.message_id
        }
