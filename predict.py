import argparse
import sys
from src.predictor import UnifiedPredictor

def main():
    parser = argparse.ArgumentParser(description="Classificador de Texto NLP (Inferência Inteligente)")
    parser.add_argument('text', type=str, nargs='?', help="O texto para classificar (opcional, se não houver lê do stdin)")
    parser.add_argument('--force-llm', action='store_true', help="Ignora o modelo local e usa GPT-4 (requer API KEY)")
    
    args = parser.parse_args()
    
    # Se não passar texto via argumento, tenta ler da entrada padrão (pipe)
    text = args.text
    if not text:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            parser.print_help()
            sys.exit(0)

    if not text:
        print("❌ Erro: Nenhum texto fornecido para classificação.")
        sys.exit(1)

    print(f"\n🔍 Analisando mensagem: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
    
    # Inicializa Predictor
    predictor = UnifiedPredictor()
    
    # Executa Classificação
    label, confidence, origin = predictor.classify(text, force_llm=args.force_llm)
    
    # Exibe Resultado de Elite
    print("-" * 40)
    print(f"🏷️  CATEGORIA: {label.upper() if label else 'N/A'}")
    print(f"📊 CONFIANÇA: {confidence:.2f}")
    print(f"🤖 MÉTODO:    {origin}")
    print("-" * 40)

if __name__ == "__main__":
    main()
