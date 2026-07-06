import os
from preprocess import carregar_texto, dividir_sentencas, limpar_sentenca
from textrank import calcular_matriz_similaridade, aplicar_pagerank, gerar_resumo
from graph import salvar_visualizacao

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(BASE_DIR, "dataset", "texto.txt")
    output_folder = os.path.join(BASE_DIR, "output")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("--- Iniciando Processamento ---")
    texto = carregar_texto(input_file)
    sentencas_originais = dividir_sentencas(texto)
    print(f"Total de sentenças: {len(sentencas_originais)}")

    sentencas_limpas = [limpar_sentenca(s) for s in sentencas_originais]

    matriz_sim = calcular_matriz_similaridade(sentencas_limpas)
    grafo, scores = aplicar_pagerank(matriz_sim)

    resumo = gerar_resumo(sentencas_originais, scores, n=3)

    print("\n--- Scores das Sentenças ---")
    for i, score in scores.items():
        print(f"Sentença {i}: {score:.4f}")

    print("\n--- Resumo Gerado ---")
    print(resumo)

    with open(f"{output_folder}/resumo.txt", "w", encoding="utf-8") as f:
        f.write(resumo)
    
    salvar_visualizacao(grafo, scores, f"{output_folder}/grafo.png")
    print(f"\nResultados salvos na pasta '{output_folder}'")

if __name__ == "__main__":
    main()