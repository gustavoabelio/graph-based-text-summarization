import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def calcular_matriz_similaridade(sentencas_limpas: list) -> np.ndarray:
    """Calcula a similaridade do cosseno entre todas as sentenças usando representação TF-IDF."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentencas_limpas)
    
    return cosine_similarity(tfidf_matrix)

def aplicar_pagerank(matriz_similaridade: np.ndarray):
    """Constrói o grafo ponderado e aplica o algoritmo PageRank."""
    grafo = nx.from_numpy_array(matriz_similaridade)
    
    grafo.remove_edges_from(nx.selfloop_edges(grafo))
    
    scores = nx.pagerank(grafo, weight='weight')
    
    return grafo, scores

def gerar_resumo(sentencas_originais: list, scores: dict, n: int = 3) -> str:
    """Seleciona as N sentenças com maiores scores e as ordena na ordem original."""
    ranking = sorted(((scores[i], s, i) for i, s in enumerate(sentencas_originais)), reverse=True)
    
    top_n = ranking[:n]
    
    top_n_ordenado = sorted(top_n, key=lambda x: x[2])
    
    resumo = [s for score, s, index in top_n_ordenado]
    return " ".join(resumo)