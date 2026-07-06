import matplotlib.pyplot as plt
import networkx as nx

def salvar_visualizacao(grafo, scores, caminho_saida: str):
    """Visualiza o grafo com os scores e salva a imagem."""
    plt.figure(figsize=(12, 8))
    
    pos = nx.spring_layout(grafo, k=0.5, seed=42)
    
    node_sizes = [v * 25000 for v in scores.values()]
    
    arestas_fortes = [(u, v) for (u, v, d) in grafo.edges(data=True) if d['weight'] > 0.1]
    pesos_fortes = [grafo[u][v]['weight'] * 20 for u, v in arestas_fortes]

    nx.draw_networkx_nodes(grafo, pos, node_size=node_sizes, node_color="skyblue", alpha=0.8)
    nx.draw_networkx_edges(grafo, pos, edgelist=arestas_fortes, width=pesos_fortes, edge_color="gray", alpha=0.5)
    nx.draw_networkx_labels(grafo, pos, font_size=10, font_family="sans-serif")

    plt.title("Visualização do Grafo de Sentenças (TextRank)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(caminho_saida)
    plt.close()