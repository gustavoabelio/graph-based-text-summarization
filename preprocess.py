import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Garantir que os recursos necessários do NLTK estejam disponíveis
resources = [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
]

for resource, path in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

def carregar_texto(caminho: str) -> str:
    with open(caminho, 'r', encoding='utf-8') as f:
        return f.read()

def dividir_sentencas(texto: str) -> list:
    return sent_tokenize(texto)

def limpar_sentenca(sentenca: str) -> str:
    """Realiza pré-processamento simples em uma sentença."""
    sentenca = sentenca.lower()

    sentenca = "".join([char for char in sentenca if char not in string.punctuation])

    stops = set(stopwords.words('portuguese'))
    palavras = word_tokenize(sentenca)
    palavras_filtradas = [p for p in palavras if p not in stops]
    
    return " ".join(palavras_filtradas)