import json
import os
import hnswlib
from sentence_transformers import SentenceTransformer
import pre_processing

class BuscaVetorialModerna:

    """
    Implementa busca semântica utilizando Sentence Embeddings
    e índice HNSW.

    Complexidade:
    - Geração dos embeddings: O(n)
    - Construção do índice HNSW: aproximadamente O(n log n)
    - Busca: aproximadamente O(log n)
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        self.movies = []

        self.model = None

        self.embeddings = None

        self.index = None

    def load_data(self):

        print("Carregando filmes...")

        if not os.path.exists(self.dataset_path):

            print("Dataset não encontrado. Executando pré-processamento")

            pre_processing.preprocessar_dados()

        with open(self.dataset_path, "r", encoding="utf-8") as arquivo:

            self.movies = json.load(arquivo)

        print(f"{len(self.movies)} filmes carregados.")
    
    def load_model(self):

        print("Carregando modelo Sentence Transformer...")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Modelo carregado.")
    
    def generate_embeddings(self):

        print("Gerando embeddings...")

        textos = []

        for movie in self.movies:

            textos.append(movie["plot_limpo"])

        self.embeddings = self.model.encode(
        textos,
        show_progress_bar=True,
        convert_to_numpy=True
)

        print("Embeddings gerados.")
    
    def create_index(self):

        print("Criando índice HNSW...")

        dimensao = self.embeddings.shape[1]

        self.index = hnswlib.Index(space="cosine", dim=dimensao)

        self.index.init_index(
            max_elements=len(self.embeddings),
            ef_construction=200,
            M=16
        )

        self.index.add_items(self.embeddings)
        self.index.set_ef(50)
        self.index.set_num_threads(4)
        print("Índice criado com sucesso!")
    
    def search(self, query, top_k=5):

        import time

        if self.index is None:
            print("O índice HNSW ainda não foi criado.")
            return []

        inicio = time.time()

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        indices, distances = self.index.knn_query(
            query_embedding,
            k=top_k
        )

        fim = time.time()

        print(f"Tempo da busca: {fim - inicio:.4f} segundos")

        resultados = []


        for indice, distancia in zip(indices[0], distances[0]):

            filme = self.movies[indice]

            resultados.append({
                "titulo": filme.get("movie_name", "Desconhecido"),
                "similaridade": round(1 - float(distancia), 4),
                "sinopse_trecho": filme.get("plot_limpo", "")[:200]
            })

        return resultados

if __name__ == "__main__":

    caminho_dataset = "data/dataset_limpo.json"
   
    buscador = BuscaVetorialModerna(caminho_dataset)

    buscador.load_data()

    buscador.load_model()

    buscador.generate_embeddings()

    buscador.create_index()

    pergunta = input("\nDigite uma pergunta sobre filmes: ")

    resultados = buscador.search(pergunta)

    # Exibe os resultados
    print("\nResultados encontrados:\n")

    for i, filme in enumerate(resultados, start=1):

        print(f"{i}. {filme['titulo']}")
        print(f"   Similaridade: {filme['similaridade']}")
        print(f"   Sinopse: {filme['sinopse_trecho']}")
        print("-" * 60)