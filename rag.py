import faiss
import time
import numpy as np
from tqdm import tqdm
from lamini.api.embedding import Embedding
from lamini import Lamini

from directory_helper import DirectoryLoader

class LaminiIndex:
    def __init__(self, loader):
        self.loader = loader
        self.build_index()

    def build_index(self):
        self.content_chunks = []
        self.index = None
        for chunk_batch in tqdm(self.loader):
            embeddings = self.get_embeddings(chunk_batch)
            if self.index is None:
                self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(embeddings)
            self.content_chunks.extend(chunk_batch)

    def get_embeddings(self, examples):
        ebd = Embedding()
        embeddings = ebd.generate(examples)
        embedding_list = [embedding[0] for embedding in embeddings]
        return np.array(embedding_list)

    def query(self, query, k=5):
        embedding = self.get_embeddings([query])[0]
        embedding_array = np.array([embedding])
        _, indices = self.index.search(embedding_array, k)
        return [self.content_chunks[i] for i in indices[0]]

class QueryEngine:
    def __init__(self, index, k=5):
        self.index = index
        self.k = k
        self.model = Lamini(model_name="mistralai/Mistral-7B-Instruct-v0.1")

    def answer_question(self, question):
        most_similar = self.index.query(question, k=self.k)
        prompt = "\n".join(reversed(most_similar)) + "\n\n" + question
        print("------------------------------ Prompt ------------------------------\n" + prompt + "\n----------------------------- End Prompt -----------------------------")
        return self.model.generate("<s>[INST]" + prompt + "[/INST]")

class RetrievalAugmentedRunner:
    def __init__(self, dir, k=5):
        self.k = k
        self.loader = DirectoryLoader(dir)

    def train(self):
        self.index = LaminiIndex(self.loader)

    def __call__(self, query):
        query_engine = QueryEngine(self.index, k=self.k)
        return query_engine.answer_question(query)

def main():
    model = RetrievalAugmentedRunner(dir="data")
    model.train()
    while True:
        prompt = input("\n\nEnter another investment question (e.g. Have we invested in any generative AI companies in 2023?): ")
        start = time.time()
        print(model(prompt))
        print("Time taken: ", time.time() - start)

main()