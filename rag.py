import os
import json
import faiss
import time
import numpy as np
from tqdm import tqdm

from lamini.api.embedding import Embedding
from lamini import MistralRunner

from directory_helper import DirectoryLoader

class QueryEngine:
    def __init__(self, index, k=5):
        self.index = index
        self.k = k
        self.model = MistralRunner()

    def answer_question(self, question):
        most_similar = self.index.query(question, k=self.k)
        prompt = "\n".join(reversed(most_similar)) + "\n\n" + question
        print("------------------------------ Prompt ------------------------------\n" + prompt + "\n----------------------------- End Prompt -----------------------------")
        return self.model(prompt)

class LaminiIndex:
    def __init__(self, loader=None):
        self.loader = loader
        if loader is not None:
            self.build_index()

    @staticmethod
    def load_index(path):
        lamini_index = LaminiIndex()
        faiss_path = os.path.join(path, "index.faiss")
        splits_path = os.path.join(path, "splits.json")
        lamini_index.index = faiss.read_index(faiss_path)
        with open(splits_path, "r") as f:
            lamini_index.splits = json.load(f)
        return lamini_index

    def build_index(self):
        self.splits = []
        self.index = None
        for split_batch in tqdm(self.loader):
            embeddings = self.get_embeddings(split_batch)
            if self.index is None:
                self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(embeddings)
            self.splits.extend(split_batch)

    def get_embeddings(self, examples):
        ebd = Embedding()
        embeddings = ebd.generate(examples)
        embedding_list = [embedding[0] for embedding in embeddings]
        return np.array(embedding_list)

    def query(self, query, k=5):
        embedding = self.get_embeddings([query])[0]
        embedding_array = np.array([embedding])
        _, indices = self.index.search(embedding_array, k)
        return [self.splits[i] for i in indices[0]]

    def save_index(self, path):
        faiss_path = os.path.join(path, "index.faiss")
        splits_path = os.path.join(path, "splits.json")
        faiss.write_index(self.index, faiss_path)
        with open(splits_path, "w") as f:
            json.dump(self.splits, f)

class RetrievalAugmentedRunner:
    def __init__(self, k=5):
        self.k = k

    def train(self, data_path):
        self.loader = DirectoryLoader(data_path)
        self.index = LaminiIndex(self.loader)

    def __call__(self, query):
        query_engine = QueryEngine(self.index, k=self.k)
        return query_engine.answer_question(query)

def main():
    model = RetrievalAugmentedRunner()
    model.train(data_path="data")
    print(model("Have we invested in any generative AI companies in 2023?"))
    while True:
        prompt = input("Enter an investment question prompt: ")
        start = time.time()
        print(model(prompt))
        print("Time taken: ", time.time() - start)

main()