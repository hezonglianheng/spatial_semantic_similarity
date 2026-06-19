# encoding: utf-8
# 注：使用sentence-transformers库减少开发量

from sentence_transformers import SentenceTransformer
import torch

import argparse

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SimilarityExtractor:
    def __init__(self, model_name_or_path, **kwargs):
        self.model = SentenceTransformer(model_name_or_path, device=DEVICE, **kwargs)

    def similarity(self, sentence1, sentence2, **kwargs):
        # Encode the sentences
        embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True, device=DEVICE, **kwargs)
        
        # Compute cosine similarity
        similarity = self.model.similarity(embeddings[0], embeddings[1])
        
        return similarity

def main():
    parser = argparse.ArgumentParser(description="Sentence Similarity Calculator")
    parser.add_argument("model", help="Path to the pre-trained model")
    parser.add_argument("sentence1", help="First sentence")
    parser.add_argument("sentence2", help="Second sentence")
    args = parser.parse_args()

    extractor = SimilarityExtractor(args.model)
    similarity = extractor.similarity(args.sentence1, args.sentence2)
    print(f"Similarity: {similarity}")

if __name__ == "__main__":
    main()