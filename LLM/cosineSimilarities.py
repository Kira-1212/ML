from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

def get_embedding(model_name, text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

sent1 = "I love natural language processing."
sent2 = "I adore NLP."

emb_wordpiece = get_embedding("bert-base-uncased", sent1)
emb_bpe = get_embedding("gpt2", sent1)

similarity = cosine_similarity(emb_wordpiece, emb_bpe)
print("Cosine similarity between WordPiece and BPE embeddings:", similarity)
