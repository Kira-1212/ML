from transformers import AutoTokenizer

# WordPiece tokenizer (used in BERT)
wordpiece_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# BPE tokenizer (used in GPT-2)
bpe_tokenizer = AutoTokenizer.from_pretrained("gpt2")

sentence = "A Cloud Digital Leader can articulate the capabilities of Google Cloud core products and services and how they benefit organizations. They can also describe common business use cases and how cloud solutions support an enterprise. This certification is for anyone who wishes to demonstrate their knowledge of cloud computing basics and how Google Cloud products and services can be used to achieve an organization’s goals."

# Encode and decode
wp_tokens = wordpiece_tokenizer.tokenize(sentence)
bpe_tokens = bpe_tokenizer.tokenize(sentence)

print("WordPiece Tokens (BERT):", wp_tokens)
print("BPE Tokens (GPT-2):", bpe_tokens)
