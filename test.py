# https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/bert-large-nli-cls-token.zip
from sentence_transformers import SentenceTransformer
import scipy

# embedder = SentenceTransformer('/home/peter/dev/sentence-transformers/models/bert-large-nli-cls-token')
embedder = SentenceTransformer('/home/peter/dev/sentence-transformers/models/bert-base-nli-mean-tokens')

# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating Italian food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'what is your age?',
          'what is the population of Cornwall?',
          'who is the prime minister of the UK?',
          'A cheetah is running behind its prey.']

corpus_embeddings = embedder.encode(corpus)

queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.', 
           'how old are you?', 'how many people live in Cornwall?', 'who is the pm of England?']
query_embeddings = embedder.encode(queries)

threshold = 0.75
print("threshold set at {}".format(threshold))
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
    print("Query: {}".format(query))
    results = [item for item in zip(corpus, distances)]
    results.sort(key = lambda t: t[1])
    for text, cos in results:
        score = 1.0 - cos
        if score > threshold:
            print("{}  Score: {}".format(text, round(score, 4)))
    print()
