from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def check_accuracy_semantic(generated_response, expected_response):
    if generated_response is None:
        return False, 0.0 
    
    gen_embedding = model.encode(generated_response)
    exp_embedding = model.encode(expected_response)
    similarity = cosine_similarity([gen_embedding], [exp_embedding])[0][0]
    
    return similarity >= 0.7, similarity
