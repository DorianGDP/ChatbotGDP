# cloud_migration.py
import pinecone
import pymongo
from tqdm import tqdm
import faiss
import numpy as np
import json

def migrate_to_cloud(pinecone_api_key, mongo_uri):
    """
    Migre les données locales vers le cloud
    """
    # Initialiser Pinecone
    pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")  # Environnement gratuit
    
    # Créer un index Pinecone si nécessaire
    if "my-docs" not in pinecone.list_indexes():
        pinecone.create_index("my-docs", dimension=1536)
    
    index = pinecone.Index("my-docs")
    
    # Charger les données locales
    faiss_index = faiss.read_index('embeddings_db/faiss_index.idx')
    with open('embeddings_db/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Extraire les vecteurs de FAISS
    vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)
    
    # Connexion à MongoDB
    client = pymongo.MongoClient(mongo_uri)
    db = client.chatbot_db
    meta_collection = db.metadata
    
    # Migration par lots
    batch_size = 100
    for i in tqdm(range(0, len(vectors), batch_size)):
        batch_vectors = vectors[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]
        
        # Préparer les données pour Pinecone
        vectors_to_upsert = []
        for j, vec in enumerate(batch_vectors):
            vectors_to_upsert.append({
                'id': str(batch_metadata[j]['id']),
                'values': vec.tolist(),
                'metadata': {'doc_id': str(batch_metadata[j]['id'])}
            })
        
        # Upserter dans Pinecone
        index.upsert(vectors=vectors_to_upsert)
        
        # Sauvegarder metadata dans MongoDB
        meta_collection.insert_many(batch_metadata)
    
    print("Migration terminée!")

if __name__ == "__main__":
    pinecone_api_key = "pcsk_9coPz_9U1G4VhXwbfPKYapvRGdXQhCBMhxU7LYwbZny8g5hGmemT7pUGaj3ZV6gKFwaWF"
    mongo_uri = "mongodb+srv://dorianmarty:Marty2024!@clustergdp.mq5yk.mongodb.net/?retryWrites=true&w=majority&appName=ClusterGDP"
    migrate_to_cloud(pinecone_api_key, mongo_uri)
