# cloud_migration.py
from pinecone import Pinecone
import pymongo
from tqdm import tqdm
import faiss
import numpy as np
import json

def migrate_to_cloud(pinecone_api_key, mongo_uri):
    """
    Migre les données locales vers le cloud
    """
    # Initialiser Pinecone avec la nouvelle API
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Créer un index Pinecone si nécessaire
    index_name = "my-docs"
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-west-2"}}
        )
    
    index = pc.Index(index_name)
    
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
    
    # Nettoyer les collections existantes
    meta_collection.delete_many({})
    index.delete(delete_all=True)
    
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
    print(f"- {len(vectors)} vecteurs migrés vers Pinecone")
    print(f"- {len(metadata)} documents sauvegardés dans MongoDB")

if __name__ == "__main__":
    pinecone_api_key = "pcsk_9coPz_9U1G4VhXwbfPKYapvRGdXQhCBMhxU7LYwbZny8g5hGmemT7pUGaj3ZV6gKFwaWF"
    mongo_uri = "mongodb+srv://dorianmarty:Marty2024!@clustergdp.mq5yk.mongodb.net/?retryWrites=true&w=majority&appName=ClusterGDP"
    migrate_to_cloud(pinecone_api_key, mongo_uri)
