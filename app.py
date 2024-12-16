# app.py
import streamlit as st
import pinecone
import pymongo
from openai import OpenAI
import os

class CloudChatbot:
    def __init__(self, pinecone_api_key, openai_api_key, mongo_uri):
        # Initialisation des clients
        pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
        self.pinecone_index = pinecone.Index("my-docs")
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.mongo_client = pymongo.MongoClient(mongo_uri)
        self.db = self.mongo_client.chatbot_db
        
    def get_query_embedding(self, question):
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=question
        )
        return response.data[0].embedding
        
    def recherche_documents(self, question, k=3):
        # Obtenir l'embedding de la question
        query_embedding = self.get_query_embedding(question)
        
        # Rechercher dans Pinecone
        results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        # Récupérer les métadonnées complètes depuis MongoDB
        docs = []
        for match in results['matches']:
            doc_id = match['metadata']['doc_id']
            metadata = self.db.metadata.find_one({'id': doc_id})
            if metadata:
                metadata['score'] = match['score']
                docs.append(metadata)
                
        return docs
        
    def generer_reponse(self, question, docs):
        # Préparer le contexte
        context = "\n\n".join([
            f"Titre: {doc['title']}\nContenu: {doc['content']}\nURL: {doc['url']}"
            for doc in docs
        ])
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Tu es un assistant expert du site web."},
                {"role": "user", "content": f"Question: {question}\n\nContexte:\n{context}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def main():
    st.set_page_config(page_title="Assistant Site Web", layout="wide")
    
    # Configuration et initialisation
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    mongo_uri = st.secrets["MONGO_URI"]
    
    chatbot = CloudChatbot(pinecone_api_key, openai_api_key, mongo_uri)
    
    # Interface utilisateur
    st.title("Assistant Site Web 🤖")
    
    initialize_session_state()
    
    # Afficher l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Zone de saisie utilisateur
    if question := st.chat_input("Posez votre question..."):
        # Ajouter la question à l'historique
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
            
        # Générer et afficher la réponse
        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):
                docs = chatbot.recherche_documents(question)
                reponse = chatbot.generer_reponse(question, docs)
                
                st.markdown(reponse)
                
                # Afficher les sources
                with st.expander("Sources"):
                    for doc in docs:
                        st.markdown(f"- [{doc['title']}]({doc['url']}) (Score: {doc['score']:.2f})")
        
        # Ajouter la réponse à l'historique
        st.session_state.messages.append({"role": "assistant", "content": reponse})

if __name__ == "__main__":
    main()
