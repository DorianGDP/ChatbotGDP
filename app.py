# app.py
import streamlit as st
import faiss
import numpy as np
import json
from openai import OpenAI
import os

class ChatBot:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        
        # Charger l'index FAISS
        self.index = faiss.read_index('embeddings_db/faiss_index.idx')
        
        # Charger les metadata
        with open('embeddings_db/metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def get_query_embedding(self, question):
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=question
        )
        return np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)

    def recherche_documents_pertinents(self, question, k=3):
        question_embedding = self.get_query_embedding(question)
        D, I = self.index.search(question_embedding, k)
        return [self.metadata[idx] for idx in I[0]]

    def generer_reponse(self, question, documents_pertinents):
        context = "\n\n".join([
            f"Titre: {doc['title']}\nContenu: {doc['content']}\nURL: {doc['url']}" 
            for doc in documents_pertinents
        ])
        
        system_prompt = """Tu es un assistant virtuel expert chargé d'aider les utilisateurs à naviguer sur notre site web. 
        Tu dois:
        1. Fournir des réponses précises basées uniquement sur le contenu fourni
        2. Inclure systématiquement les URLs pertinentes dans ta réponse
        3. Indiquer clairement si tu ne trouves pas l'information dans le contexte
        4. Formuler des réponses naturelles et engageantes"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}\n\nContexte:\n{context}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Erreur lors de la génération de la réponse: {str(e)}"

    def repondre_question(self, question):
        docs_pertinents = self.recherche_documents_pertinents(question)
        reponse = self.generer_reponse(question, docs_pertinents)
        return reponse, docs_pertinents

# Configuration de la page Streamlit
st.set_page_config(page_title="Assistant Site Web", page_icon="🤖", layout="wide")

# Style CSS personnalisé
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e6f3ff;
        border-left: 5px solid #2196F3;
    }
    .bot-message {
        background-color: #f8f9fa;
        border-left: 5px solid #4CAF50;
    }
    .source-link {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de la session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Titre de l'application
st.title("💬 Assistant Site Web")
st.markdown("Posez vos questions sur notre site web !")

# Sidebar pour les paramètres
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("---")
    st.markdown("### À propos")
    st.markdown("Cet assistant utilise GPT-4 et la recherche sémantique pour répondre à vos questions.")

# Zone de chat principale
chat_container = st.container()

# Zone de saisie
question = st.text_input("Votre question:", key="question_input")

if question and api_key:
    # Initialiser le chatbot
    chatbot = ChatBot(api_key)
    
    # Obtenir la réponse
    reponse, sources = chatbot.repondre_question(question)
    
    # Ajouter les messages à l'historique
    st.session_state['messages'].append({"role": "user", "content": question})
    st.session_state['messages'].append({"role": "assistant", "content": reponse, "sources": sources})

# Afficher l'historique des messages
with chat_container:
    for msg in st.session_state['messages']:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div><strong>Vous:</strong> {msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <div><strong>Assistant:</strong> {msg["content"]}</div>
                <div class="source-link">
                    <strong>Sources consultées:</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Afficher les sources dans un expander
            with st.expander("Voir les sources"):
                for source in msg.get("sources", []):
                    st.markdown(f"- [{source['title']}]({source['url']})")

# Bouton pour effacer l'historique
if st.button("Effacer l'historique"):
    st.session_state['messages'] = []
    st.experimental_rerun()
