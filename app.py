import streamlit as st
import faiss
import numpy as np
import json
from openai import OpenAI
import os

class ChatBot:
    def __init__(self):
        # Récupérer la clé API depuis les secrets de Streamlit
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        try:
            # Charger l'index FAISS
            self.index = faiss.read_index('embeddings_db/faiss_index.idx')
            
            # Charger les metadata
            with open('embeddings_db/metadata.json', 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        except Exception as e:
            st.error(f"Erreur de chargement des données: {str(e)}")
            
    def get_query_embedding(self, question):
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=question
        )
        return np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)

    def recherche_documents_pertinents(self, question, k=3):
        try:
            question_embedding = self.get_query_embedding(question)
            D, I = self.index.search(question_embedding, k)
            
            valid_docs = []
            for idx in I[0]:
                if idx >= 0 and idx < len(self.metadata):
                    valid_docs.append(self.metadata[idx])
            return valid_docs
        except Exception as e:
            st.error(f"Erreur de recherche: {str(e)}")
            return []

    def generer_reponse(self, question, documents_pertinents):
        if not documents_pertinents:
            return "Je n'ai pas trouvé de documents pertinents pour répondre à votre question."
            
        context = "\n\n".join([
            f"Titre: {doc.get('title', '')}\nContenu: {doc.get('content', '')}" 
            for doc in documents_pertinents
        ])
        
        system_prompt = """Tu es un assistant virtuel expert. 
        Réponds de manière précise et naturelle en te basant uniquement sur le contexte fourni.
        Si tu ne trouves pas l'information dans le contexte, indique-le clairement."""

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
            return f"Désolé, une erreur s'est produite: {str(e)}"

    def answer_question(self, question):
        """
        Méthode principale pour répondre aux questions
        """
        try:
            docs = self.recherche_documents_pertinents(question)
            response = self.generer_reponse(question, docs)
            return response
        except Exception as e:
            return f"Erreur: {str(e)}"

# Configuration de la page
st.set_page_config(page_title="Assistant IA", page_icon="🤖")

# Initialisation de la session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
# Initialisation du chatbot (une seule fois)
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = ChatBot()

# Interface utilisateur
st.title("💬 Assistant IA")
st.markdown("Posez vos questions, je suis là pour vous aider !")

# Zone de saisie
user_input = st.text_input("Votre question:")

# Traitement de la question
if user_input:
    try:
        response = st.session_state.chatbot.answer_question(user_input)
        
        # Ajouter à l'historique
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    except Exception as e:
        st.error(f"Une erreur s'est produite: {str(e)}")

# Affichage de l'historique
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**Vous:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")
    st.markdown("---")

# Bouton pour effacer l'historique
if st.button("Effacer l'historique"):
    st.session_state.messages = []
    st.rerun()
