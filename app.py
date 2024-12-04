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
            # Vérifier l'existence des fichiers
            if not os.path.exists('embeddings_db/faiss_index.idx'):
                raise FileNotFoundError("Index FAISS non trouvé")
            if not os.path.exists('embeddings_db/metadata.json'):
                raise FileNotFoundError("Fichier metadata non trouvé")
            
            # Charger l'index FAISS
            self.index = faiss.read_index('embeddings_db/faiss_index.idx')
            
            # Charger les metadata
            with open('embeddings_db/metadata.json', 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
            st.sidebar.success(f"Base de données chargée: {len(self.metadata)} documents")
            
        except Exception as e:
            st.error(f"Erreur de chargement des données: {str(e)}")
            raise e

    def get_query_embedding(self, question):
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=question
            )
            return np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)
        except Exception as e:
            st.error(f"Erreur dans la création de l'embedding: {str(e)}")
            raise e

    def recherche_documents_pertinents(self, question, k=3):
        try:
            # Obtenir l'embedding de la question
            question_embedding = self.get_query_embedding(question)
            
            # Rechercher les documents similaires
            D, I = self.index.search(question_embedding, k)
            
            # Récupérer les documents trouvés
            documents = []
            for i, idx in enumerate(I[0]):
                if idx >= 0 and idx < len(self.metadata):
                    doc = self.metadata[idx]
                    doc['score'] = float(D[0][i])  # Ajouter le score de similarité
                    documents.append(doc)
            
            # Afficher les documents trouvés dans la sidebar pour debug
            st.sidebar.write(f"Documents trouvés: {len(documents)}")
            for doc in documents:
                st.sidebar.markdown(f"- {doc['title']} (score: {doc['score']:.2f})")
            
            return documents
            
        except Exception as e:
            st.error(f"Erreur dans la recherche: {str(e)}")
            return []

    def generer_reponse(self, question, documents_pertinents):
        if not documents_pertinents:
            return "Je n'ai pas trouvé de documents pertinents dans la base de données pour répondre à votre question."
            
        # Préparer le contexte avec les URLs
        context = "\n\n".join([
            f"Titre: {doc.get('title', '')}\n"
            f"URL: {doc.get('url', '')}\n"
            f"Contenu: {doc.get('content', '')}"
            for doc in documents_pertinents
        ])
        
        system_prompt = """Tu es un assistant expert chargé d'aider les utilisateurs à naviguer sur notre site web.
        Instructions:
        1. Base tes réponses UNIQUEMENT sur les documents fournis dans le contexte
        2. Cite les URLs pertinentes dans ta réponse
        3. Si l'information n'est pas dans le contexte, indique-le clairement
        4. Formule des réponses naturelles et précises"""

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
            # Rechercher les documents pertinents
            docs = self.recherche_documents_pertinents(question)
            
            # Générer la réponse
            response = self.generer_reponse(question, docs)
            
            return response, docs
        except Exception as e:
            return f"Erreur: {str(e)}", []

# Configuration de la page
st.set_page_config(page_title="Assistant IA", page_icon="🤖", layout="wide")

# Style CSS pour améliorer l'affichage
st.markdown("""
<style>
    .source-info {
        font-size: 0.85em;
        color: #666;
        border-left: 3px solid #4CAF50;
        padding-left: 10px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des states
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
# Initialisation du chatbot
if 'chatbot' not in st.session_state:
    try:
        st.session_state.chatbot = ChatBot()
    except Exception as e:
        st.error("Impossible d'initialiser le chatbot. Vérifiez vos fichiers de données.")
        st.stop()

# Interface principale
st.title("💬 Assistant Site Web")
st.markdown("Je peux vous aider à trouver les informations sur notre site !")

# Zone de saisie
user_input = st.text_input("Votre question:")

# Traitement de la question
if user_input:
    try:
        response, docs = st.session_state.chatbot.answer_question(user_input)
        
        # Ajouter à l'historique
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input
        })
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "documents": docs
        })
        
    except Exception as e:
        st.error(f"Une erreur s'est produite: {str(e)}")

# Affichage de l'historique
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**Vous:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")
        if "documents" in msg and msg["documents"]:
            with st.expander("Sources utilisées"):
                for doc in msg["documents"]:
                    st.markdown(f"""
                    <div class="source-info">
                        <strong>{doc['title']}</strong><br>
                        <a href="{doc['url']}" target="_blank">{doc['url']}</a>
                    </div>
                    """, unsafe_allow_html=True)
    st.markdown("---")

# Bouton pour effacer l'historique
if st.button("Effacer l'historique"):
    st.session_state.messages = []
    st.rerun()
