import streamlit as st
import faiss
import numpy as np
import json
from openai import OpenAI
import os

class ChatBot:
    def __init__(self):
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        try:
            if not os.path.exists('embeddings_db/faiss_index.idx'):
                raise FileNotFoundError("Index FAISS non trouvé")
            if not os.path.exists('embeddings_db/metadata.json'):
                raise FileNotFoundError("Fichier metadata non trouvé")
            
            self.index = faiss.read_index('embeddings_db/faiss_index.idx')
            
            with open('embeddings_db/metadata.json', 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
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
            question_embedding = self.get_query_embedding(question)
            D, I = self.index.search(question_embedding, k)
            
            documents = []
            for idx in I[0]:
                if idx >= 0 and idx < len(self.metadata):
                    documents.append(self.metadata[idx])
            return documents
            
        except Exception as e:
            st.error(f"Erreur dans la recherche: {str(e)}")
            return []

    def generer_reponse(self, question, documents_pertinents):
        if not documents_pertinents:
            return "Je ne trouve pas d'informations pertinentes pour répondre à votre question. Pourriez-vous la reformuler différemment ?"
            
        context = "\n\n".join([
            f"Titre: {doc.get('title', '')}\n"
            f"URL: {doc.get('url', '')}\n"
            f"Contenu: {doc.get('content', '')}"
            for doc in documents_pertinents
        ])
        
        system_prompt = """Tu es Claude, un assistant expert et professionnel. 
        Instructions:
        1. Réponds de manière naturelle et engageante, comme dans une vraie conversation
        2. Structure tes réponses avec des paragraphes clairs et aérés
        3. Base-toi uniquement sur le contexte fourni
        4. Si pertinent, mentionne les sources disponibles en les intégrant naturellement
        5. Si l'information n'est pas dans le contexte, propose de reformuler la question"""

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
            return "Je rencontre actuellement des difficultés techniques. Pourriez-vous réessayer dans quelques instants ?"

    def answer_question(self, question):
        try:
            docs = self.recherche_documents_pertinents(question)
            response = self.generer_reponse(question, docs)
            return response, docs
        except Exception as e:
            return "Je suis désolé, je ne peux pas traiter votre demande pour le moment.", []

# Configuration de la page
st.set_page_config(
    page_title="Assistant IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style CSS amélioré
st.markdown("""
<style>
    .chat-container {
        margin-bottom: 20px;
    }
    .message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        line-height: 1.5;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #ffffff;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .sources-section {
        font-size: 0.9em;
        margin-top: 10px;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    .source-link {
        color: #1976D2;
        text-decoration: none;
        margin-right: 10px;
    }
    .source-link:hover {
        text-decoration: underline;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des states
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'chatbot' not in st.session_state:
    try:
        st.session_state.chatbot = ChatBot()
    except Exception as e:
        st.error("Impossible d'initialiser le système. Veuillez réessayer ultérieurement.")
        st.stop()

# Interface principale
st.title("💬 Assistant virtuel")
st.markdown("Bonjour ! Je suis là pour vous aider à trouver les informations dont vous avez besoin. Que puis-je faire pour vous ?")

# Zone de saisie
user_input = st.text_input("", placeholder="Posez votre question ici...")

# Traitement de la question
if user_input:
    try:
        response, docs = st.session_state.chatbot.answer_question(user_input)
        
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
        st.error("Je rencontre des difficultés techniques. Merci de réessayer.")

# Affichage de l'historique
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="message user-message">
            <strong>Vous :</strong><br>
            {msg['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = ""
        if "documents" in msg and msg["documents"]:
            sources_html = """
            <div class="sources-section">
                <strong>📚 Sources consultées :</strong><br>
                """ + " | ".join([f"""<a href="{doc['url']}" target="_blank" class="source-link">{doc['title']}</a>""" 
                                 for doc in msg["documents"]]) + """
            </div>
            """
            
        st.markdown(f"""
        <div class="message assistant-message">
            <strong>Assistant :</strong><br>
            {msg['content']}
            {sources_html}
        </div>
        """, unsafe_allow_html=True)

# Footer avec bouton de réinitialisation
st.markdown("---")
cols = st.columns([3, 1, 3])
with cols[1]:
    if st.button("🔄 Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()
