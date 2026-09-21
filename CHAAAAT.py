import streamlit as st
import numpy as np
import re
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# Configuration de la page
st.set_page_config(page_title="Assistant RAG - ENSA Safi (Groq / GPT-OSS 20B)", layout="wide")

# --- Initialisation du modèle d'embedding (mis en cache) ---
@st.cache_resource
def charger_modele_embedding():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embedding_model = charger_modele_embedding()

# --- Fonctions de traitement du document ---
def extraire_texte_pdf(fichier_pdf):
    reader = PdfReader(fichier_pdf)
    pages = []
    for numero, page in enumerate(reader.pages, start=1):
        texte = page.extract_text() or ""
        texte = re.sub(r"\s+", " ", texte).strip()
        if texte:
            pages.append({"page": numero, "texte": texte})
    return pages

def decouper_texte(texte, taille=900, chevauchement=150):
    morceaux = []
    debut = 0
    while debut < len(texte):
        fin = min(debut + taille, len(texte))
        morceau = texte[debut:fin].strip()
        if morceau:
            morceaux.append(morceau)
        if fin == len(texte):
            break
        debut = debut + taille - chevauchement
    return morceaux

def indexer_document(pages):
    chunks = []
    for page in pages:
        for idx, texte_chunk in enumerate(decouper_texte(page["texte"]), start=1):
            chunks.append({"page": page["page"], "chunk": idx, "texte": texte_chunk})
    
    textes = [c["texte"] for c in chunks]
    vecteurs = embedding_model.encode(textes, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    
    dimension = vecteurs.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vecteurs)
    return index, chunks

def rechercher(query, index, chunks, k=4):
    query_vec = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, k)
    resultats = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            item = chunks[int(idx)].copy()
            item["score"] = float(score)
            resultats.append(item)
    return resultats

def construire_prompt(question, passages):
    contexte = "\n\n".join(
        f'[Source : page {p["page"]}]\n{p["texte"]}' for p in passages
    )
    return f"""Tu es un assistant documentaire pour l'ENSA Safi.
RÈGLE 1 : Réponds en te fondant exclusivement sur les extraits fournis ci-dessous.
RÈGLE 2 : N'invente aucune information. Si l'information n'est pas présente dans les extraits, réponds exactement : « Information non précisée dans la charte. »
RÈGLE 3 : Indique systématiquement les numéros de page des sources utilisées.

CONTEXTE :
{contexte}

QUESTION : {question}
RÉPONSE :"""

# --- Interface utilisateur ---
st.title("📚 Assistant RAG — Charte ENSA Safi")
st.caption("Modèle : openai/gpt-oss-20b via Groq")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Clé API Groq", type="password", help="Obtenez une clé sur console.groq.com")
    modele_choisi = st.selectbox(
        "Modèle Groq",
        [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ],
        index=0
    )
    fichier_charge = st.file_uploader("Charger le document PDF", type=["pdf"])
    top_k = st.slider("Nombre de passages à récupérer (top-k)", min_value=1, max_value=8, value=4)

    if st.button("Réinitialiser la conversation"):
        st.session_state["messages"] = []
        st.rerun()

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Indexation du fichier PDF
if fichier_charge is not None:
    if "nom_fichier" not in st.session_state or st.session_state["nom_fichier"] != fichier_charge.name:
        with st.spinner("Extraction et indexation vectorielle en cours..."):
            pages = extraire_texte_pdf(fichier_charge)
            index, chunks = indexer_document(pages)
            st.session_state["index"] = index
            st.session_state["chunks"] = chunks
            st.session_state["nom_fichier"] = fichier_charge.name
            st.session_state["total_pages"] = len(pages)
        st.sidebar.success(f"{st.session_state['total_pages']} pages indexées ({len(chunks)} fragments).")
else:
    st.info("Veuillez charger un fichier PDF dans la barre latérale pour démarrer.")

# Affichage des messages passés
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources consultées"):
                for src in msg["sources"]:
                    st.markdown(f"**Page {src['page']}** (Score: `{src['score']:.3f}`)")
                    st.caption(src["texte"][:300] + "...")

# Saisie de la question
if prompt := st.chat_input("Posez votre question sur le document..."):
    if not api_key:
        st.error("Veuillez renseigner votre clé API Groq dans la barre latérale.")
    elif "index" not in st.session_state:
        st.error("Veuillez charger et indexer un fichier PDF avant de poser une question.")
    else:
        # Message utilisateur
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Recherche de fragments pertinents
        passages = rechercher(prompt, st.session_state["index"], st.session_state["chunks"], k=top_k)
        prompt_augmente = construire_prompt(prompt, passages)

        # Appel LLM via Groq
        with st.chat_message("assistant"):
            with st.spinner(f"Génération avec {modele_choisi}..."):
                try:
                    client = Groq(api_key=api_key)
                    completion = client.chat.completions.create(
                        model=modele_choisi,
                        messages=[
                            {"role": "user", "content": prompt_augmente}
                        ],
                        temperature=0.1
                    )
                    texte_reponse = completion.choices[0].message.content
                    st.markdown(texte_reponse)

                    with st.expander("Sources consultées"):
                        for p in passages:
                            st.markdown(f"**Page {p['page']}** (Score: `{p['score']:.3f}`)")
                            st.caption(p["texte"][:300] + "...")

                    nouveau_message = {
                        "role": "assistant",
                        "content": texte_reponse,
                        "sources": passages
                    }
                    st.session_state["messages"].append(nouveau_message)
                except Exception as e:
                    st.error(f"Erreur lors de l'appel à l'API Groq : {e}")
