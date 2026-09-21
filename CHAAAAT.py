import streamlit as st
import numpy as np
import re
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# Configuration générale (sans icône)
st.set_page_config(
    page_title="Charte ENSA Safi — Assistant Académique",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS professionnels
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* En-tête principal */
.app-header {
    padding: 1.2rem 1.6rem;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.04) 0%, rgba(15, 23, 42, 0.01) 100%);
    border: 1px solid rgba(148, 163, 184, 0.25);
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.app-header h1 {
    font-size: 1.45rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.app-header p {
    font-size: 0.85rem;
    margin: 0.25rem 0 0 0;
    color: #64748b;
}

/* Badges de statut */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    background-color: rgba(16, 185, 129, 0.1);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.25);
}
.status-pill.inactive {
    background-color: rgba(148, 163, 184, 0.1);
    color: #64748b;
    border: 1px solid rgba(148, 163, 184, 0.2);
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: currentColor;
}

/* Cartes de statistiques de la barre latérale */
.metric-box {
    background: rgba(148, 163, 184, 0.06);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 10px;
}
.metric-box .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    font-weight: 600;
}
.metric-box .val {
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: 2px;
}

/* Écran d'accueil / Empty State */
.welcome-card {
    padding: 2.2rem 2rem;
    border-radius: 14px;
    border: 1px dashed rgba(148, 163, 184, 0.35);
    background: rgba(248, 250, 252, 0.6);
    text-align: center;
    margin: 1.5rem auto 2rem auto;
    max-width: 680px;
}
.welcome-card h3 {
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.welcome-card p {
    font-size: 0.88rem;
    color: #64748b;
    line-height: 1.5;
    margin-bottom: 0;
}

/* Badges de citations et cartes de sources */
.source-tag {
    display: inline-flex;
    align-items: center;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
    background: rgba(59, 130, 246, 0.08);
    color: #2563eb;
    border: 1px solid rgba(59, 130, 246, 0.2);
    margin-right: 6px;
}
.source-item {
    border-left: 3px solid #3b82f6;
    background: rgba(148, 163, 184, 0.04);
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 10px;
    font-size: 0.85rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialisation du modèle d'embedding (mis en cache)
@st.cache_resource(show_spinner=False)
def charger_modele_embedding():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embedding_model = charger_modele_embedding()

# Fonctions de traitement de documents
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
    return f"""Tu es un assistant documentaire académique pour l'ENSA Safi.
RÈGLE 1 : Réponds de manière précise, structurée et formelle en te fondant exclusivement sur le contexte fourni.
RÈGLE 2 : Si l'information ne figure pas expressément dans le document, réponds uniquement : « Information non précisée dans la charte. »
RÈGLE 3 : Mentionne explicitement à la fin ou dans le texte les numéros de page des articles pertinents.

CONTEXTE :
{contexte}

QUESTION : {question}
RÉPONSE FORMULÉE :"""

# Initialisation de la session
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "prefilled_prompt" not in st.session_state:
    st.session_state["prefilled_prompt"] = None

# Panneau latéral de configuration
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("<p style='font-size:0.8rem; color:#64748b;'>Renseignez votre clé d'accès et gérez le document de référence.</p>", unsafe_allow_html=True)
    
    api_key = st.text_input("Clé API Groq", type="password", placeholder="gsk_...", help="Générez votre clé sur console.groq.com")
    
    modele_choisi = st.selectbox(
        "Moteur d'inférence",
        [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ],
        index=0,
        help="OpenAI GPT-OSS 20B offre une exécution rapide optimisée pour le RAG."
    )

    st.markdown("---")
    st.markdown("### Document de référence")
    fichier_charge = st.file_uploader("Fichier PDF de la charte", type=["pdf"], label_visibility="collapsed")
    top_k = st.slider("Passages analysés (k)", min_value=2, max_value=8, value=4)

    if "index" in st.session_state:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="label">Document actif</div>
                <div class="val" style="font-size:0.92rem;">{st.session_state.get('nom_fichier', 'Document')}</div>
            </div>
            <div style="display: flex; gap: 8px;">
                <div class="metric-box" style="flex: 1;">
                    <div class="label">Pages</div>
                    <div class="val">{st.session_state.get('total_pages', 0)}</div>
                </div>
                <div class="metric-box" style="flex: 1;">
                    <div class="label">Fragments</div>
                    <div class="val">{len(st.session_state.get('chunks', []))}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.button("Réinitialiser l'historique", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# Indexation automatique à l'import
if fichier_charge is not None:
    if "nom_fichier" not in st.session_state or st.session_state["nom_fichier"] != fichier_charge.name:
        with st.spinner("Indexation vectorielle du document..."):
            pages = extraire_texte_pdf(fichier_charge)
            index, chunks = indexer_document(pages)
            st.session_state["index"] = index
            st.session_state["chunks"] = chunks
            st.session_state["nom_fichier"] = fichier_charge.name
            st.session_state["total_pages"] = len(pages)
        st.rerun()

# Bannière supérieure
document_actif = "index" in st.session_state
status_html = """
<div class="status-pill"><span class="status-dot"></span> Prêt à l'analyse</div>
""" if document_actif else """
<div class="status-pill inactive"><span class="status-dot"></span> En attente du PDF</div>
"""

st.markdown(
    f"""
    <div class="app-header">
        <div>
            <h1>Assistant Réglementaire — ENSA Safi</h1>
            <p>Interrogation certifiée et sourcée sur la charte des études via RAG & Groq ({modele_choisi})</p>
        </div>
        <div>
            {status_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Vue par défaut si aucun échange n'a débuté
if not st.session_state["messages"]:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>Bienvenue sur votre assistant réglementaire</h3>
            <p>Cet outil analyse rigoureusement les articles de la charte de l'ENSA Safi pour vous apporter des réponses factuelles avec indication des numéros de page.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if document_actif:
        st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#64748b; margin-bottom:8px;'>SUGGESTIONS DE RECHERCHE :</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Conditions de réussite en cycle préparatoire", use_container_width=True):
                st.session_state["prefilled_prompt"] = "Quelles sont les conditions de réussite en cycle préparatoire ?"
                st.rerun()
        with col2:
            if st.button("Note éliminatoire et compensation", use_container_width=True):
                st.session_state["prefilled_prompt"] = "Quelle est la note éliminatoire en cycle préparatoire ?"
                st.rerun()
        with col3:
            if st.button("Règles d'assiduité et absences", use_container_width=True):
                st.session_state["prefilled_prompt"] = "Quelles sont les règles régissant l'assiduité et les absences injustifiées ?"
                st.rerun()
    else:
        st.info("Pour démarrer l'analyse, importez le fichier PDF dans le panneau latéral gauche.")

# Historique de discussion
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Consulter les extraits sources vérifiés"):
                for p in msg["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-item">
                            <div><span class="source-tag">Page {p['page']}</span> <span style="color:#64748b; font-size:0.75rem;">Score : {p['score']:.2f}</span></div>
                            <div style="margin-top:6px; color:#334155;">{p['texte'][:320]}...</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# Traitement de la question
prompt_input = st.chat_input("Posez votre question sur la réglementation...")
prompt = prompt_input or st.session_state.get("prefilled_prompt")

if prompt:
    if st.session_state.get("prefilled_prompt"):
        st.session_state["prefilled_prompt"] = None

    if not api_key:
        st.error("Veuillez renseigner votre clé API Groq dans le panneau latéral.")
    elif not document_actif:
        st.error("Veuillez charger le document PDF dans le panneau latéral avant d'interroger l'assistant.")
    else:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        passages = rechercher(prompt, st.session_state["index"], st.session_state["chunks"], k=top_k)
        prompt_final = construire_prompt(prompt, passages)

        with st.chat_message("assistant"):
            with st.spinner("Recherche et analyse documentaire..."):
                try:
                    client = Groq(api_key=api_key)
                    completion = client.chat.completions.create(
                        model=modele_choisi,
                        messages=[{"role": "user", "content": prompt_final}],
                        temperature=0.1
                    )
                    reponse = completion.choices[0].message.content
                    st.markdown(reponse)

                    with st.expander("Consulter les extraits sources vérifiés"):
                        for p in passages:
                            st.markdown(
                                f"""
                                <div class="source-item">
                                    <div><span class="source-tag">Page {p['page']}</span> <span style="color:#64748b; font-size:0.75rem;">Score : {p['score']:.2f}</span></div>
                                    <div style="margin-top:6px; color:#334155;">{p['texte'][:320]}...</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    nouveau_message = {
                        "role": "assistant",
                        "content": reponse,
                        "sources": passages
                    }
                    st.session_state["messages"].append(nouveau_message)
                except Exception as e:
                    st.error(f"Erreur d'inférence lors de la requête Groq : {e}")
