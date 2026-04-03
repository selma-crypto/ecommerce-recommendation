import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Agent IA de recommandation e-commerce", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("electronics_filtered.csv")


def build_user_item_matrix(df):
    return df.pivot_table(
        index="user_id",
        columns="product_id",
        values="rating"
    ).fillna(0)


def compute_similarity(user_item_matrix):
    return cosine_similarity(user_item_matrix)


def get_product_stats(df):
    stats = df.groupby("product_id").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()
    return stats


def recommend_products(user_id, user_item_matrix, user_similarity, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    user_idx = user_item_matrix.index.get_loc(user_id)
    sim_scores = list(enumerate(user_similarity[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    similar_users = [i for i, score in sim_scores[1:11]]

    seen_products = set(
        user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    )

    product_scores = {}

    for sim_user_idx in similar_users:
        sim_user_id = user_item_matrix.index[sim_user_idx]
        sim_user_ratings = user_item_matrix.loc[sim_user_id]

        for product_id, rating in sim_user_ratings.items():
            if rating > 0 and product_id not in seen_products:
                product_scores[product_id] = product_scores.get(product_id, 0) + rating

    recommended = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
    return [product_id for product_id, score in recommended[:top_n]]


def recommend_products_with_stats(user_id, user_item_matrix, user_similarity, df, top_n=5):
    recommendations = recommend_products(
        user_id,
        user_item_matrix,
        user_similarity,
        top_n=top_n
    )

    if not recommendations:
        return pd.DataFrame()

    product_stats = get_product_stats(df)
    result = product_stats[product_stats["product_id"].isin(recommendations)].copy()
    result = result.sort_values(by=["rating_count", "avg_rating"], ascending=[False, False])
    return result


def get_seen_products(user_id, user_item_matrix):
    seen = user_item_matrix.loc[user_id]
    seen = seen[seen > 0].sort_values(ascending=False).reset_index()
    seen.columns = ["product_id", "rating"]
    return seen


def run_agent_query(user_query, product_stats, collaborative_recommendations, top_n=5):
    """
    Interprète une requête simple en langage naturel et retourne :
    - un DataFrame de recommandations
    - une explication
    - le type de logique utilisée
    """
    if not user_query:
        return collaborative_recommendations, (
            "Aucune préférence texte n'a été fournie. "
            "L'agent utilise donc la recommandation collaborative basée sur des utilisateurs similaires."
        ), "collaborative"

    query = user_query.lower().strip()

    # Règles simples pour donner une dimension 'agent'
    if "bien noté" in query or "bien note" in query or "meilleur" in query or "haute note" in query:
        result = product_stats.sort_values(by=["avg_rating", "rating_count"], ascending=[False, False]).head(top_n)
        explanation = (
            "L'agent a interprété votre demande comme une recherche de produits les mieux évalués. "
            "Il a donc priorisé la note moyenne, puis le volume d'avis."
        )
        return result, explanation, "rule_high_rating"

    if "populaire" in query or "connu" in query or "tendance" in query:
        result = product_stats.sort_values(by=["rating_count", "avg_rating"], ascending=[False, False]).head(top_n)
        explanation = (
            "L'agent a interprété votre demande comme une recherche de produits populaires. "
            "Il a donc priorisé le nombre d'évaluations, puis la note moyenne."
        )
        return result, explanation, "rule_popular"

    if "fiable" in query:
        result = product_stats[
            (product_stats["avg_rating"] >= 4.0) & (product_stats["rating_count"] >= 20)
        ].sort_values(by=["avg_rating", "rating_count"], ascending=[False, False]).head(top_n)
        explanation = (
            "L'agent a interprété votre demande comme une recherche de produits fiables. "
            "Il a donc retenu des produits bien notés avec un volume d'avis suffisant."
        )
        return result, explanation, "rule_reliable"

    if "premium" in query or "haut de gamme" in query:
        result = product_stats[
            (product_stats["avg_rating"] >= 4.2) & (product_stats["rating_count"] >= 10)
        ].sort_values(by=["avg_rating", "rating_count"], ascending=[False, False]).head(top_n)
        explanation = (
            "L'agent a interprété votre demande comme une recherche de produits premium. "
            "Il a donc retenu des produits très bien notés avec une certaine crédibilité d'usage."
        )
        return result, explanation, "rule_premium"

    # Fallback : collaborative filtering classique
    explanation = (
        "La demande n'entrait pas dans une règle métier spécifique. "
        "L'agent a donc utilisé le moteur de recommandation basé sur la similarité entre utilisateurs."
    )
    return collaborative_recommendations, explanation, "collaborative"


# Données et objets principaux
df = load_data()
user_item_matrix = build_user_item_matrix(df)
user_similarity = compute_similarity(user_item_matrix)
product_stats = get_product_stats(df)

# Mémoire légère de session
if "history" not in st.session_state:
    st.session_state.history = []

# Titre
st.title("Agent IA de recommandation e-commerce")
st.markdown(
    "Assistant intelligent capable d'analyser un profil utilisateur, d'interpréter une demande simple "
    "et de recommander des produits pertinents."
)

# Sidebar
st.sidebar.header("Paramètres")
user_id = st.sidebar.selectbox(
    "Choisissez un utilisateur",
    user_item_matrix.index.tolist()
)

top_n = st.sidebar.slider(
    "Nombre de recommandations",
    min_value=3,
    max_value=10,
    value=5
)

# Entrée de type agent
st.markdown("### Demande utilisateur")
user_query = st.text_input(
    "Que recherchez-vous ?",
    placeholder="Exemples : produit bien noté, produit populaire, produit fiable, produit premium"
)

if user_query:
    st.session_state.history.append(user_query)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Profil utilisateur")
    seen_products = get_seen_products(user_id, user_item_matrix)
    st.write("Produits déjà notés par cet utilisateur :")
    st.dataframe(seen_products.head(10), use_container_width=True)

    if st.session_state.history:
        st.markdown("#### Historique des demandes")
        history_df = pd.DataFrame({"demandes": st.session_state.history[-5:]})
        st.dataframe(history_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Produits recommandés")

    if st.button("Générer les recommandations"):
        collaborative_recommendations = recommend_products_with_stats(
            user_id,
            user_item_matrix,
            user_similarity,
            df,
            top_n=top_n
        )

        recommendations, explanation, mode = run_agent_query(
            user_query=user_query,
            product_stats=product_stats,
            collaborative_recommendations=collaborative_recommendations,
            top_n=top_n
        )

        if recommendations.empty:
            st.warning("Aucune recommandation disponible pour cet utilisateur ou cette demande.")
        else:
            st.dataframe(recommendations.head(top_n), use_container_width=True)

            st.markdown("#### Pourquoi ces recommandations ?")
            st.write(explanation)

            if mode == "collaborative":
                st.info("Mode utilisé : recommandation collaborative basée sur des utilisateurs similaires.")
            else:
                st.info("Mode utilisé : agent IA piloté par règles métier + données produits.")

st.markdown("---")
st.subheader("Informations projet")
st.write(
    "Ce MVP combine deux briques : un moteur de recommandation collaborative et une couche d'agent IA "
    "capable d'interpréter une demande simple en langage naturel."
)
st.write(
    "Le moteur principal repose sur la similarité entre utilisateurs. "
    "L'agent ajoute une logique de décision et d'explication pour rendre le système plus compréhensible et plus interactif."
)
st.write(
    "Limites actuelles : le dataset ne contient pas les noms produits, donc le MVP affiche des IDs produits. "
    "Une version future pourrait intégrer un LLM, des métadonnées produits et un système hybride plus avancé."
)
