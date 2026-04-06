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
    recommendations = recommend_products(user_id, user_item_matrix, user_similarity, top_n)

    if not recommendations:
        return pd.DataFrame()

    product_stats = get_product_stats(df)
    result = product_stats[product_stats["product_id"].isin(recommendations)].copy()
    result = result.sort_values(by=["rating_count", "avg_rating"], ascending=[False, False])

    return result

def run_agent_query(user_query, product_stats, collaborative_recommendations, top_n=5):

    if not user_query:
        return collaborative_recommendations, "Pas de requête utilisateur.", "collaborative"

    query = user_query.lower().strip()

    if "bien noté" in query or "meilleur" in query:
        result = product_stats.sort_values(by=["avg_rating", "rating_count"], ascending=[False, False]).head(top_n)
        return result, "Produits les mieux notés", "rule_high_rating"

    if "populaire" in query:
        result = product_stats.sort_values(by=["rating_count", "avg_rating"], ascending=[False, False]).head(top_n)
        return result, "Produits populaires", "rule_popular"

    if "fiable" in query:
        result = product_stats[
            (product_stats["avg_rating"] >= 4.0) &
            (product_stats["rating_count"] >= 20)
        ].sort_values(by=["avg_rating", "rating_count"], ascending=[False, False]).head(top_n)
        return result, "Produits fiables", "rule_reliable"

    if "premium" in query:
        result = product_stats[
            (product_stats["avg_rating"] >= 4.2) &
            (product_stats["rating_count"] >= 10)
        ].sort_values(by=["avg_rating", "rating_count"], ascending=[False, False]).head(top_n)
        return result, "Produits premium", "rule_premium"

    return collaborative_recommendations, "Recommandation collaborative", "collaborative"


# --- DATA ---
df = load_data()
user_item_matrix = build_user_item_matrix(df)
user_similarity = compute_similarity(user_item_matrix)
product_stats = get_product_stats(df)

# --- SESSION ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- UI ---
st.title("Agent IA de recommandation e-commerce")

st.sidebar.header("Paramètres")
user_id = st.sidebar.selectbox("Choisissez un utilisateur", user_item_matrix.index.tolist())
top_n = st.sidebar.slider("Nombre de recommandations", 1, 10, 5)

st.markdown("### Demande utilisateur")
user_query = st.text_input("Que recherchez-vous ?")

if user_query and (not st.session_state.history or st.session_state.history[-1] != user_query):
    st.session_state.history.append(user_query)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Profil utilisateur")
    seen_products = user_item_matrix.loc[user_id]
    st.dataframe(seen_products.head(10), use_container_width=True)

with col2:
    st.subheader("Produits recommandés")

    if st.button("Générer les recommandations"):
        try:
            collaborative_recommendations = recommend_products_with_stats(
                user_id,
                user_item_matrix,
                user_similarity,
                df,
                top_n
            )

            recommendations, explanation, mode = run_agent_query(
                user_query,
                product_stats,
                collaborative_recommendations,
                top_n
            )

            st.write("Type :", type(recommendations))

            if recommendations.empty:
                st.warning("Aucune recommandation.")
            else:
                st.dataframe(recommendations.head(top_n), use_container_width=True)
                st.write(explanation)

        except Exception as e:
            st.error("Erreur capturée :")
            st.exception(e)
            