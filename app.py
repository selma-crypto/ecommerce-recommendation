import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="E-commerce Product Recommender", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("electronics_filtered.csv")
    return df

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
    recommendations = recommend_products(user_id, user_item_matrix, user_similarity, top_n=top_n)

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

st.title("AI Product Recommender")
st.markdown("Moteur de recommandation e-commerce basé sur la similarité entre utilisateurs.")

df = load_data()
user_item_matrix = build_user_item_matrix(df)
user_similarity = compute_similarity(user_item_matrix)

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

col1, col2 = st.columns(2)

with col1:
    st.subheader("Profil utilisateur")
    seen_products = get_seen_products(user_id, user_item_matrix)
    st.write("Produits déjà notés par cet utilisateur :")
    st.dataframe(seen_products.head(10), use_container_width=True)

with col2:
    st.subheader("Produits recommandés")

    if st.button("Générer les recommandations"):
        recommendations = recommend_products_with_stats(
            user_id,
            user_item_matrix,
            user_similarity,
            df,
            top_n=top_n
        )

        if recommendations.empty:
            st.warning("Aucune recommandation disponible pour cet utilisateur.")
        else:
            st.dataframe(recommendations, use_container_width=True)

            st.markdown("### Pourquoi ces recommandations ?")
            st.write(
                "Ces produits sont recommandés car des utilisateurs ayant un profil similaire "
                "les ont bien notés."
            )