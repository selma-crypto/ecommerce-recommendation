import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Agent IA de recommandation e-commerce", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("electronics_filtered.csv")
    df = df[["user_id", "product_id", "rating"]].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna()
    return df


@st.cache_data
def build_matrices(df):
    user_item_matrix = df.pivot_table(
        index="user_id",
        columns="product_id",
        values="rating",
        fill_value=0
    )
    user_similarity = cosine_similarity(user_item_matrix)
    product_stats = (
        df.groupby("product_id")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
        .sort_values(["rating_count", "avg_rating"], ascending=[False, False])
    )
    return user_item_matrix, user_similarity, product_stats


def recommend_products(user_id, user_item_matrix, user_similarity, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    user_idx = user_item_matrix.index.get_loc(user_id)
    sim_scores = list(enumerate(user_similarity[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    similar_users = [i for i, _ in sim_scores[1:11]]

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
    return [product_id for product_id, _ in recommended[:top_n]]


def collaborative_recommendations(user_id, user_item_matrix, user_similarity, product_stats, top_n):
    ids = recommend_products(user_id, user_item_matrix, user_similarity, top_n)
    if not ids:
        return pd.DataFrame(columns=["product_id", "avg_rating", "rating_count"])

    result = product_stats[product_stats["product_id"].isin(ids)].copy()
    result = result.sort_values(["rating_count", "avg_rating"], ascending=[False, False]).head(top_n)
    return result


def agent_filter(query, product_stats, fallback_df, top_n):
    if not query:
        return fallback_df, "Recommandation collaborative basée sur des utilisateurs similaires."

    q = query.lower().strip()

    if "premium" in q or "haut de gamme" in q:
        result = product_stats[
            (product_stats["avg_rating"] >= 4.2) &
            (product_stats["rating_count"] >= 10)
        ].head(top_n)
        if result.empty:
            return fallback_df, "Aucun produit premium exact trouvé. Retour au moteur collaboratif."
        return result, "Filtre agent IA : produits premium."

    if "populaire" in q or "tendance" in q:
        result = product_stats.sort_values(["rating_count", "avg_rating"], ascending=[False, False]).head(top_n)
        return result, "Filtre agent IA : produits populaires."

    if "bien noté" in q or "meilleur" in q:
        result = product_stats.sort_values(["avg_rating", "rating_count"], ascending=[False, False]).head(top_n)
        return result, "Filtre agent IA : produits les mieux notés."

    if "fiable" in q:
        result = product_stats[
            (product_stats["avg_rating"] >= 4.0) &
            (product_stats["rating_count"] >= 20)
        ].sort_values(["avg_rating", "rating_count"], ascending=[False, False]).head(top_n)
        if result.empty:
            return fallback_df, "Aucun produit fiable exact trouvé. Retour au moteur collaboratif."
        return result, "Filtre agent IA : produits fiables."

    return fallback_df, "Aucune règle métier détectée. Retour au moteur collaboratif."


def main():
    st.title("Agent IA de recommandation e-commerce")
    st.write("Version simplifiée et stable pour démonstration.")

    df = load_data()
    user_item_matrix, user_similarity, product_stats = build_matrices(df)

    st.sidebar.header("Paramètres")
    user_id = st.sidebar.selectbox("Choisissez un utilisateur", user_item_matrix.index.tolist())
    top_n = st.sidebar.slider("Nombre de recommandations", min_value=3, max_value=10, value=5)

    query = st.text_input(
        "Que recherchez-vous ?",
        placeholder="Exemples : produit premium, produit populaire, produit bien noté, produit fiable"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Profil utilisateur")
        seen = user_item_matrix.loc[user_id]
        seen = seen[seen > 0].sort_values(ascending=False).reset_index()
        seen.columns = ["product_id", "rating"]
        st.dataframe(seen.head(10))

    with col2:
        st.subheader("Produits recommandés")
        if st.button("Générer les recommandations"):
            base_recos = collaborative_recommendations(
                user_id, user_item_matrix, user_similarity, product_stats, top_n
            )
            final_recos, explanation = agent_filter(query, product_stats, base_recos, top_n)

            if final_recos.empty:
                st.warning("Aucune recommandation disponible.")
            else:
                st.dataframe(final_recos)
                st.write(explanation)

    st.markdown("---")
    st.subheader("Informations projet")
    st.write(
        "Ce MVP combine un moteur de recommandation collaborative et une logique agent simple "
        "basée sur des règles métier en langage naturel."
    )


try:
    main()
except Exception as e:
    st.error("Erreur capturée dans l'application :")
    st.exception(e)
