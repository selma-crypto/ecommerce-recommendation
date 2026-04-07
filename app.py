import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Agent IA de recommandation e-commerce", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("electronics_filtered.csv")
    df.columns = [c.strip() for c in df.columns]

    required = {"user_id", "product_id", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes obligatoires manquantes : {missing}")

    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["user_id", "product_id", "rating"])

    if "product_title" not in df.columns:
        if "product_name" in df.columns:
            df["product_title"] = df["product_name"].astype(str)
        elif "title" in df.columns:
            df["product_title"] = df["title"].astype(str)
        else:
            df["product_title"] = "Produit " + df["product_id"].astype(str)

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price"] = pd.NA

    return df


@st.cache_data
def build_artifacts(df):
    user_item_matrix = df.pivot_table(
        index="user_id",
        columns="product_id",
        values="rating",
        fill_value=0
    )

    user_similarity = cosine_similarity(user_item_matrix)

    product_catalog = (
        df.groupby("product_id", as_index=False)
        .agg(
            product_title=("product_title", "first"),
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count"),
            price=("price", "first"),
        )
        .sort_values(["rating_count", "avg_rating"], ascending=[False, False])
    )

    return user_item_matrix, user_similarity, product_catalog


def recommend_products(user_id, user_item_matrix, user_similarity, top_n=6):
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


def collaborative_recommendations(user_id, user_item_matrix, user_similarity, product_catalog, top_n):
    ids = recommend_products(user_id, user_item_matrix, user_similarity, top_n)

    if not ids:
        return pd.DataFrame(columns=product_catalog.columns)

    result = product_catalog[product_catalog["product_id"].isin(ids)].copy()
    result = result.sort_values(["rating_count", "avg_rating"], ascending=[False, False]).head(top_n)
    return result


def agent_filter(query, product_catalog, fallback_df, top_n):
    if not query:
        return fallback_df, "Recommandation collaborative basée sur des utilisateurs similaires."

    q = query.lower().strip()

    if "premium" in q or "haut de gamme" in q:
        result = product_catalog[
            (product_catalog["avg_rating"] >= 4.2) &
            (product_catalog["rating_count"] >= 10)
        ].sort_values(["avg_rating", "rating_count"], ascending=[False, False]).head(top_n)
        if result.empty:
            return fallback_df, "Aucun produit premium exact trouvé. Retour au moteur collaboratif."
        return result, "Filtre agent IA : produits premium."

    if "populaire" in q or "tendance" in q:
        result = product_catalog.sort_values(["rating_count", "avg_rating"], ascending=[False, False]).head(top_n)
        return result, "Filtre agent IA : produits populaires."

    if "bien noté" in q or "mieux noté" in q or "meilleur" in q:
        result = product_catalog.sort_values(["avg_rating", "rating_count"], ascending=[False, False]).head(top_n)
        return result, "Filtre agent IA : produits les mieux notés."

    if "fiable" in q:
        result = product_catalog[
            (product_catalog["avg_rating"] >= 4.0) &
            (product_catalog["rating_count"] >= 20)
        ].sort_values(["avg_rating", "rating_count"], ascending=[False, False]).head(top_n)
        if result.empty:
            return fallback_df, "Aucun produit fiable exact trouvé. Retour au moteur collaboratif."
        return result, "Filtre agent IA : produits fiables."

    if "budget" in q or "pas cher" in q or "moins cher" in q:
        priced = product_catalog.dropna(subset=["price"]).sort_values(["price", "avg_rating"], ascending=[True, False]).head(top_n)
        if priced.empty:
            return fallback_df, "Aucun prix disponible dans le catalogue. Retour au moteur collaboratif."
        return priced, "Filtre agent IA : produits budget."

    return fallback_df, "Aucune règle métier détectée. Retour au moteur collaboratif."


def price_to_text(price):
    if pd.isna(price):
        return "Prix non disponible"
    return f"{float(price):,.2f} €".replace(",", " ").replace(".", ",")


def stars(avg_rating):
    if pd.isna(avg_rating):
        return "Pas de note"
    full = int(round(float(avg_rating)))
    full = max(0, min(full, 5))
    return "★" * full + "☆" * (5 - full)


def render_product_card(row):
    title = str(row.get("product_title", "Produit"))
    price_text = price_to_text(row.get("price", pd.NA))
    avg_rating = row.get("avg_rating", None)
    rating_count = row.get("rating_count", None)
    product_id = str(row.get("product_id", ""))

    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.write(price_text)

        if pd.notna(avg_rating):
            st.caption(f"{stars(avg_rating)} {float(avg_rating):.2f}/5 - {int(rating_count)} avis")
        else:
            st.caption("Pas de note disponible")

        st.caption(f"Référence : {product_id}")


def main():
    st.title("Agent IA de recommandation e-commerce")
    st.caption("Version finale propre pour soutenance.")
    st.caption("Recommandations personnalisées à partir du profil utilisateur connecté.")

    df = load_data()
    user_item_matrix, user_similarity, product_catalog = build_artifacts(df)

    user_id = user_item_matrix.index[0]
    top_n = 6

    query = st.text_input(
        "Quel type de produit recherchez-vous ?",
        placeholder="Exemples : produit premium, produit populaire, produit bien noté, produit fiable, produit budget"
    )

    st.markdown("### Profil utilisateur")
    seen = user_item_matrix.loc[user_id]
    seen = seen[seen > 0].sort_values(ascending=False).reset_index()
    seen.columns = ["product_id", "rating"]

    seen_preview = seen.head(8).merge(
        product_catalog[["product_id", "product_title"]],
        on="product_id",
        how="left"
    )[["product_title", "product_id", "rating"]]

    st.dataframe(seen_preview, use_container_width=True)

    if st.button("Générer les recommandations", use_container_width=True):
        base_recos = collaborative_recommendations(
            user_id, user_item_matrix, user_similarity, product_catalog, top_n
        )
        final_recos, explanation = agent_filter(query, product_catalog, base_recos, top_n)

        st.markdown("### Produits recommandés")
        st.info(explanation)

        if final_recos.empty:
            st.warning("Aucune recommandation disponible.")
        else:
            cols = st.columns(3)
            rows = final_recos.reset_index(drop=True).to_dict(orient="records")
            for i, product in enumerate(rows):
                with cols[i % 3]:
                    render_product_card(product)

    st.markdown("---")
    st.subheader("Informations projet")
    st.write(
        "Cette version finale met l'accent sur une présentation propre et lisible pour la soutenance."
    )
    st.write(
        "Le moteur combine une recommandation collaborative entre utilisateurs et une couche agent IA "
        "basée sur des règles métier simples en langage naturel."
    )
    st.write(
        "Dans une version production, l'utilisateur serait identifié automatiquement via son compte, "
        "sans sélecteur manuel de profil."
    )


try:
    main()
except Exception as e:
    st.error("Erreur capturée dans l'application :")
    st.exception(e)
