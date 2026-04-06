import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote

st.set_page_config(page_title="Agent IA de recommandation e-commerce", layout="wide")


# =========================
# Data loading and cleaning
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("electronics_filtered.csv")
    df.columns = [c.strip() for c in df.columns]

    required = {"user_id", "product_id", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes obligatoires manquantes dans electronics_filtered.csv : {missing}")

    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["user_id", "product_id", "rating"])

    # Fallbacks si le dataset ne contient pas encore les colonnes enrichies
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

    if "image_url" not in df.columns:
        df["image_url"] = pd.NA

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

    # Catalogue produit enrichi
    product_catalog = (
        df.groupby("product_id", as_index=False)
        .agg(
            product_title=("product_title", "first"),
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count"),
            price=("price", "first"),
            image_url=("image_url", "first"),
        )
    )

    return user_item_matrix, user_similarity, product_catalog


# =========================
# Recommendation logic
# =========================

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

    if "bien noté" in q or "meilleur" in q:
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

    if "pas cher" in q or "moins cher" in q or "budget" in q:
        priced = product_catalog.dropna(subset=["price"]).sort_values(["price", "avg_rating"], ascending=[True, False]).head(top_n)
        if priced.empty:
            return fallback_df, "Pas de prix disponible dans le dataset. Retour au moteur collaboratif."
        return priced, "Filtre agent IA : produits budget."

    return fallback_df, "Aucune règle métier détectée. Retour au moteur collaboratif."


# =========================
# Display helpers
# =========================

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


def safe_image_url(row):
    image_url = row.get("image_url", None)
    if pd.notna(image_url) and str(image_url).strip():
        return str(image_url).strip()

    title = str(row.get("product_title", "Produit"))
    return f"https://via.placeholder.com/300x200.png?text={quote(title[:30])}"


def render_product_card(row):
    title = str(row.get("product_title", "Produit"))
    price_text = price_to_text(row.get("price", pd.NA))
    avg_rating = row.get("avg_rating", None)
    rating_count = row.get("rating_count", None)
    product_id = str(row.get("product_id", ""))

    st.markdown(
        """
        <style>
        .product-card {
            border: 1px solid #e6e6e6;
            border-radius: 16px;
            padding: 14px;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
            height: 100%;
        }
        .product-title {
            font-size: 16px;
            font-weight: 600;
            line-height: 1.3;
            min-height: 42px;
            margin-bottom: 8px;
        }
        .product-meta {
            color: #555;
            font-size: 13px;
            margin-top: 6px;
        }
        .product-price {
            font-size: 22px;
            font-weight: 700;
            margin-top: 8px;
            margin-bottom: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=False):
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        st.image(safe_image_url(row), use_container_width=True)
        st.markdown(f'<div class="product-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="product-price">{price_text}</div>', unsafe_allow_html=True)

        if pd.notna(avg_rating):
            st.markdown(
                f'<div class="product-meta">{stars(avg_rating)} '
                f'{float(avg_rating):.2f}/5 - {int(rating_count)} avis</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="product-meta">Pas de note disponible</div>', unsafe_allow_html=True)

        st.caption(f"Référence : {product_id}")
        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# App
# =========================

def main():
    st.title("Agent IA de recommandation e-commerce")
    st.caption("UX inspirée des marketplaces pour une démo plus claire et plus professionnelle.")

    df = load_data()
    user_item_matrix, user_similarity, product_catalog = build_artifacts(df)

    st.sidebar.header("Paramètres")
    user_id = st.sidebar.selectbox("Choisissez un utilisateur", user_item_matrix.index.tolist())
    top_n = st.sidebar.slider("Nombre de recommandations", min_value=3, max_value=10, value=6)

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
        "Cette version ajoute une présentation plus proche d'un site e-commerce : "
        "nom produit, prix, image et cartes visuelles."
    )
    st.write(
        "Si ton dataset ne contient pas encore de vraies colonnes `product_title`, `price` ou `image_url`, "
        "l'application utilise des valeurs de secours pour garder une UX propre."
    )


try:
    main()
except Exception as e:
    st.error("Erreur capturée dans l'application :")
    st.exception(e)
