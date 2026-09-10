import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="Prédiction état véhicule",
    page_icon="🚗",
    layout="centered"
)

# ==========================================
# CHARGEMENT DES FICHIERS
# ==========================================

@st.cache_resource
def charger_modele():

    # Importer les encodeurs
    encoders = jb.load("encoders_vehicule.joblib")

    # Importer les valeurs uniques
    uniques = jb.load("uniques_vehicule.joblib")

    # Importer le normaliseur
    scaler = jb.load("scaler_vehicule.joblib")

    # Importer le modèle
    xgb = jb.load("xgb_model_vehicule.joblib")

    return encoders, uniques, scaler, xgb


encoders, uniques, scaler, xgb = charger_modele()

# Noms des classes
clasnames = uniques[3]


# ==========================================
# FONCTION DE PREDICTION SIMPLE
# ==========================================

def Pred_func(annee, marque, transmission, quartier, prix):

    # Encoder les variables catégorielles
    marque = encoders[0].transform([marque])[0]

    transmission = encoders[1].transform([transmission])[0]

    quartier = encoders[2].transform([quartier])[0]

    # Créer le vecteur des données
    x_new = np.array([
        annee,
        marque,
        transmission,
        quartier,
        prix
    ])

    # Transformer en tableau 2D
    x_new = x_new.reshape(1, -1)

    # Normaliser les données
    x_new = scaler.transform(x_new)

    # Prédiction
    y_pred = xgb.predict(x_new)

    # Retourner le nom de la classe
    return clasnames[y_pred[0]]


# ==========================================
# FONCTION DE PREDICTION MULTIPLE CSV
# ==========================================

def Pred_func_csv(df):

    predictions = []

    # Boucle sur toutes les lignes
    for row in df.iloc[:, :5].values:

        y_pred = Pred_func(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4]
        )

        predictions.append(y_pred)

    # Ajouter la colonne Etat
    df["etat"] = predictions

    return df


# ==========================================
# TITRE DE L'APPLICATION
# ==========================================

st.title("🚗 Prédiction de l'état d'un véhicule")

st.write("""
Cette application utilise le **Machine Learning** pour prédire
si un véhicule est **neuf ou d'occasion** à partir de ses caractéristiques.
""")


# ==========================================
# CREATION DES ONGLETS
# ==========================================

tab1, tab2 = st.tabs([
    "🔍 Prédiction simple",
    "📂 Prédiction multiple"
])


# ==========================================
# ONGLET 1 : PREDICTION SIMPLE
# ==========================================

with tab1:

    st.subheader("Prédiction avec une seule entrée")

    annee = st.number_input(
        "Année",
        min_value=1900,
        max_value=2030,
        value=2020
    )

    marque = st.selectbox(
        "Marque",
        uniques[0]
    )

    transmission = st.selectbox(
        "Transmission",
        uniques[1]
    )

    quartier = st.selectbox(
        "Quartier",
        uniques[2]
    )

    prix = st.number_input(
        "Prix",
        min_value=0.0,
        value=1000000.0
    )

    if st.button("🚗 Prédire l'état du véhicule"):

        prediction = Pred_func(
            annee,
            marque,
            transmission,
            quartier,
            prix
        )

        st.success(
            f"### État prédit du véhicule : **{prediction}**"
        )


# ==========================================
# ONGLET 2 : PREDICTION MULTIPLE
# ==========================================

with tab2:

    st.subheader("Prédiction à partir d'un fichier CSV")

    st.write("""
    Le fichier CSV doit contenir les colonnes dans cet ordre :

    **Année | Marque | Transmission | Quartier | Prix**
    """)

    uploaded_file = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("### Aperçu des données")

        st.dataframe(df.head())

        if st.button("📊 Prédire les états des véhicules"):

            resultat = Pred_func_csv(df)

            st.write("### Résultats")

            st.dataframe(resultat)

            # Conversion en CSV
            csv = resultat.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Télécharger les prédictions",
                data=csv,
                file_name="predictions_vehicules.csv",
                mime="text/csv"
            )
