import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="💳 Détection de Fraude", layout="wide")

def load_model():
    return joblib.load("fraud_detector_model.pkl")

model = load_model()

st.title("💳 Système de Détection de Fraude Bancaire")

st.markdown("""
Bienvenue sur l'application de détection de fraudes ! 
Importez vos transactions bancaires (format `.csv`), puis cliquez sur **Détecter les fraudes**.
""")

uploaded_file = st.file_uploader("📥 Choisissez un fichier CSV", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("🧾 Aperçu des données")
        st.dataframe(data.head(), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Erreur de lecture du fichier : {e}")

    if st.button("🔍 Détecter les fraudes"):
        with st.spinner("Analyse en cours..."):
            X = data.drop(columns=["Class"]) if "Class" in data.columns else data.copy()
            predictions = model.predict(X)

            data["Fraude"] = predictions

            n_fraudes = (data["Fraude"] == 1).sum()
            total = len(data)
            taux = round((n_fraudes / total) * 100, 2)

            st.success("✅ Analyse terminée.")
            col1, col2 = st.columns(2)
            col1.metric("⚠️ Fraudes détectées", n_fraudes)
            col2.metric("📊 Taux de fraude", f"{taux} %")

            st.subheader("📊 Répartition des transactions")
            fig = px.histogram(data, x="Fraude", color="Fraude",
                               labels={"Fraude": "Transaction"},
                               color_discrete_map={0: "green", 1: "red"},
                               category_orders={"Fraude": [0, 1]},
                               nbins=2)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🚨 Transactions Frauduleuses")
            fraudes_df = data[data["Fraude"] == 1].reset_index(drop=True)
            nb_total = len(fraudes_df)
            page_size = 10

            if nb_total > 0:
                nb_pages = (nb_total - 1) // page_size + 1
                page = st.number_input("📄 Page", min_value=1, max_value=nb_pages, value=1, step=1)

                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, nb_total)

                st.write(f"Affichage des lignes {start_idx + 1} à {end_idx} sur {nb_total} fraudes détectées.")
                st.dataframe(fraudes_df.iloc[start_idx:end_idx], use_container_width=True)
            else:
                st.info("✅ Aucune fraude détectée.")

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Télécharger les résultats", data=csv,
                               file_name="transactions_avec_fraudes.csv", mime="text/csv")

            def afficher_bar_chart(df):
                total = len(df)
                fraudes = df[df['Fraude'] == 1].shape[0]
                normales = total - fraudes
                taux_fraude = (fraudes / total) * 100

                fig, ax = plt.subplots()
                sns.barplot(
                    x=['Transactions normales', 'Fraudes détectées'],
                    y=[normales, fraudes],
                    palette='Set2',
                    ax=ax
                )
                ax.set_ylabel("Nombre de transactions")
                ax.set_title(f"Taux de fraude détecté : {taux_fraude:.2f}%")
                st.pyplot(fig)

            def afficher_donut_chart(df):
                total = len(df)
                fraudes = df[df['Fraude'] == 1].shape[0]
                normales = total - fraudes

                labels = ['Normales', 'Fraudes']
                sizes = [normales, fraudes]
                colors = ['#66b3ff', '#ff6666']

                fig, ax = plt.subplots()
                wedges, texts, autotexts = ax.pie(
                    sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90, textprops=dict(color="white")
                )
                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                fig.gca().add_artist(centre_circle)
                ax.set_title("Répartition des transactions")
                st.pyplot(fig)

            st.subheader("📈 Visualisations complémentaires")
            afficher_bar_chart(data)
            afficher_donut_chart(data)
