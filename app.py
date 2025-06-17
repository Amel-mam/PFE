import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# 1. FONCTION TOPSIS (INCHANGÉE)
# Elle attend toujours les poids sous forme décimale (ex: 0.35)
# ==============================================================================
def topsis(df, criteria_weights, benefit_criteria, cost_criteria):
    """
    Implémente la méthode TOPSIS. Prend un DataFrame directement en entrée.
    """
    df_topsis = df.copy()
    criteria_columns = list(criteria_weights.keys())
    
    for col in criteria_columns:
        if col not in df_topsis.columns:
            st.error(f"Erreur : La colonne '{col}' semble avoir disparu du fichier. Veuillez recharger la page.")
            return None
    
    for col in criteria_columns:
        df_topsis[col] = pd.to_numeric(df_topsis[col], errors='coerce')

    initial_rows = len(df_topsis)
    df_topsis.dropna(subset=criteria_columns, inplace=True)
    final_rows = len(df_topsis)
    if final_rows < initial_rows:
        st.warning(f"{initial_rows - final_rows} ligne(s) ont été ignorées en raison de données manquantes ou non-valides dans les colonnes de critères.")

    if df_topsis.empty:
        st.error("Le DataFrame est vide après suppression des lignes avec données manquantes. Impossible de continuer.")
        return None
    
    df_topsis.reset_index(drop=True, inplace=True)

    alternatives = df_topsis.iloc[:, 0].tolist()
    evaluations_df = df_topsis[criteria_columns]

    sum_sq_evaluations = np.sqrt(np.sum(evaluations_df**2, axis=0))
    sum_sq_evaluations[sum_sq_evaluations == 0] = 1
    normalized_matrix = evaluations_df / sum_sq_evaluations

    weighted_matrix = normalized_matrix.copy()
    for col, weight in criteria_weights.items():
        weighted_matrix[col] = weighted_matrix[col] * weight

    pis, nis = [], []
    for col in criteria_columns:
        if col in benefit_criteria:
            pis.append(weighted_matrix[col].max())
            nis.append(weighted_matrix[col].min())
        else:
            pis.append(weighted_matrix[col].min())
            nis.append(weighted_matrix[col].max())
            
    pis, nis = np.array(pis), np.array(nis)
    d_plus = np.sqrt(np.sum((weighted_matrix - pis)**2, axis=1))
    d_minus = np.sqrt(np.sum((weighted_matrix - nis)**2, axis=1))

    denominator = d_plus + d_minus
    cc_values = np.divide(d_minus.to_numpy(), denominator.to_numpy(), out=np.zeros_like(d_minus.to_numpy(), dtype=float), where=denominator.to_numpy()!=0)

    results_df = pd.DataFrame({
        'Alternative': alternatives,
        'Coefficient de Proximité (CC)': cc_values
    })
    results_df['Classement'] = results_df['Coefficient de Proximité (CC)'].rank(ascending=False, method='min').astype(int)
    results_df = results_df.sort_values(by='Classement').reset_index(drop=True)

    return results_df

# ==============================================================================
# 2. INTERFACE STREAMLIT AVEC SAISIE EN POURCENTAGE
# ==============================================================================
st.set_page_config(layout="wide")
st.title("Classement des puits par la Méthode TOPSIS")

st.sidebar.header("Paramètres des puits")

uploaded_file = st.sidebar.file_uploader("1. Chargez votre fichier de données (Excel)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.replace('\xa0', ' ').str.strip()
        st.sidebar.success("Fichier chargé avec succès !")
        
        all_cols = df.columns.tolist()
        potential_criteria = all_cols[1:]

        st.sidebar.subheader("2. Sélectionnez les Critères")
        
        benefit_criteria = st.sidebar.multiselect(
            "Critères à maximiser", 
            options=potential_criteria,
            default=[c for c in ['Débit cumulée', 'Volume de gaz', 'THP moyen'] if c in potential_criteria]
        )
        
        remaining_criteria = [c for c in potential_criteria if c not in benefit_criteria]
        
        cost_criteria = st.sidebar.multiselect(
            "Critères à minimiser",
            options=remaining_criteria,
            default=[c for c in ["Production d'eau", 'Distance au CPF'] if c in remaining_criteria]
        )

        selected_criteria = benefit_criteria + cost_criteria
        
        # Ce dictionnaire stockera les poids saisis par l'utilisateur (ex: 35.0)
        weights_in_percent = {}
        
        if selected_criteria:
            # MODIFIÉ : Titre de la section des poids
            st.sidebar.subheader("3. Poids des Critères (résultats de la méthode AHP)")
            
            total_weight_percent = 0
            for crit in selected_criteria:
                default_weight = 100.0 / len(selected_criteria)
                
                # MODIFIÉ : Saisie en pourcentage (0 à 100)
                weights_in_percent[crit] = st.sidebar.number_input(
                    f"Poids pour '{crit}' (%)", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=default_weight, 
                    step=0.1, 
                    format="%.2f",
                    key=f"weight_{crit}"
                )
                total_weight_percent += weights_in_percent[crit]
            
            # MODIFIÉ : Affichage de la somme en pourcentage
            st.sidebar.metric("Somme des Poids", f"{total_weight_percent:.2f} %")
            if not np.isclose(total_weight_percent, 100.0):
                st.sidebar.warning("La somme des poids devrait être égale à 100%.")

        # --- PAGE PRINCIPALE POUR LES RÉSULTATS ---
        st.header("Résultats du Classement")

        if st.button("Lancer l'Analyse"):
            if not selected_criteria:
                st.error("Veuillez sélectionner au moins un critère dans la barre latérale.")
            # MODIFIÉ : Vérification par rapport à 100
            elif not np.isclose(sum(weights_in_percent.values()), 100.0):
                 st.error("La somme des poids doit être égale à 100% pour lancer l'analyse.")
            else:
                with st.spinner("Calcul du classement en cours..."):
                    # MODIFIÉ : Conversion des poids de % en décimales avant d'appeler TOPSIS
                    weights_for_topsis = {key: value / 100.0 for key, value in weights_in_percent.items()}
                    
                    topsis_results_df = topsis(df, weights_for_topsis, benefit_criteria, cost_criteria)
                    
                    if topsis_results_df is not None:
                        st.dataframe(topsis_results_df, height=600, use_container_width=True)

                        st.subheader("Meilleur Puits")
                        best_alternative = topsis_results_df.iloc[0]
                        st.success(f"L'alternative la mieux classée est **{best_alternative['Alternative']}** avec un score de {best_alternative['Coefficient de Proximité (CC)']:.4f}.")
                        
                        csv = topsis_results_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
                        st.download_button(
                           label="Télécharger les résultats en CSV",
                           data=csv,
                           file_name='classement_puits_topsis.csv',
                           mime='text/csv',
                        )

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement du fichier : {e}")
else:
    st.info("Veuillez charger un fichier Excel pour commencer l'analyse.")