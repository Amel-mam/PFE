import streamlit as st
import math
import random
import copy
import numpy as np
import pandas as pd
import time
from collections import deque
import plotly.graph_objects as go

# =============================================================================
# PART 1: CORE LOGIC (Unchanged)
# =============================================================================

@st.cache_data
def charger_donnees_depuis_excel(uploaded_file):
    try:
        donnees = {}
        CONVERSION_MILE_TO_M = 1609.34
        df_puits = pd.read_excel(uploaded_file, sheet_name='Puits')
        puits_data = {row['NomPuits']: {'x': float(row['X']), 'y': float(row['Y']), 'production_sm3d': float(str(row['Debit_kSm3_day']).replace(',', '.')) * 1000, 'diametre_flowline_in': row['DiametreFlowline_in'], 'CDF_i_dollar_par_m': round(358.69 / CONVERSION_MILE_TO_M, 5) if row['DiametreFlowline_in'] == 4 else round(454.06 / CONVERSION_MILE_TO_M, 5)} for _, row in df_puits.iterrows()}
        donnees['puits'] = puits_data
        df_manifolds = pd.read_excel(uploaded_file, sheet_name='TypesManifold')
        donnees['manifolds'] = {row['TypeID']: {'capacite_puits': int(row['CapacitePuits']), 'cout_installation_dollar': float(row['CoutInstallation_dollar'])} for _, row in df_manifolds.iterrows()}
        df_trunklines = pd.read_excel(uploaded_file, sheet_name='SpecsTrunkline')
        donnees['trunklines'] = sorted([{'diametre_in': int(r['Diametre_in']), 'cout_par_m_dollar': round(float(r['CoutParMile_dollar']) / CONVERSION_MILE_TO_M, 5), 'min_total_prod_sm3d': int(r['MinProd_Sm3d']), 'max_total_prod_sm3d': int(r['MaxProd_Sm3d'])} for _, r in df_trunklines.iterrows()], key=lambda x: x['min_total_prod_sm3d'])
        return donnees
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier Excel : {e}")
        return None

def distance(p1, p2):
    return math.sqrt((p1.get('x', 0) - p2.get('x', 0))**2 + (p1.get('y', 0) - p2.get('y', 0))**2)

def get_trunkline_params(prod, specs):
    for spec in specs:
        if spec['min_total_prod_sm3d'] <= prod <= spec['max_total_prod_sm3d']:
            return spec['diametre_in'], spec['cout_par_m_dollar']
    if prod > 0:
        s_specs = sorted(specs, key=lambda x: x['max_total_prod_sm3d'])
        if prod > s_specs[-1]['max_total_prod_sm3d']: return s_specs[-1]['diametre_in'], s_specs[-1]['cout_par_m_dollar']
        if prod < s_specs[0]['min_total_prod_sm3d']: return s_specs[0]['diametre_in'], s_specs[0]['cout_par_m_dollar']
    return None, None

def calculer_cout_total_complet(sol, puits, m_types, t_specs, cpf_cost):
    if not sol or not sol.get('manifolds_ouverts') or not sol.get('cpf_location'): return float('inf')
    cost = cpf_cost
    active_manifolds = {k: v for k, v in sol['manifolds_ouverts'].items() if v.get('puits_connectes')}
    for info in active_manifolds.values():
        cost += m_types[info['type_id']]['cout_installation_dollar']
    for p_id, m_id in sol['affectations_puits'].items():
        if m_id not in active_manifolds: return float('inf')
        cost += distance(puits[p_id], active_manifolds[m_id]) * puits[p_id]['CDF_i_dollar_par_m']
    cpf_loc = sol['cpf_location']
    for m_id, info in active_manifolds.items():
        prod = sum(puits[p_id]['production_sm3d'] for p_id in info['puits_connectes'])
        d, c = get_trunkline_params(prod, t_specs)
        sol['manifolds_ouverts'][m_id]['total_production_sm3d'] = prod
        sol['manifolds_ouverts'][m_id]['DTj_in'] = d
        sol['manifolds_ouverts'][m_id]['CDTj_dollar_par_m'] = c
        if c: cost += distance(info, cpf_loc) * c
        elif prod > 0: return float('inf')
    return cost

def construire_solution_clustering(puits_data, types_manifold_data):
    clusters = {p_id: {'puits': {p_id}, 'center': p_data} for p_id, p_data in puits_data.items()}
    max_cap = max(info['capacite_puits'] for info in types_manifold_data.values())
    while len(clusters) > 1:
        meilleure_fusion = {'dist': float('inf'), 'pair': None}
        cluster_ids = list(clusters.keys())
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                id1, id2 = cluster_ids[i], cluster_ids[j]
                if len(clusters[id1]['puits']) + len(clusters[id2]['puits']) <= max_cap:
                    dist_clusters = distance(clusters[id1]['center'], clusters[id2]['center'])
                    if dist_clusters < meilleure_fusion['dist']:
                        meilleure_fusion = {'dist': dist_clusters, 'pair': (id1, id2)}
        if meilleure_fusion['pair'] is None: break
        id1, id2 = meilleure_fusion['pair']
        c1, c2 = clusters.pop(id1), clusters.pop(id2)
        nouveau_cluster_puits = c1['puits'].union(c2['puits'])
        n_puits = len(nouveau_cluster_puits)
        nouveau_centre_x = sum(puits_data[p]['x'] for p in nouveau_cluster_puits) / n_puits
        nouveau_centre_y = sum(puits_data[p]['y'] for p in nouveau_cluster_puits) / n_puits
        nouveau_id = f"Cluster-{len(clusters)}"
        clusters[nouveau_id] = {'puits': nouveau_cluster_puits, 'center': {'x': nouveau_centre_x, 'y': nouveau_centre_y}}
    solution = {'manifolds_ouverts': {}, 'affectations_puits': {}}
    types_tries = sorted(types_manifold_data.items(), key=lambda item: item[1]['capacite_puits'])
    for i, cluster in enumerate(clusters.values()):
        manifold_id = f"MFD-{i+1}"
        type_choisi_id = next((tid for tid, tinfo in types_tries if len(cluster['puits']) <= tinfo['capacite_puits']), None)
        if type_choisi_id is None: continue
        solution['manifolds_ouverts'][manifold_id] = {'type_id': type_choisi_id, 'x': cluster['center']['x'], 'y': cluster['center']['y'], 'puits_connectes': cluster['puits']}
        for p_id in cluster['puits']: solution['affectations_puits'][p_id] = manifold_id
    return solution

def optimiser_cpf_pour_manifolds_fixes(solution_p1, puits_data, m_types, sites_cpf, t_specs, cpf_cost):
    if not solution_p1 or not solution_p1.get('manifolds_ouverts'): return None, float('inf')
    meilleure_solution, meilleur_cout = None, float('inf')
    for cpf_coords in sites_cpf:
        sol_temp = copy.deepcopy(solution_p1)
        sol_temp['cpf_location'] = cpf_coords
        cout_actuel = calculer_cout_total_complet(sol_temp, puits_data, m_types, t_specs, cpf_cost)
        if cout_actuel < meilleur_cout:
            meilleur_cout, meilleure_solution = cout_actuel, sol_temp
    if meilleure_solution: meilleure_solution['cout_total'] = meilleur_cout
    return meilleure_solution, meilleur_cout

def recuit_simule(sol_init, data, params, sites_candidats, cpf_is_fixed, progress_bar):
    T, T_FINAL, ALPHA, N_ITER = params['SA']['T_INITIAL'], params['SA']['T_FINAL'], params['SA']['ALPHA'], params['SA']['N_ITER_TEMP']
    prob_moves = copy.deepcopy(params['PROB_MOVES'])
    if cpf_is_fixed and 'relocate_cpf' in prob_moves: del prob_moves['relocate_cpf']
    sol_actuelle, meilleur_sol = copy.deepcopy(sol_init), copy.deepcopy(sol_init)
    cout_actuel = meilleur_cout = calculer_cout_total_complet(sol_actuelle, data['puits'], data['manifolds'], data['trunklines'], params['CPF_COST'])
    total_steps = int(math.log(T_FINAL / T) / math.log(ALPHA)) if T > 0 and ALPHA < 1 else 1
    current_step = 0
    status_text = st.empty()
    while T > T_FINAL:
        for _ in range(N_ITER):
            sol_voisine = generer_voisin_sa(sol_actuelle, data['manifolds'], sites_candidats, prob_moves)
            cout_voisin = calculer_cout_total_complet(sol_voisine, data['puits'], data['manifolds'], data['trunklines'], params['CPF_COST'])
            delta = cout_voisin - cout_actuel
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                sol_actuelle, cout_actuel = sol_voisine, cout_voisin
                if cout_actuel < meilleur_cout:
                    meilleur_cout, meilleur_sol = cout_actuel, copy.deepcopy(sol_actuelle)
        T *= ALPHA
        current_step += 1
        progress_bar.progress(min(current_step / total_steps, 1.0)) 
        status_text.text(f"Température: {T:.2f}, Meilleur Coût: ${meilleur_cout:,.2f}")
    status_text.text(f"Recuit Simulé terminé. Meilleur coût final: ${meilleur_cout:,.2f}")
    meilleur_sol['cout_total'] = meilleur_cout
    return meilleur_sol, meilleur_cout

def recherche_tabou(sol_init, data, params, sites_candidats, cpf_is_fixed, progress_bar):
    MAX_ITER, TENURE, N_NEIGHBORS = params['TS']['MAX_ITERATIONS'], params['TS']['TABU_TENURE'], params['TS']['N_NEIGHBORS_TO_EVAL']
    prob_moves = copy.deepcopy(params['PROB_MOVES'])
    if cpf_is_fixed and 'relocate_cpf' in prob_moves: del prob_moves['relocate_cpf']
    sol_actuelle = copy.deepcopy(sol_init)
    meilleure_sol = copy.deepcopy(sol_init)
    meilleur_cout = calculer_cout_total_complet(meilleure_sol, data['puits'], data['manifolds'], data['trunklines'], params['CPF_COST'])
    tabu_list = deque(maxlen=TENURE)
    status_text = st.empty()
    for i in range(MAX_ITER):
        sol_voisine, tabu_attr = generer_voisin_et_move_info_ts(sol_actuelle, data['manifolds'], sites_candidats, prob_moves)
        if sol_voisine:
            cout_voisin = calculer_cout_total_complet(sol_voisine, data['puits'], data['manifolds'], data['trunklines'], params['CPF_COST'])
            is_tabu = tabu_attr in tabu_list
            if not is_tabu or cout_voisin < meilleur_cout:
                sol_actuelle = sol_voisine
                if tabu_attr: tabu_list.append(tabu_attr)
                if cout_voisin < meilleur_cout:
                    meilleur_cout, meilleure_sol = cout_voisin, copy.deepcopy(sol_actuelle)
        progress_bar.progress(min((i + 1) / MAX_ITER, 1.0)) 
        status_text.text(f"Itération {i+1}/{MAX_ITER}, Meilleur Coût: ${meilleur_cout:,.2f}")
    status_text.text(f"Recherche Tabou terminée. Meilleur coût final: ${meilleur_cout:,.2f}")
    meilleure_sol['cout_total'] = meilleur_cout
    return meilleure_sol, meilleur_cout

def generer_voisin_sa(solution, types_manifold_data, sites_candidats, prob_moves):
    voisin = copy.deepcopy(solution)
    if not prob_moves: return voisin
    move_type = random.choices(list(prob_moves.keys()), weights=list(prob_moves.values()))[0]
    active_mfd_ids = [mid for mid, mdata in voisin['manifolds_ouverts'].items() if mdata.get('puits_connectes')]
    if not active_mfd_ids: return voisin
    if move_type == 'reassign_well' and len(active_mfd_ids) > 1:
        p_id = random.choice(list(voisin['affectations_puits'].keys()))
        m_id_origine = voisin['affectations_puits'].get(p_id)
        if not m_id_origine: return voisin
        candidats_dest = [mid for mid in active_mfd_ids if mid != m_id_origine and len(voisin['manifolds_ouverts'][mid]['puits_connectes']) < types_manifold_data[voisin['manifolds_ouverts'][mid]['type_id']]['capacite_puits']]
        if not candidats_dest: return voisin
        m_id_dest = random.choice(candidats_dest)
        voisin['affectations_puits'][p_id] = m_id_dest
        voisin['manifolds_ouverts'][m_id_origine]['puits_connectes'].remove(p_id)
        voisin['manifolds_ouverts'][m_id_dest]['puits_connectes'].add(p_id)
    elif move_type == 'relocate_manifold':
        m_id = random.choice(active_mfd_ids)
        new_site = random.choice(sites_candidats)
        voisin['manifolds_ouverts'][m_id]['x'] = new_site['x']
        voisin['manifolds_ouverts'][m_id]['y'] = new_site['y']
    elif move_type == 'relocate_cpf':
        voisin['cpf_location'] = random.choice(sites_candidats)
    return voisin

def generer_voisin_et_move_info_ts(solution, types_manifold_data, sites_candidats, prob_moves):
    voisin, tabu_attribute = copy.deepcopy(solution), None
    if not prob_moves: return None, None
    move_type = random.choices(list(prob_moves.keys()), weights=list(prob_moves.values()))[0]
    active_mfds = {mid: mdata for mid, mdata in voisin['manifolds_ouverts'].items() if mdata.get('puits_connectes')}
    active_mfd_ids = list(active_mfds.keys())
    if not active_mfd_ids: return None, None
    if move_type == 'reassign_well' and len(active_mfd_ids) > 1:
        p_id = random.choice(list(voisin['affectations_puits'].keys()))
        m_from = voisin['affectations_puits'].get(p_id)
        if not m_from: return None, None
        candidats_dest = [mid for mid in active_mfd_ids if mid != m_from and len(active_mfds[mid]['puits_connectes']) < types_manifold_data[active_mfds[mid]['type_id']]['capacite_puits']]
        if not candidats_dest: return None, None
        m_to = random.choice(candidats_dest)
        voisin['affectations_puits'][p_id] = m_to
        voisin['manifolds_ouverts'][m_from]['puits_connectes'].remove(p_id)
        voisin['manifolds_ouverts'][m_to]['puits_connectes'].add(p_id)
        tabu_attribute = ('well_reassign', p_id)
    elif move_type == 'relocate_manifold':
        m_id = random.choice(active_mfd_ids)
        new_site = random.choice(sites_candidats)
        voisin['manifolds_ouverts'][m_id]['x'] = new_site['x']
        voisin['manifolds_ouverts'][m_id]['y'] = new_site['y']
        tabu_attribute = ('manifold_relocate', m_id)
    elif move_type == 'relocate_cpf':
        voisin['cpf_location'] = random.choice(sites_candidats)
        tabu_attribute = ('cpf_relocate',)
    else: return None, None
    return voisin, tabu_attribute

def creer_visualisation(solution, puits_data):
    fig = go.Figure()
    puits_x = [p['x'] for p in puits_data.values()]
    puits_y = [p['y'] for p in puits_data.values()]
    puits_text = [f"Puits: {name}" for name in puits_data.keys()]
    fig.add_trace(go.Scatter(x=puits_x, y=puits_y, mode='markers', name='Puits', marker=dict(color='blue', size=8, symbol='circle'), text=puits_text, hoverinfo='text'))
    if not solution: return fig
    manifolds = solution.get('manifolds_ouverts', {})
    cpf_loc = solution.get('cpf_location')
    man_x = [m['x'] for m in manifolds.values()]
    man_y = [m['y'] for m in manifolds.values()]
    man_text = [f"Manifold: {mid}<br>Type: {m['type_id']}" for mid, m in manifolds.items()]
    fig.add_trace(go.Scatter(x=man_x, y=man_y, mode='markers', name='Manifolds', marker=dict(color='green', size=12, symbol='square'), text=man_text, hoverinfo='text'))
    if cpf_loc:
        fig.add_trace(go.Scatter(x=[cpf_loc['x']], y=[cpf_loc['y']], mode='markers', name='CPF', marker=dict(color='red', size=16, symbol='star'), text=[f"CPF<br>X:{cpf_loc['x']:.2f}, Y:{cpf_loc['y']:.2f}"], hoverinfo='text'))
    line_shapes = []
    for p_id, m_id in solution.get('affectations_puits', {}).items():
        if p_id in puits_data and m_id in manifolds:
            p, m = puits_data[p_id], manifolds[m_id]
            line_shapes.append(dict(type='line', x0=p['x'], y0=p['y'], x1=m['x'], y1=m['y'], line=dict(color='rgba(0,0,255,0.3)', width=1)))
    if cpf_loc:
        for m in manifolds.values():
            line_shapes.append(dict(type='line', x0=m['x'], y0=m['y'], x1=cpf_loc['x'], y1=cpf_loc['y'], line=dict(color='rgba(0,128,0,0.5)', width=3)))
    fig.update_layout(title="Visualisation de la solution", xaxis_title="Coordonnée X", yaxis_title="Coordonnée Y", shapes=line_shapes, legend_title="Légende", hovermode='closest')
    return fig

def afficher_resultats_streamlit(solution, types_manifold_data, titre):
    st.subheader(f"Rapport Détaillé: {titre}")
    if not solution or not solution.get('manifolds_ouverts'):
        st.warning("Aucune solution valide à afficher.")
        return
    cpf_loc = solution.get('cpf_location', {'x': 'N/A', 'y': 'N/A'})
    st.write(f"**Localisation CPF:** (X={cpf_loc.get('x', 'N/A'):.2f}, Y={cpf_loc.get('y', 'N/A'):.2f})")
    manifolds_actifs = {k: v for k, v in solution.get('manifolds_ouverts', {}).items() if v.get('puits_connectes')}
    st.write(f"**Nombre de Manifolds Actifs:** {len(manifolds_actifs)}")
    with st.expander("Voir les détails des manifolds"):
        for m_id, m_data in sorted(manifolds_actifs.items()):
            type_id, capacite = m_data.get('type_id', 'N/A'), types_manifold_data.get(m_data.get('type_id', 'N/A'), {}).get('capacite_puits', 'N/A')
            st.markdown(f"--- \n**Manifold ID: {m_id}**")
            st.write(f"- Coordonnées: (X={m_data.get('x', 0):.2f}, Y={m_data.get('y', 0):.2f})")
            st.write(f"- Type: {type_id} (Capacité: {capacite})")
            st.write(f"- Trunkline vers CPF: Diamètre {m_data.get('DTj_in', 'N/A')} in")
            puits_connectes = sorted(list(m_data.get('puits_connectes', [])))
            st.write(f"- Puits Connectés ({len(puits_connectes)}):")
            st.code(f"{' | '.join(puits_connectes)}")

# =============================================================================
# PART 3: STREAMLIT APP MAIN LOGIC
# =============================================================================

st.set_page_config(layout="wide")
st.title("Optimisation des réseaux de collecte gaziers")
st.write("Cette application conçoit la structure complète et la plus rentable pour des réseaux de collecte gaziers. Elle détermine simultanément l'emplacement le plus avantageux des manifolds et du CPF (Central Processing Facility), tout en assurant l'affectation la plus économique de chaque puits à un manifold. La solution initiale, générée par l'heuristique clustering, est ensuite affinée par des métaheuristiques (Recuit Simulé, Recherche Tabou).")
st.sidebar.header("Configuration de la Simulation")
uploaded_file = st.sidebar.file_uploader("1. Chargez votre fichier de données", type=["xlsx"])

if uploaded_file is not None:
    scenario_choice = st.sidebar.selectbox("2. Choisissez le scénario", ["CPF Optimisé", "CPF Fixe"])
    
    ### MODIFICATION 1: Changement du nom de l'heuristique ###
    algo_choice_map = {
        "Heuristique Clustering hiérarchique ascendant": "none",
        "Recuit Simulé (SA)": "sa",
        "Recherche Tabou (TS)": "ts"
    }
    algo_display_name = st.sidebar.selectbox("3. Choisissez l'algorithme", list(algo_choice_map.keys()))
    algo_choice = algo_choice_map[algo_display_name]

    with st.sidebar.expander("Paramètres avancés"):
        st.subheader("Paramètres Généraux")
        cpf_cost = st.number_input("Coût d'installation du CPF ($)", value=450000000.0, format="%.2f")
        marge_grille = st.number_input("Marge pour grille de sites (m)", value=5000.0)
        pas_grille = st.number_input("Pas de la grille de sites (m)", value=2000.0)
        if scenario_choice == "CPF Fixe":
            st.subheader("Coordonnées CPF Fixe")
            cpf_fixe_x = st.number_input("CPF Fixe X", value=449397.35)
            cpf_fixe_y = st.number_input("CPF Fixe Y", value=3048477.39)
        if algo_choice == "sa":
            st.subheader("Paramètres Recuit Simulé")
            t_initial = st.number_input("Température Initiale (T_INITIAL)", value=100000.0)
            t_final = st.number_input("Température Finale (T_FINAL)", value=1.0)
            alpha = st.slider("Coefficient de refroidissement (ALPHA)", 0.8, 0.999, 0.99, 0.001, format="%.3f")
            n_iter_temp = st.number_input("Itérations par palier (N_ITER_TEMP)", value=100)
        if algo_choice == 'ts':
            st.subheader("Paramètres Recherche Tabou")
            max_iter = st.number_input("Nombre max d'itérations", value=500)
            tabu_tenure = st.number_input("Taille de la liste Tabou", value=10)
            n_neighbors = st.number_input("Voisins à évaluer par itération", value=50)

    if st.sidebar.button("🚀 Lancer l'optimisation", use_container_width=True):
        donnees = charger_donnees_depuis_excel(uploaded_file)
        if donnees:
            PARAMS = {'CPF_COST': cpf_cost, 'PROB_MOVES': {'reassign_well': 0.5, 'relocate_manifold': 0.3, 'relocate_cpf': 0.2}, 'SA': {'T_INITIAL': t_initial if algo_choice == 'sa' else 0, 'T_FINAL': t_final if algo_choice == 'sa' else 0, 'ALPHA': alpha if algo_choice == 'sa' else 0, 'N_ITER_TEMP': n_iter_temp if algo_choice == 'sa' else 0}, 'TS': {'MAX_ITERATIONS': max_iter if algo_choice == 'ts' else 0, 'TABU_TENURE': tabu_tenure if algo_choice == 'ts' else 0, 'N_NEIGHBORS_TO_EVAL': n_neighbors if algo_choice == 'ts' else 0}}
            xmin, xmax, ymin, ymax = (min(d['x'] for d in donnees['puits'].values()), max(d['x'] for d in donnees['puits'].values()), min(d['y'] for d in donnees['puits'].values()), max(d['y'] for d in donnees['puits'].values()))
            sites_candidats = [{'x': round(x, 2), 'y': round(y, 2)} for x in np.arange(xmin - marge_grille, xmax + marge_grille, pas_grille) for y in np.arange(ymin - marge_grille, ymax + marge_grille, pas_grille)]
            
            with st.spinner("1. Génération de la solution via l'Heuristique Clustering hiérarchique ascendant..."):
                sol_p1 = construire_solution_clustering(donnees['puits'], donnees['manifolds'])
            
            if scenario_choice == "CPF Fixe":
                sol_initiale = copy.deepcopy(sol_p1)
                sol_initiale['cpf_location'] = {'x': cpf_fixe_x, 'y': cpf_fixe_y}
                cout_initial = calculer_cout_total_complet(sol_initiale, donnees['puits'], donnees['manifolds'], donnees['trunklines'], PARAMS['CPF_COST'])
            else:
                with st.spinner("2. Optimisation de la position initiale du CPF..."):
                    sol_initiale, cout_initial = optimiser_cpf_pour_manifolds_fixes(sol_p1, donnees['puits'], donnees['manifolds'], sites_candidats, donnees['trunklines'], PARAMS['CPF_COST'])
            
            if not sol_initiale or cout_initial == float('inf'):
                st.error("Impossible de générer une solution initiale valide.")
            else:
                sol_initiale['cout_total'] = cout_initial
                sol_finale, cout_final = sol_initiale, cout_initial
                cpf_is_fixed = (scenario_choice == "CPF Fixe")
                if algo_choice != 'none':
                    st.info(f"3. Amélioration avec {algo_display_name}...")
                    progress_bar = st.progress(0)
                    if algo_choice == 'sa':
                        sol_finale, cout_final = recuit_simule(sol_initiale, donnees, PARAMS, sites_candidats, cpf_is_fixed, progress_bar)
                    elif algo_choice == 'ts':
                        sol_finale, cout_final = recherche_tabou(sol_initiale, donnees, PARAMS, sites_candidats, cpf_is_fixed, progress_bar)
                    progress_bar.empty()
                st.success("✅ Calcul terminé !")

                st.header("📈 Résultats")
                
                ### MODIFICATION 2: Affichage conditionnel des résultats ###
                if algo_choice == 'none':
                    # Si seule l'heuristique est choisie, on n'affiche qu'un seul coût
                    st.metric("Coût Total (Heuristique)", f"${cout_final:,.2f}")
                else:
                    # Si une métaheuristique est choisie, on affiche la comparaison
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Coût Initial (Heuristique)", f"${cout_initial:,.2f}")
                    col2.metric("Coût Final (Amélioré)", f"${cout_final:,.2f}")
                    improvement = ((cout_initial - cout_final) / cout_initial) * 100 if cout_initial > 0 else 0
                    col3.metric("Amélioration", f"{improvement:.2f}%", delta=f"{improvement:.2f}%" if improvement > 0 else None)
                
                fig = creer_visualisation(sol_finale, donnees['puits'])
                st.plotly_chart(fig, use_container_width=True)
                afficher_resultats_streamlit(sol_finale, donnees['manifolds'], f"{scenario_choice} - {algo_display_name}")
else:
    st.info("Veuillez charger un fichier de données Excel pour commencer.")