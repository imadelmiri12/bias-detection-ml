import streamlit as st

# =====================================================
# CONFIGURATION (TOUJOURS EN PREMIER)
# =====================================================
st.set_page_config(
    page_title="Détection de biais – Adult Income",
    layout="centered"
)

# =====================================================
# IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from statsmodels.stats.proportion import proportions_ztest

from streamlit_option_menu import option_menu

# =====================================================
# STYLE CSS
# =====================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TITRE
# =====================================================
st.title("Détection de biais dans un modèle de recommandation")
st.write("**Dataset : Adult Income (UCI)**")

# =====================================================
# MENU
# =====================================================
with st.sidebar:
    page = option_menu(
        "Navigation",
        [
            "Description des données",
            "Machine Learning",
            "Analyse statistique",
            "Test d’hypothèse",
            "Comparaison",
            "Conclusion"
        ],
        icons=["table", "cpu", "graph-up", "check-square", "shuffle", "book"],
        menu_icon="list",
        default_index=0
    )

# =====================================================
# CHARGEMENT DES DONNÉES
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("adult.csv")

df = load_data()

# =====================================================
# PRÉTRAITEMENT (IDENTIQUE)
# =====================================================
df["income"] = df["income"].astype(str).str.strip()
df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

y = df["income"]
S = df["gender"]

X = df.drop(columns=["income", "gender"])
X = X.replace("?", pd.NA)

mask = X.notna().all(axis=1)
X = X.loc[mask]
y = y.loc[mask]
S = S.loc[mask]

df_clean = pd.concat([X, S, y], axis=1)
X_encoded = pd.get_dummies(X, drop_first=True)

# 🔴 copies pour éviter bug WRITEABLE
X_encoded = X_encoded.copy()
y = y.copy()
S = S.copy()

X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
    X_encoded,
    y,
    S,
    test_size=0.30,
    random_state=42,
    stratify=y
)

male_mask = (S_test == "Male").values
female_mask = (S_test == "Female").values

# =====================================================
# ENTRAÎNEMENT DES MODÈLES (CACHE)
# =====================================================
@st.cache_resource
def train_models(X_train, y_train):
    logreg = LogisticRegression(max_iter=2000)
    logreg.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    return logreg, rf

with st.spinner("Entraînement des modèles..."):
    logreg, rf = train_models(X_train, y_train)

y_pred_lr = logreg.predict(X_test)
y_proba_lr = logreg.predict_proba(X_test)[:, 1]

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# =====================================================
# BOOTSTRAP (CACHE — CONTENU IDENTIQUE)
# =====================================================
@st.cache_data
def bootstrap_diff(y_pred, male_mask, female_mask, B=2000):
    np.random.seed(42)
    diff_boot = []

    male_idx = np.where(male_mask)[0]
    female_idx = np.where(female_mask)[0]

    for _ in range(B):
        sm = np.random.choice(male_idx, size=len(male_idx), replace=True)
        sf = np.random.choice(female_idx, size=len(female_idx), replace=True)
        diff_boot.append(np.mean(y_pred[sm]) - np.mean(y_pred[sf]))

    return np.array(diff_boot)

@st.cache_data
def bootstrap_diff(y_pred, male_mask, female_mask, B=2000):
    """
    Bootstrap non paramétrique de la différence des taux d’acceptation
    entre hommes et femmes.
    """
    np.random.seed(42)
    diff_boot = []

    male_idx = np.where(male_mask)[0]
    female_idx = np.where(female_mask)[0]

    for _ in range(B):
        sm = np.random.choice(male_idx, size=len(male_idx), replace=True)
        sf = np.random.choice(female_idx, size=len(female_idx), replace=True)

        p_m = np.mean(y_pred[sm])
        p_f = np.mean(y_pred[sf])
        diff_boot.append(p_m - p_f)

    return np.array(diff_boot)


# =====================================================
# 1️⃣ DESCRIPTION DES DONNÉES
# =====================================================
if page == "Description des données":

    st.header("1. Description statistique des données")

    st.write(f"Nombre d’observations : **{df_clean.shape[0]}**")
    st.write(f"Nombre de variables : **{df_clean.shape[1]}**")

    st.subheader("Aperçu du dataset")
    st.dataframe(df_clean.head())

    st.subheader("Distribution de la variable cible")
    income_dist = y.value_counts(normalize=True)

    fig, ax = plt.subplots()
    ax.bar(["<=50K", ">50K"], income_dist.sort_index())
    ax.set_ylabel("Proportion")
    st.pyplot(fig)

    st.subheader("Répartition income par genre")
    st.dataframe(pd.crosstab(S, y, normalize="index"))

    st.subheader("Statistiques descriptives")
    st.dataframe(df_clean.describe())

# =====================================================
# 2️⃣ MACHINE LEARNING
# =====================================================
elif page == "Machine Learning":

    st.header("2. Machine Learning")

    # =====================================================
    # OBJECTIF
    # =====================================================
    st.subheader("Objectif du modèle")

    st.write(
        """
        L’objectif du modèle de Machine Learning est de prédire si
        le revenu annuel d’un individu est supérieur à 50K$.
        Il s’agit d’un **problème de classification binaire**.
        """
    )

    # =====================================================
    # CHOIX DU MODELE
    # =====================================================
    st.subheader("Choix du modèle : Régression logistique")

    st.write(
        """
        La **régression logistique** a été retenue comme modèle principal
        pour les raisons suivantes :
        - elle est adaptée à la classification binaire,
        - elle fournit des probabilités interprétables,
        - elle est largement utilisée comme **modèle de référence (baseline)**,
        - elle facilite l’analyse de biais et l’audit algorithmique.

        La variable sensible **gender** n’est **pas incluse** dans les variables
        d’entraînement, afin d’évaluer l’apparition de biais indirects.
        """
    )

    # =====================================================
    # PIPELINE DE DONNEES
    # =====================================================
    st.subheader("Pipeline de données")

    st.markdown(
        """
        Le pipeline de Machine Learning est le suivant :
        - nettoyage des valeurs manquantes,
        - encodage des variables catégorielles par *One-Hot Encoding*,
        - séparation des données en **jeu d’entraînement (70 %)** et
          **jeu de test (30 %)**,
        - entraînement du modèle sur le jeu d’apprentissage,
        - évaluation des performances sur le jeu de test.
        """
    )

    # =====================================================
    # PERFORMANCE GLOBALE
    # =====================================================
    st.subheader("Performance globale du modèle")

    acc = accuracy_score(y_test, y_pred_lr)
    auc = roc_auc_score(y_test, y_proba_lr)

    st.write(f"🔹 **Accuracy** : {acc:.3f}")
    st.write(f"🔹 **AUC** : {auc:.3f}")

    st.write(
        """
        Ces métriques indiquent une **bonne performance globale** du modèle.
        Cependant, elles ne permettent pas à elles seules d’évaluer
        l’équité des décisions entre différents groupes.
        """
    )

    # =====================================================
    # MATRICE DE CONFUSION
    # =====================================================
    st.subheader("Matrice de confusion")

    cm = pd.crosstab(
        y_test,
        y_pred_lr,
        rownames=["Valeurs réelles"],
        colnames=["Prédictions"]
    )
    st.dataframe(cm)

    st.write(
        """
        La matrice de confusion permet d’identifier les erreurs de classification
        (faux positifs et faux négatifs), qui peuvent avoir des impacts
        différents selon les groupes étudiés.
        """
    )

    # =====================================================
    # COURBE ROC
    # =====================================================
    st.subheader("Courbe ROC")

    fpr, tpr, _ = roc_curve(y_test, y_proba_lr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    st.pyplot(fig)

    # =====================================================
    # INTERPRETABILITE
    # =====================================================
    st.subheader("Interprétation du modèle")

    coef_df = pd.DataFrame({
        "Variable": X_train.columns,
        "Coefficient": logreg.coef_[0]
    }).sort_values(by="Coefficient", key=abs, ascending=False).head(10)

    st.dataframe(coef_df)

    st.warning(
        """
        Les coefficients mesurent des **associations statistiques**
        et ne doivent pas être interprétés comme des relations causales.
        """
    )

    # =====================================================
    # LIEN AVEC LE BIAIS
    # =====================================================
    st.info(
        """
        Bien que la variable *gender* ne soit pas utilisée lors de l’entraînement,
        le modèle peut exploiter des **corrélations indirectes**
        présentes dans les données (ex. type de profession, nombre d’heures travaillées).
        Cela peut conduire à des décisions biaisées entre groupes.
        """
    )

# =====================================================
# 3️⃣ ANALYSE STATISTIQUE
# =====================================================
elif page == "Analyse statistique":

    st.header("3. Analyse statistique du biais")

    # =====================================================
    # CADRE STATISTIQUE
    # =====================================================
    st.subheader("Cadre statistique")

    st.write(
        """
        Les décisions produites par le modèle (acceptation / refus)
        sont assimilées à des **variables aléatoires de Bernoulli**.
        Le biais est analysé en comparant les **probabilités empiriques
        de décision positive** entre les groupes *Male* et *Female*.
        """
    )

    # =====================================================
    # ESTIMATION DU BIAIS
    # =====================================================
    st.subheader("Estimation des taux d’acceptation")

    p_male = np.mean(y_pred_lr[male_mask])
    p_female = np.mean(y_pred_lr[female_mask])
    diff_hat = p_male - p_female

    st.write(f"🔹 Taux d’acceptation (Male) : **{p_male:.3f}**")
    st.write(f"🔹 Taux d’acceptation (Female) : **{p_female:.3f}**")
    st.write(f"🔹 Différence estimée (Male − Female) : **{diff_hat:.3f}**")
    st.subheader("Définition de l’estimateur du biais")

    st.write(
    "Le biais est mesuré par la différence entre les probabilités "
    "empiriques de décision positive pour les groupes Male et Female."
)

    st.write("**Estimateur du biais :**")
    st.latex(r"\hat{d} = \hat{p}_{\text{Male}} - \hat{p}_{\text{Female}}")

    
    # =====================================================
    # VISUALISATION DES TAUX
    # =====================================================
    st.subheader("Visualisation des taux d’acceptation")

    fig1, ax1 = plt.subplots()
    ax1.bar(["Male", "Female"], [p_male, p_female])
    ax1.set_ylabel("Probabilité de décision positive")
    ax1.set_ylim(0, max(p_male, p_female) + 0.05)

    for i, v in enumerate([p_male, p_female]):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center")

    st.pyplot(fig1)

    # =====================================================
    # BOOTSTRAP : INTERVALLE DE CONFIANCE
    # =====================================================
    st.subheader("Incertitude : Bootstrap (Intervalle de confiance à 95 %)")

    st.write(
        """
        Afin de quantifier l’incertitude associée à l’estimation du biais,
        nous utilisons une méthode de **bootstrap non paramétrique**
        avec rééchantillonnage indépendant au sein de chaque groupe.
        """
    )

    with st.spinner("Calcul du bootstrap en cours..."):
        diff_boot = bootstrap_diff(
            y_pred_lr,
            male_mask,
            female_mask,
            B=2000
        )

    ci_low, ci_high = np.percentile(diff_boot, [2.5, 97.5])

    st.write(
        f" **Intervalle de confiance à 95 %** : "
        f"**[{ci_low:.3f} ; {ci_high:.3f}]**"
    )

    # =====================================================
    # DISTRIBUTION BOOTSTRAP
    # =====================================================
    st.subheader("Distribution bootstrap de la différence")

    fig2, ax2 = plt.subplots()
    ax2.hist(diff_boot, bins=40, density=True)
    ax2.axvline(ci_low, linestyle="--", label="IC 95 % (borne basse)")
    ax2.axvline(ci_high, linestyle="--", label="IC 95 % (borne haute)")
    ax2.axvline(0, linestyle=":", label="0 (absence de biais)")
    ax2.set_xlabel("Différence (Male − Female)")
    ax2.set_ylabel("Densité")
    ax2.legend()

    st.pyplot(fig2)

    # =====================================================
    # INTERPRÉTATION
    # =====================================================
    st.info(
        """
        👉 Si l’intervalle de confiance **ne contient pas 0**,
        la différence observée entre les groupes ne peut pas être
        attribuée uniquement au hasard d’échantillonnage.

        Cela constitue un **indice statistique fort de biais**
        dans les décisions du modèle.
        """
    )


elif page == "Test d’hypothèse":

    st.header("4. Test d’hypothèse : comparaison des proportions")

    # =====================================================
    # FORMULATION DU PROBLÈME
    # =====================================================
    st.subheader("Formulation des hypothèses")

    st.write(
            "On cherche à déterminer si la différence observée entre les "
            "taux d’acceptation des groupes Male et Female est "
            "statistiquement significative."
        )

    st.write("**Hypothèse nulle (H₀)** :")
    st.latex(r"p_{\text{Male}} = p_{\text{Female}}")

    st.write("**Hypothèse alternative (H₁)** :")
    st.latex(r"p_{\text{Male}} \neq p_{\text{Female}}")

    st.write("Test bilatéral avec un seuil de signification :")
    st.latex(r"\alpha = 5\%")




    # =====================================================
    # DONNÉES DU TEST
    # =====================================================
    st.subheader("Données utilisées pour le test")

    success_male = np.sum(y_pred_lr[male_mask])
    success_female = np.sum(y_pred_lr[female_mask])

    n_male = np.sum(male_mask)
    n_female = np.sum(female_mask)

    st.write(f"Nombre de décisions positives (Male) : **{success_male} / {n_male}**")
    st.write(f"Nombre de décisions positives (Female) : **{success_female} / {n_female}**")

    # =====================================================
    # TEST STATISTIQUE
    # =====================================================
    st.subheader("Z-test sur la différence de proportions")

    count = np.array([success_male, success_female])
    nobs = np.array([n_male, n_female])

    z_stat, p_value = proportions_ztest(count, nobs)

    st.write(f"**Z-statistic** : **{z_stat:.3f}**")
    st.write(f"**p-value** : **{p_value:.4f}**")

    # =====================================================
    # DÉCISION STATISTIQUE
    # =====================================================
    st.subheader("Décision")

    if p_value < 0.05:
        st.error(
            """
             **Rejet de H₀**

            La différence observée entre les taux d’acceptation
            est statistiquement significative au seuil de 5 %.
            """
        )
    else:
        st.success(
            """
            **Impossible de rejeter H₀**

            Les données ne fournissent pas de preuve suffisante
            d’une différence significative entre les groupes.
            """
        )

    # =====================================================
    # INTERPRÉTATION
    # =====================================================
    st.info(
        """
        Ce test confirme (ou non) les résultats obtenus par
        l’intervalle de confiance bootstrap.
        Il fournit une **preuve statistique formelle**
        permettant d’évaluer l’existence d’un biais
        entre les groupes.
        """
    )

# =====================================================
# 5️⃣ COMPARAISON
# =====================================================
elif page == "Comparaison":

    st.header("5. Comparaison : Performance ML vs Analyse statistique")

    # =====================================================
    # OBJECTIF
    # =====================================================
    st.subheader("Objectif de la comparaison")

    st.write(
        """
        Cette section compare deux lectures complémentaires
        des résultats du modèle de régression logistique :
        l’évaluation **Machine Learning classique**, centrée sur
        la performance globale, et l’analyse **statistique**,
        centrée sur l’équité des décisions.
        """
    )

    # =====================================================
    # POINT DE VUE MACHINE LEARNING
    # =====================================================
    st.subheader("Point de vue Machine Learning")

    acc = accuracy_score(y_test, y_pred_lr)
    auc = roc_auc_score(y_test, y_proba_lr)

    ml_df = pd.DataFrame({
        "Métrique": ["Accuracy", "AUC"],
        "Valeur": [acc, auc]
    })

    st.dataframe(ml_df.style.format({"Valeur": "{:.3f}"}))

    st.write(
        """
        Du point de vue du Machine Learning,
        le modèle présente de **bonnes performances globales**.
        Ces métriques résument la qualité moyenne des prédictions,
        sans distinction entre les groupes.
        """
    )

    # =====================================================
    # POINT DE VUE STATISTIQUE
    # =====================================================
    st.subheader("Point de vue statistique (équité)")

    p_male = np.mean(y_pred_lr[male_mask])
    p_female = np.mean(y_pred_lr[female_mask])
    diff_hat = p_male - p_female

    stat_df = pd.DataFrame({
        "Indicateur": [
            "Taux d’acceptation (Male)",
            "Taux d’acceptation (Female)",
            "Différence (Male − Female)"
        ],
        "Valeur": [p_male, p_female, diff_hat]
    })

    st.dataframe(stat_df.style.format({"Valeur": "{:.3f}"}))

    st.write(
        """
        L’analyse statistique met en évidence une **disparité importante**
        entre les groupes, révélant un biais potentiel
        dans les décisions du modèle.
        """
    )

    # =====================================================
    # COMPARAISON SYNTHÉTIQUE
    # =====================================================
    st.subheader("Comparaison synthétique")

    comparison_df = pd.DataFrame({
        "Aspect analysé": [
            "Objectif",
            "Type de métriques",
            "Niveau d’analyse",
            "Information fournie"
        ],
        "Machine Learning": [
            "Performance prédictive",
            "Accuracy, AUC",
            "Global",
            "Qualité moyenne des prédictions"
        ],
        "Analyse statistique": [
            "Équité / biais",
            "Taux d’acceptation, IC, Z-test",
            "Par groupe",
            "Disparités entre populations"
        ]
    })

    st.dataframe(comparison_df)

    # =====================================================
    # MESSAGE CLÉ
    # =====================================================
    st.warning(
        """
        👉 Un modèle peut présenter de bonnes performances globales
        tout en produisant des décisions inéquitables.
        **La performance ne garantit pas l’équité.**
        """
    )

# =====================================================
# 6️⃣ CONCLUSION
# =====================================================
elif page == "Conclusion":

    st.header("6. Conclusion générale")

    # =====================================================
    # SYNTHÈSE DU PROJET
    # =====================================================
    st.subheader("Synthèse")

    st.write(
        """
        Ce projet avait pour objectif d’évaluer l’existence de biais
        dans un modèle de Machine Learning appliqué à la prédiction
        du revenu à partir du jeu de données *Adult Income*.
        """
    )

    st.write(
        """
        Une **régression logistique** a été utilisée comme modèle principal,
        en excluant explicitement la variable sensible *gender*,
        afin d’analyser l’apparition de biais indirects
        dus aux corrélations présentes dans les données.
        """
    )

    # =====================================================
    # PRINCIPAUX RÉSULTATS
    # =====================================================
    st.subheader("Principaux résultats")

    st.markdown(
        """
        - Le modèle présente une **bonne performance globale**
          en termes d’accuracy et d’AUC.
        - Malgré cela, une **disparité importante** est observée
          entre les taux d’acceptation des groupes *Male* et *Female*.
        - L’analyse statistique (bootstrap) et le test d’hypothèse
          confirment que cette différence est
          **statistiquement significative**.
        """
    )

    # =====================================================
    # INTERPRÉTATION
    # =====================================================
    st.subheader("Interprétation")

    st.write(
        """
        Ces résultats montrent que de bonnes performances prédictives
        ne garantissent pas l’équité des décisions du modèle.
        Même en l’absence de la variable sensible,
        le modèle peut exploiter des **corrélations indirectes**
        présentes dans les données.
        """
    )

    # =====================================================
    # MESSAGE CLÉ
    # =====================================================
    st.warning(
        """
        👉 **La performance d’un modèle ne doit jamais être évaluée
        indépendamment de son impact sur les différents groupes.**
        """
    )

    # =====================================================
    # LIMITES ET PERSPECTIVES
    # =====================================================
    st.subheader("Limites et perspectives")

    st.markdown(
        """
        - Cette étude se limite à une analyse **post-hoc** du biais.
        - Des méthodes de mitigation du biais pourraient être explorées :
          *pré-processing*, *in-processing* ou *post-processing*.
        - L’analyse pourrait être étendue à d’autres attributs sensibles
          (âge, origine, statut marital).
        - L’étude de métriques d’équité supplémentaires
          permettrait une évaluation plus complète.
        """
    )

    # =====================================================
    # CONCLUSION FINALE
    # =====================================================
    st.info(
        """
        En conclusion, ce projet met en évidence l’importance
        d’intégrer des analyses statistiques rigoureuses
        dans l’évaluation des modèles de Machine Learning,
        afin de concilier **performance**, **équité** et **responsabilité**.
        """
    )
