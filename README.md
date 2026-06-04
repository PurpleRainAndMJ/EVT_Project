# Le risque cyber comme cas limite de dépendance multivariée

Projet — *Extrema et théorie des valeurs extrêmes* (M2 GRAF, ISFA).

**Thèse.** Dans un modèle à choc commun `X_i = Z_i + I·B_i·S` (inspiré de
Zeller & Scherer, *EAJ* 2022), le taux de contagion `p` est exactement le
coefficient de dépendance de queue (`χ = p`), la copule limite est de type
Marshall–Olkin (`A(w) = 1 − p·min(w, 1−w)`) et le bénéfice de diversification
en TVaR plafonne dès que `p > 0` — l'anti-diversification, qui distingue
structurellement le cyber du nat-cat.

## Contenu

| Fichier | Rôle |
|---|---|
| `projet_cyber_extremes.pdf` / `.tex` | Document principal (~7 pages) |
| `cyber_extremes.py` | Modèle, estimateurs (χ, χ̄, Pickands CFG, mesure spectrale), TVaR/DR |
| `make_figures.py` | Reproduit les 5 figures du document (`figures/`) |
| `app.py` | Application Streamlit interactive |

## Installation et reproduction

```bash
pip install numpy scipy matplotlib streamlit
python make_figures.py        # régénère figures/fig1..fig5 (seed = 42)
streamlit run app.py          # application interactive
```

## L'app en deux gestes

1. Préréglage **Nat-cat multi-régions** (`p = 0`) : masse spectrale aux coins,
   `χ(u) ≈ 0`, DR décroît avec la taille du portefeuille.
2. Préréglage **Cyber** (`p = 0.3`) puis curseur `p` vers 0.6 : un atome
   apparaît au centre du simplexe, `χ̂(0.99) → p`, et la courbe DR(d) plafonne.

Astuce : passer `α_s ≥ α_z` bascule dans le régime d'*indépendance
asymptotique* (`χ → 0`, `χ̄ < 1`) — le diagnostic (χ, χ̄) du cours fait la
différence entre un cyber accumulatif et un cyber diversifiable.

## Référence

Zeller, G., Scherer, M. (2022). *A comprehensive model for cyber risk based on
marked point processes and its application to insurance.* European Actuarial
Journal, 12, 33–85.
