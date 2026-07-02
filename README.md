# vol_ml_fund — Recherche quantitative sur le trading de volatilité

Plateforme de recherche Python pour étudier la prévision de volatilité réalisée
par machine learning et sa transformation en signaux de trading backtestés,
autour de la **prime de risque de volatilité** (écart entre vol implicite et
vol réalisée future).

⚠️ **Avertissement** : ce projet est une base de recherche pédagogique. Ce n'est
ni un bot de trading, ni un conseil en investissement, et aucune performance
financière n'est promise ou attendue. Les ETP de volatilité utilisés comme
proxies (VIXY, SVXY) comportent des risques extrêmes documentés (cf.
« volmageddon », février 2018).

La théorie complète (estimateurs de volatilité, HAR, prime de risque de
variance, protocole walk-forward, construction du signal) et la justification
de chaque choix de conception sont dans **[docs/theory.tex](docs/theory.tex)**
(document rédigé en anglais).

## Objectif

1. Télécharger et nettoyer des données quotidiennes OHLC (SPY, QQQ, VIX,
   VIX3M, VIXY, SVXY).
2. Calculer des features de volatilité : vol réalisée close-to-close (5/20/60j),
   estimateurs Garman-Klass et Parkinson (high-low), semi-vol baissière,
   term structure du VIX, mémoire des chocs, drawdown, saisonnalité.
3. Prédire la vol réalisée future à 5 jours via la target
   `log(RV future / vol implicite)` — le résidu que le marché d'options ne
   price pas déjà.
4. Comparer HAR-RV, HAR-X, Ridge, Lasso, Random Forest, Gradient Boosting,
   quantiles (q10/q90), ensemble, et deux benchmarks naïfs, en protocole
   **walk-forward purgé** (RMSE/MAE/R², RMSE par année, tests Diebold-Mariano).
5. Générer un signal avec hystérésis, sizing proportionnel à la conviction et
   vol targeting.
6. Backtester sur deux jambes tradables (VIXY long vol / SVXY short vol) avec
   coûts de transaction.
7. Produire un rapport : equity curve, Sharpe, max drawdown, CAGR.

## Architecture

```
vol_ml_fund/
├── config/config.yaml        # Configuration centrale (tickers, fenêtres, seuils, coûts…)
├── docs/theory.tex           # Document théorique (LaTeX, en anglais)
├── data/{raw,processed,external}/
├── notebooks/                # Exploration interactive
├── src/
│   ├── data/                 # download.py (yfinance OHLC), preprocess.py
│   ├── features/             # volatility_features.py (estimateurs, target, transforms)
│   ├── models/               # baseline.py (HAR, HAR-X), ml_model.py (builders),
│   │                         # walkforward.py (protocole purgé), evaluation.py (DM…)
│   ├── signals/              # volatility_signal.py (score, hystérésis, sizing, vol target)
│   ├── backtest/             # engine.py (2 jambes + coûts), metrics.py
│   └── utils/                # io.py (config YAML, IO CSV)
├── scripts/                  # Pipeline exécutable en 4 étapes
├── tests/                    # Tests pytest anti look-ahead
└── app.py                    # Dashboard Streamlit (analyse de sensibilité)
```

Principes de conception :

- **Pas de look-ahead bias** : features à *t* ⇒ données ≤ *t* ; target à *t* ⇒
  vol réalisée sur *t+1…t+5* ; walk-forward avec **purge** de 5 jours entre le
  train et les prédictions (chevauchement de target) ; le backtest décale les
  poids d'un jour. Ces trois garde-fous sont **testés** dans `tests/`.
- **Walk-forward** : fenêtre glissante de ~5 ans, ré-entraînement trimestriel —
  pas de split unique figé.
- **Évaluation en espace RV** : toutes les prédictions sont reconverties en
  niveau de vol annualisée pour comparer modèles, transforms et benchmarks
  naïfs sur la même échelle.
- **Coûts de transaction** en bps sur chaque changement de poids, par jambe.
- **Configuration unique** dans `config/config.yaml`, multi-actifs par
  construction.

## Installation

Prérequis : Python ≥ 3.10.

```bash
cd vol_ml_fund
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer le pipeline

Les quatre étapes s'exécutent depuis la racine du projet, dans l'ordre :

```bash
python scripts/run_download.py    # 1. OHLC quotidiens + log returns
python scripts/run_features.py    # 2. Features de vol + target log(RV/IV)
python scripts/run_train.py       # 3. Walk-forward : 8 modèles + benchmarks + DM
python scripts/run_backtest.py    # 4. Signal, backtest VIXY/SVXY, rapport
```

Étape optionnelle mais recommandée — le test « le ML gagne-t-il sa place ? » :

```bash
python scripts/run_strategy_benchmarks.py   # 5. Stratégie ML vs benchmarks sans modèle
```

Ce script compare, à moteur/coûts/vol targeting strictement identiques, la
stratégie ML à trois références sans apprentissage : short vol constant (carry
pur), règle de contango (short vol si VIX/VIX3M < 1), et carry avec
kill-switch ML (carry coupé quand le modèle prédit RV > IV). Il produit une
table de métriques, les rendements sur les fenêtres de stress (volmageddon
2018, covid 2020…) et les equity curves comparées.

Tests :

```bash
python -m pytest tests/ -q
```

## Dashboard interactif

Une fois le pipeline exécuté (au moins jusqu'à `run_train.py`) :

```bash
streamlit run app.py
```

Le dashboard est un **outil d'analyse de sensibilité** : il relit les
prédictions walk-forward figées et recalcule le signal et le backtest à la
volée via les mêmes modules `src/` que les scripts (aucune logique dupliquée).
Trois onglets :

- **Backtest interactif** — sliders sur les seuils d'entrée/sortie, le sizing,
  la vol target et les coûts ; equity curve, drawdown et position recalculés
  en direct ; heatmap du Sharpe sur une grille entrée×sortie pour détecter
  l'overfitting de seuils (un plateau est rassurant, un pic isolé non).
- **Comparaison des modèles** — métriques out-of-sample, RMSE par année,
  tests de Diebold-Mariano, prédictions vs réalisé zoomables par période.
- **État du marché** — dernier score, position hypothétique du jour, RV
  prédite vs vol implicite sur les 250 derniers jours.

Il ne lance ni téléchargement ni entraînement, et n'exécute aucun ordre.

Sorties principales :

- `data/processed/predictions_rv.csv` — prédictions out-of-sample (espace RV)
  de tous les modèles + quantiles + benchmarks ;
- `data/processed/report/equity_curve.png` — equity curve, position nette, score ;
- `data/processed/report/performance_summary.csv` — Sharpe, max drawdown, CAGR…

## Le signal (v2)

1. **Score** : `log(RV prédite / vol implicite)` — positif ⇒ la vol est
   sous-évaluée par le marché d'options ⇒ long vol ; négatif ⇒ short vol.
2. **Hystérésis** : entrée quand |score| > `entry_threshold`, sortie quand
   |score| < `exit_threshold` (réduit fortement le turnover).
3. **Sizing** : taille ∝ min(|score| / `sizing_scale`, 1), réduite de moitié
   quand la bande de quantiles [q10, q90] chevauche zéro.
4. **Vol targeting** : levier ajusté pour viser `vol_target` annualisée sur la
   jambe tradée, plafonné à `max_leverage`.

## Prochaines améliorations possibles

- Futures VIX (term structure complète, roll explicite) à la place des ETP.
- Sélection d'hyperparamètres par validation croisée temporelle *dans* chaque
  fenêtre walk-forward (nested CV).
- Options réelles : straddles delta-hedgés, surfaces de vol, skew.
- Cible multi-horizon (1j, 5j, 20j) et modèles multi-actifs simultanés.
- Régimes de marché (HMM) conditionnant le sizing.
- Intégration continue (GitHub Actions) exécutant les tests.

## Stack

pandas · numpy · scikit-learn · yfinance · matplotlib · pyyaml · joblib · pytest · streamlit
