# Interview guide

## Pourquoi prédire le churn ?

Le modèle sert à prioriser les clients qui méritent une intervention avant leur départ. Il ne remplace pas l’équipe de rétention et ne prouve pas qu’une action empêchera le churn.

## Pourquoi pas l’accuracy seule ?

Le churn est minoritaire (26,54 % dans ce fichier). Une accuracy élevée peut cacher un rappel très faible des churners. Le projet suit precision, recall, F1, ROC-AUC, PR-AUC, spécificité, balanced accuracy et Brier score.

## Pourquoi PR-AUC ?

La courbe precision-recall décrit mieux la qualité de la file de clients positifs dans un problème déséquilibré. ROC-AUC reste utile pour comparer le classement global, mais peut sembler favorable avec beaucoup de négatifs.

## Comment le déséquilibre est-il traité ?

Le benchmark compare la baseline logistique, la logistique équilibrée, Random Forest équilibré et XGBoost avec `scale_pos_weight`. SMOTE n’est pas retenu dans ce run. S’il était utilisé, il devrait être placé dans une pipeline d’entraînement et appliqué uniquement aux folds de train.

## Pourquoi Logistic Regression comme baseline ?

Elle est rapide, interprétable et fournit un point de comparaison solide. Ici, la version équilibrée obtient le meilleur PR-AUC validation et un recall supérieur à la version non pondérée.

## Pourquoi XGBoost n’a-t-il pas été choisi ?

Parce qu’un modèle plus complexe ne gagne pas automatiquement. Dans le run réel, son PR-AUC validation est inférieur à celui de la logistique équilibrée. La sélection est fondée sur les mesures exécutées, pas sur le prestige de l’algorithme.

## Pourquoi le seuil n’est-il pas 0,5 ?

Le seuil traduit le compromis entre churners atteints, faux positifs et charge de l’équipe. La simulation configurable retient 0,35 avec les hypothèses actuelles et une capacité de 900 contacts.

## Qu’est-ce que la calibration ?

Une probabilité calibrée doit correspondre approximativement à une fréquence observée. Le projet compare le Brier score brut et calibré, puis utilise la probabilité calibrée pour les segments et le seuil.

## Que montre SHAP ?

SHAP décompose la contribution des variables à une prédiction par rapport à une référence. Les graphiques globaux et les explications locales sont des outils de transparence, pas des preuves causales.

## Comment analyser les faux négatifs ?

`results/reports/error_analysis.md` compte les churners passés sous le seuil. Il faut ensuite comparer tenure, contrat, paiement, services et charges, puis décider si la capacité de contact ou les features doivent changer.

## Comment déployer et monitorer ?

Le pipeline complet est persisté avec joblib ; Flask expose `/predict` et `/predict/batch`. Le monitoring hors ligne suit PSI/KS pour les features et prévoit le suivi de performance une fois les labels futurs disponibles. Le drift ne doit pas écraser automatiquement le modèle : il déclenche une revue/reformation contrôlée.

