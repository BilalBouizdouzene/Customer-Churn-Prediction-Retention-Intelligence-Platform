# Captures d’écran du projet

Les captures utilisées dans le README sont conservées directement dans ce dossier :

```text
docs/screenshots/
```

## Captures présentes

| Nom de fichier | Contenu |
|---|---|
| `01-overview.png` | Page d’accueil du dashboard avec les indicateurs principaux |
| `02-customer-risk.png` | Résultat d’un scoring individuel depuis `/predict` |
| `03-shap-explanation.png` | Graphique SHAP global et facteurs locaux |
| `04-model-performance.png` | Benchmark des modèles, calibration et seuil |
| `05-retention-queue.png` | File de clients HIGH et CRITICAL |

Ces cinq images sont déjà intégrées dans la section « Démonstration de l’application » du README principal.

## Comment les obtenir

1. Démarrer l’application :

   ```powershell
   $env:PYTHONPATH = "src"
   .venv\Scripts\python.exe -m churn_intelligence.api.app
   ```

2. Ouvrir http://127.0.0.1:5000 et capturer la page d’accueil.
3. Tester `/predict` avec l’exemple JSON du README et capturer la réponse.
4. Utiliser les fichiers suivants pour les vues analytiques :

   ```text
   results/explainability/shap_summary.png
   results/figures/calibration_curve.png
   results/figures/threshold_business_value.png
   results/metrics/model_benchmark.csv
   results/scores/customer_churn_scores.csv
   ```

5. Enregistrer les nouvelles images sous les noms indiqués ci-dessus si elles doivent être remplacées.

## Insérer les images dans le README

Les chemins relatifs utilisés dans le README sont :

```markdown
![Vue générale](docs/screenshots/01-overview.png)
![Risque client](docs/screenshots/02-customer-risk.png)
![Explication SHAP](docs/screenshots/03-shap-explanation.png)
```

Une capture supplémentaire de la documentation API peut être ajoutée plus tard sous le nom `06-api-docs.png`.
