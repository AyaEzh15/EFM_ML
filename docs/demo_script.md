# Script de démonstration - Projet MLOps IDS

## Projet

**Titre du projet :** Pipeline MLOps pour la détection d’intrusions réseau
**Cas d’usage :** Cybersécurité / IDS intelligent
**Dataset utilisé :** NSL-KDD
**Modèle retenu :** Random Forest
**API :** FastAPI
**Monitoring :** Evidently AI et Prometheus
**CI/CD :** GitHub Actions

---

## 1. Objectif de la démonstration

L’objectif de cette démonstration est de présenter un pipeline MLOps complet appliqué à la cybersécurité.

Le système permet de :

* Charger automatiquement un dataset de détection d’intrusions réseau.
* Préparer et nettoyer les données.
* Entraîner plusieurs modèles de Machine Learning.
* Comparer les performances des modèles avec MLflow.
* Sauvegarder le meilleur modèle.
* Déployer le modèle via une API FastAPI.
* Tester l’API avec Swagger UI.
* Vérifier le projet avec pytest.
* Automatiser les tests avec GitHub Actions.
* Surveiller les données avec Evidently AI.
* Exposer des métriques techniques avec Prometheus.

---

## 2. Prétraitement des données

### Commande à exécuter

```bash
python src/preprocessing.py
```

### Explication

Cette étape permet de charger automatiquement le dataset NSL-KDD depuis GitHub.

Le script réalise les traitements suivants :

* Chargement du dataset d’entraînement.
* Chargement du dataset de test.
* Suppression des doublons.
* Gestion des valeurs manquantes.
* Transformation du label en classification binaire :

  * `normal` devient `0`
  * toute attaque devient `1`
* Encodage des variables catégorielles.
* Sauvegarde des données préparées.
* Sauvegarde du scaler.
* Sauvegarde de la liste des colonnes utilisées.

### Fichiers générés

```text
data/raw/KDDTrain.csv
data/raw/KDDTest.csv
data/processed/train_processed.csv
data/processed/test_processed.csv
models/scaler.pkl
models/columns.pkl
```

### Résultat attendu

```text
Train chargé : (125973, 43)
Test chargé : (22543, 43)

Distribution train :
0    67343
1    58630

Distribution test :
1    12833
0     9710

Prétraitement terminé avec succès.
```

---

## 3. Entraînement des modèles Machine Learning

### Commande à exécuter

```bash
python src/train.py
```

### Explication

Cette étape entraîne plusieurs modèles de Machine Learning supervisé pour détecter les intrusions réseau.

Les modèles utilisés sont :

* Logistic Regression
* Random Forest

Les métriques calculées sont :

* Accuracy
* Precision
* Recall
* F1-score

Les expériences sont enregistrées automatiquement avec MLflow.

### Résultats obtenus

| Modèle              | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   0.7535 |    0.9173 | 0.6232 |   0.7422 |
| Random Forest       |   0.7648 |    0.9671 | 0.6075 |   0.7462 |

### Meilleur modèle

Le meilleur modèle obtenu est :

```text
Random Forest
```

Il est sauvegardé dans :

```text
models/best_model.pkl
```

### Fichiers générés

```text
models/best_model.pkl
models/results.csv
```

---

## 4. Suivi des expériences avec MLflow

### Commande à exécuter

```bash
mlflow ui
```

### Lien à ouvrir

```text
http://127.0.0.1:5000
```

### Explication

MLflow permet de suivre les expériences réalisées pendant l’entraînement des modèles.

Pour chaque modèle, MLflow enregistre :

* Le nom du modèle.
* Les paramètres.
* L’accuracy.
* La precision.
* Le recall.
* Le F1-score.
* Le modèle entraîné.

### Ce qu’il faut montrer pendant la démonstration

Dans l’interface MLflow, montrer :

* L’expérience `MLOps_IDS_Cybersecurity`.
* Les différents runs.
* Les métriques des modèles.
* La comparaison entre Logistic Regression et Random Forest.

---

## 5. Évaluation du meilleur modèle

### Commande à exécuter

```bash
python src/evaluate.py
```

### Explication

Cette étape permet d’évaluer le meilleur modèle sauvegardé sur les données de test.

Le script génère :

* Les métriques finales.
* Le classification report.
* La matrice de confusion.
* Les fichiers CSV des résultats.

### Résultats obtenus avec Random Forest

```text
Accuracy  : 0.7648
Precision : 0.9671
Recall    : 0.6075
F1-score  : 0.7462
```

### Interprétation

Le modèle Random Forest présente une bonne précision pour la classe Attack. Cela signifie que lorsqu’il prédit une attaque, la prédiction est généralement fiable.

Cependant, le recall reste plus faible. Cela signifie que certaines attaques ne sont pas détectées et sont classées comme trafic normal. Dans un contexte cybersécurité, ce point est important car les faux négatifs représentent des attaques non détectées.

### Fichiers générés

```text
reports/figures/confusion_matrix.png
reports/evaluation_metrics.csv
reports/classification_report.csv
reports/confusion_matrix.csv
```

---

## 6. Lancement de l’API FastAPI

### Commande à exécuter

```bash
uvicorn api.main:app --reload
```

### Lien principal

```text
http://127.0.0.1:8000
```

### Lien Swagger UI

```text
http://127.0.0.1:8000/docs
```

### Explication

L’API FastAPI permet de déployer le modèle entraîné sous forme de service web.

Elle contient les routes suivantes :

```text
GET /
GET /health
POST /predict
GET /metrics
```

---

## 7. Test de la route principale

### Route

```text
GET /
```

### Lien

```text
http://127.0.0.1:8000
```

### Résultat attendu

```json
{
  "message": "API IDS MLOps fonctionne correctement",
  "status": "OK"
}
```

### Explication

Cette route permet simplement de vérifier que l’API est bien lancée.

---

## 8. Test de la route health

### Route

```text
GET /health
```

### Lien

```text
http://127.0.0.1:8000/health
```

### Résultat attendu

```json
{
  "status": "healthy",
  "model": "Random Forest",
  "task": "Network Intrusion Detection"
}
```

### Explication

Cette route permet de vérifier l’état du service et du modèle utilisé.

---

## 9. Test de prédiction avec Swagger

### Route

```text
POST /predict
```

### Lien Swagger

```text
http://127.0.0.1:8000/docs
```

### Étapes

1. Ouvrir Swagger UI.
2. Aller vers la route `POST /predict`.
3. Cliquer sur `Try it out`.
4. Coller le JSON d’exemple.
5. Cliquer sur `Execute`.

### Exemple de données à envoyer

```json
{
  "features": {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 181,
    "dst_bytes": 5450,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 8,
    "srv_count": 8,
    "serror_rate": 0,
    "srv_serror_rate": 0,
    "rerror_rate": 0,
    "srv_rerror_rate": 0,
    "same_srv_rate": 1,
    "diff_srv_rate": 0,
    "srv_diff_host_rate": 0,
    "dst_host_count": 9,
    "dst_host_srv_count": 9,
    "dst_host_same_srv_rate": 1,
    "dst_host_diff_srv_rate": 0,
    "dst_host_same_src_port_rate": 0.11,
    "dst_host_srv_diff_host_rate": 0,
    "dst_host_serror_rate": 0,
    "dst_host_srv_serror_rate": 0,
    "dst_host_rerror_rate": 0,
    "dst_host_srv_rerror_rate": 0
  }
}
```

### Exemple de réponse possible

```json
{
  "prediction": 0,
  "label": "Normal",
  "interpretation": "Le trafic réseau est considéré comme normal."
}
```

ou :

```json
{
  "prediction": 1,
  "label": "Attack",
  "interpretation": "Le trafic réseau est considéré comme une attaque potentielle."
}
```

### Remarque importante

Si on ouvre directement `/predict` dans le navigateur, on obtient :

```json
{
  "detail": "Method Not Allowed"
}
```

Ce n’est pas une erreur du projet. Cela arrive parce que `/predict` est une route POST. Elle doit être testée avec Swagger, Postman ou une requête HTTP POST.

---

## 10. Tests automatiques avec pytest

### Commande à exécuter

```bash
pytest
```

### Explication

Les tests automatiques vérifient :

* Le bon fonctionnement de la route `/`.
* Le bon fonctionnement de la route `/health`.
* Le bon fonctionnement de la route `/predict`.
* L’existence du modèle entraîné.
* L’existence du scaler.
* L’existence du fichier des colonnes.
* Le chargement correct du modèle.

### Résultat obtenu

```text
7 passed
```

### Interprétation

Le résultat `7 passed` signifie que les tests unitaires ont été exécutés avec succès. Cela valide le fonctionnement principal de l’API et la présence des artefacts nécessaires au modèle.

---

## 11. CI/CD avec GitHub Actions

### Fichier utilisé

```text
.github/workflows/ci.yml
```

### Explication

GitHub Actions permet d’automatiser la validation du projet.

À chaque push sur GitHub, le workflow :

* Récupère le code.
* Installe Python.
* Installe les dépendances.
* Vérifie la structure du projet.
* Lance les tests avec pytest.

### Résultat obtenu

```text
Status: Success
```

### Interprétation

Le pipeline CI/CD a été exécuté avec succès. Cela signifie que le projet peut être validé automatiquement après chaque modification envoyée sur GitHub.

---

## 12. Monitoring avec Evidently AI

### Commande à exécuter

```bash
python monitoring/evidently_report.py
```

### Fichier généré

```text
monitoring/reports/evidently_data_drift_report.html
```

### Explication

Evidently AI est utilisé pour analyser la dérive des données entre les données d’entraînement et les données de test.

Le monitoring permet de vérifier si les données changent de distribution avec le temps.

### Résultat obtenu

```text
Dataset Drift is NOT detected
123 columns
17 drifted columns
Share of Drifted Columns = 0.138
```

### Interprétation

Le drift global du dataset n’est pas détecté, car la part des colonnes ayant dérivé reste inférieure au seuil de 0.5.

Cependant, 17 colonnes sur 123 présentent une dérive, soit environ 13.82 %. Cela montre que certaines variables réseau ont une distribution différente entre les données d’entraînement et les données de test.

Cette analyse est importante en MLOps, car elle permet d’anticiper une possible dégradation du modèle en production.

---

## 13. Monitoring avec Prometheus

### Lien à ouvrir après lancement de l’API

```text
http://127.0.0.1:8000/metrics
```

### Explication

La route `/metrics` expose des métriques compatibles avec Prometheus.

Elle permet de surveiller :

* Le nombre de requêtes HTTP.
* Les temps de réponse.
* Les statuts des requêtes.
* Les informations sur l’environnement Python.
* Le comportement technique de l’API.

### Interprétation

Ces métriques améliorent l’observabilité du système. Dans un environnement de production, Prometheus pourrait collecter ces métriques et Grafana pourrait les afficher sous forme de tableaux de bord.

---

## 14. Docker

### Fichier préparé

```text
Dockerfile
```

### Explication

Un Dockerfile a été préparé afin de permettre la containerisation de l’API FastAPI.

Docker permet de rendre l’application portable et exécutable dans différents environnements sans devoir réinstaller manuellement toutes les dépendances.

### Commandes prévues

```bash
docker build -t ids-mlops-api .
docker run -p 8000:8000 ids-mlops-api
```

### Remarque

Le test Docker pourra être effectué dans un environnement disposant de Docker Desktop.

---

## 15. Résumé de la démonstration

Pendant la soutenance, l’ordre recommandé est :

1. Présenter rapidement la structure du projet.
2. Montrer le prétraitement des données.
3. Montrer les résultats d’entraînement.
4. Montrer MLflow.
5. Montrer la matrice de confusion.
6. Lancer l’API FastAPI.
7. Tester `/`, `/health` et `/predict`.
8. Montrer les tests pytest.
9. Montrer GitHub Actions.
10. Montrer Evidently AI.
11. Montrer `/metrics`.
12. Conclure sur les limites et améliorations futures.

---

## 16. Conclusion de la démonstration

Ce projet montre la mise en place d’un pipeline MLOps complet pour un cas d’usage en cybersécurité.

Le système couvre toutes les étapes principales :

* Préparation des données.
* Entraînement des modèles.
* Suivi des expériences.
* Évaluation.
* Déploiement via API.
* Tests automatiques.
* CI/CD.
* Monitoring.

Le projet peut être amélioré par la suite avec :

* L’ajout d’autres modèles comme XGBoost ou SVM.
* L’optimisation des hyperparamètres.
* Le déploiement cloud.
* L’intégration complète avec Docker.
* L’ajout de Grafana pour visualiser les métriques Prometheus.
* La détection en temps réel à partir de logs réseau.
