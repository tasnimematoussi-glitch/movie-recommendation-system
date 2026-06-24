# Système de Recommandation Distribué
## Style Netflix — MovieLens + Hadoop + Spark + Python

---

## Architecture

```
MovieLens CSV  →  HDFS (Hadoop)  →  Spark ALS  →  Recommandations  →  API Flask
     ↑                 ↑                ↑                  ↑               ↑
   Dataset       Stockage dist.   Entraînement       Prédictions      Endpoint HTTP
```

## Technologies

| Composant | Rôle |
|-----------|------|
| **Hadoop HDFS** | Stockage distribué (3 DataNodes, réplication ×3) |
| **Apache Spark** | Traitement distribué + MLlib ALS |
| **Python** | Orchestration, API REST Flask |
| **MovieLens** | Dataset (100K → 25M évaluations) |

## Algorithme : ALS (Alternating Least Squares)

ALS décompose la matrice utilisateur-item R de dimension (U × I) en deux matrices de rang k :

```
R ≈ U × V^T       (U : facteurs utilisateurs, V : facteurs items)
```

**Objectif** : minimiser l'erreur de reconstruction :

```
min Σ (r_ui - u_u · v_i^T)² + λ(‖u_u‖² + ‖v_i‖²)
```

**Alternance** : fixer V pour optimiser U, puis fixer U pour optimiser V, répéter.

## Installation et démarrage

### Prérequis
- Docker Desktop ou Docker Engine + Compose
- Python 3.10+
- 16 GB RAM recommandés (cluster Hadoop + Spark)

### 1. Cloner et préparer

```bash
git clone <repo>
cd recommendation_system
pip install -r requirements.txt
```

### 2. Télécharger MovieLens

```bash
# Version rapide (100K évaluations)
python scripts/download_dataset.py --version small --dest /tmp/movielens

# Version production (25M évaluations)
python scripts/download_dataset.py --version 25m --dest /tmp/movielens
```

### 3. Démarrer le cluster

```bash
docker-compose up -d

# Vérifier que tous les services sont opérationnels
docker-compose ps
```

Interfaces web disponibles :
- **HDFS NameNode** → http://localhost:9870
- **Spark Master**  → http://localhost:8080
- **API REST**      → http://localhost:5000

### 4. Exécuter le pipeline complet

```bash
# Pipeline complet (ingestion → entraînement → recommandations)
python scripts/run_pipeline.py \
  --data-dir /tmp/movielens/ml-latest-small \
  --mode full \
  --rank 50 \
  --max-iter 10

# Entraînement uniquement
python scripts/run_pipeline.py --mode train --data-dir /tmp/movielens/ml-latest-small

# Recommandations pour un utilisateur (modèle déjà entraîné)
python scripts/run_pipeline.py --mode recommend --user-id 42 --top-n 10

# Films similaires
python scripts/run_pipeline.py --mode similar --movie-id 318 --top-n 10

# Optimisation automatique des hyperparamètres
python scripts/run_pipeline.py --mode tune --data-dir /tmp/movielens/ml-latest-small
```

### 5. Utiliser l'API REST

```bash
# Recommandations pour l'utilisateur 42
curl http://localhost:5000/api/recommend/42?n=10

# Films similaires à Toy Story (movieId=1)
curl http://localhost:5000/api/similar/1?n=10

# Informations sur un film
curl http://localhost:5000/api/movies/318

# Statistiques globales
curl http://localhost:5000/api/stats

# Health check
curl http://localhost:5000/health
```

## Paramètres ALS

| Paramètre | Valeur par défaut | Description |
|-----------|------------------|-------------|
| `rank` | 50 | Dimension des facteurs latents |
| `maxIter` | 10 | Nombre d'itérations alternées |
| `regParam` | 0.1 | Coefficient de régularisation λ |
| `alpha` | 1.0 | Confiance (données implicites) |
| `coldStartStrategy` | drop | Gestion nouveaux utilisateurs |

## Métriques d'évaluation

- **RMSE** (Root Mean Square Error) — objectif : < 0.90
- **MAE** (Mean Absolute Error)    — objectif : < 0.70

## Structure du projet

```
recommendation_system/
├── src/
│   ├── recommendation_engine.py   # Moteur ALS principal
│   └── hdfs_manager.py            # Interface HDFS
├── api/
│   └── app.py                     # API REST Flask
├── scripts/
│   ├── run_pipeline.py            # Orchestrateur principal
│   └── download_dataset.py        # Téléchargement MovieLens
├── docker-compose.yml             # Cluster Hadoop + Spark
├── requirements.txt
└── README.md
```

## Performances attendues

| Dataset | Entraînement | RMSE | RAM |
|---------|-------------|------|-----|
| Small (100K) | ~2 min | ~0.87 | 4 GB |
| 25M         | ~25 min | ~0.82 | 16 GB |
