"""
Système de Recommandation Distribué - Moteur ALS
================================================
Technologies : Apache Spark + Hadoop HDFS + Python
Dataset      : MovieLens (ml-25m ou ml-latest-small)
Algorithme   : ALS (Alternating Least Squares) - Filtrage collaboratif
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, avg, count
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
import pyspark.sql.functions as F
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Moteur de recommandation distribué basé sur ALS (Alternating Least Squares).
    Similaire au système de recommandation de Netflix.
    
    ALS décompose la matrice utilisateur-item R en deux matrices de rang inférieur :
        R ≈ U × V^T
    où U = matrice des facteurs utilisateurs, V = matrice des facteurs items.
    """

    def __init__(self, hdfs_path: str = "hdfs://localhost:9000", app_name: str = "MovieRecommender"):
        self.hdfs_path = hdfs_path
        self.spark = self._create_spark_session(app_name)
        self.model = None
        self.ratings_df = None
        self.movies_df = None

    def _create_spark_session(self, app_name: str) -> SparkSession:
        """Initialise la session Spark avec configuration HDFS et optimisations."""
        return (
            SparkSession.builder
            .appName(app_name)
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.default.parallelism", "100")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            # Connexion HDFS
            .config("spark.hadoop.fs.defaultFS", self.hdfs_path)
            .config("spark.hadoop.dfs.replication", "3")
            # Optimisation ALS
            .config("spark.ml.recommendation.als.numUserBlocks", "10")
            .config("spark.ml.recommendation.als.numItemBlocks", "10")
            .getOrCreate()
        )

    # ─────────────────────────────────────────────
    # 1. CHARGEMENT DES DONNÉES
    # ─────────────────────────────────────────────

    def load_movielens_data(self, data_path: str) -> tuple:
        """
        Charge le dataset MovieLens depuis HDFS ou le système de fichiers local.
        
        Structure MovieLens :
          ratings.csv → userId, movieId, rating, timestamp
          movies.csv  → movieId, title, genres
          tags.csv    → userId, movieId, tag, timestamp
        """
        logger.info(f"Chargement des données depuis : {data_path}")

        # Schéma explicite pour performances optimales
        ratings_schema = StructType([
            StructField("userId",    IntegerType(), True),
            StructField("movieId",   IntegerType(), True),
            StructField("rating",    FloatType(),   True),
            StructField("timestamp", IntegerType(), True),
        ])

        movies_schema = StructType([
            StructField("movieId", IntegerType(), True),
            StructField("title",   StringType(),  True),
            StructField("genres",  StringType(),  True),
        ])

        self.ratings_df = (
            self.spark.read
            .option("header", "true")
            .schema(ratings_schema)
            .csv(f"{data_path}/ratings.csv")
            .cache()   # persist en mémoire pour réutilisation
        )

        self.movies_df = (
            self.spark.read
            .option("header", "true")
            .schema(movies_schema)
            .csv(f"{data_path}/movies.csv")
        )

        # Statistiques du dataset
        n_ratings = self.ratings_df.count()
        n_users   = self.ratings_df.select("userId").distinct().count()
        n_movies  = self.ratings_df.select("movieId").distinct().count()
        avg_rating = self.ratings_df.select(avg("rating")).collect()[0][0]

        logger.info(f"Dataset chargé :")
        logger.info(f"  Évaluations : {n_ratings:,}")
        logger.info(f"  Utilisateurs: {n_users:,}")
        logger.info(f"  Films       : {n_movies:,}")
        logger.info(f"  Note moyenne: {avg_rating:.2f}")

        return self.ratings_df, self.movies_df

    # ─────────────────────────────────────────────
    # 2. PRÉTRAITEMENT
    # ─────────────────────────────────────────────

    def preprocess(self) -> tuple:
        """
        Nettoie et prépare les données pour ALS.
        - Suppression des doublons
        - Filtrage des utilisateurs/films avec trop peu d'évaluations
        - Normalisation optionnelle
        """
        logger.info("Prétraitement des données...")

        # Supprimer les doublons (garder la dernière évaluation)
        df = (
            self.ratings_df
            .dropDuplicates(["userId", "movieId"])
            .na.drop()
        )

        # Filtrer les utilisateurs avec au moins 20 évaluations
        user_counts = df.groupBy("userId").agg(count("rating").alias("n"))
        active_users = user_counts.filter(col("n") >= 20).select("userId")
        df = df.join(active_users, "userId")

        # Filtrer les films avec au moins 10 évaluations
        movie_counts = df.groupBy("movieId").agg(count("rating").alias("n"))
        popular_movies = movie_counts.filter(col("n") >= 10).select("movieId")
        df = df.join(popular_movies, "movieId")

        logger.info(f"Après prétraitement : {df.count():,} évaluations")

        # Division train/test (80/20)
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        train_df = train_df.cache()
        test_df  = test_df.cache()

        logger.info(f"Entraînement : {train_df.count():,} | Test : {test_df.count():,}")
        return train_df, test_df

    # ─────────────────────────────────────────────
    # 3. ENTRAÎNEMENT DU MODÈLE ALS
    # ─────────────────────────────────────────────

    def train(self, train_df, rank: int = 50, max_iter: int = 10,
              reg_param: float = 0.1, alpha: float = 1.0) -> ALSModel:
        """
        Entraîne le modèle ALS (Alternating Least Squares).
        
        Paramètres clés :
          rank      → dimension des facteurs latents (complexité du modèle)
          max_iter  → nombre d'itérations alternées
          reg_param → coefficient de régularisation (évite le surapprentissage)
          alpha     → confiance pour les données implicites
        
        Mathématiquement, ALS minimise :
          ∑_{(u,i)∈Ω} (r_ui - u_u · v_i^T)² + λ(‖u_u‖² + ‖v_i‖²)
        """
        logger.info(f"Entraînement ALS : rank={rank}, iter={max_iter}, reg={reg_param}")

        als = ALS(
            rank=rank,
            maxIter=max_iter,
            regParam=reg_param,
            alpha=alpha,
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            implicitPrefs=False,          # Explicite (notes 0.5–5)
            coldStartStrategy="drop",     # Gestion nouveaux utilisateurs/films
            nonnegative=False,
            numUserBlocks=10,             # Parallélisme distribué
            numItemBlocks=10,
        )

        self.model = als.fit(train_df)
        logger.info("Modèle ALS entraîné avec succès")
        return self.model

    def tune_hyperparameters(self, train_df, test_df) -> ALSModel:
        """
        Optimisation des hyperparamètres par validation croisée.
        Explore une grille de rank × regParam.
        """
        logger.info("Recherche des meilleurs hyperparamètres...")

        als = ALS(
            userCol="userId", itemCol="movieId", ratingCol="rating",
            coldStartStrategy="drop", nonnegative=False,
        )

        # Grille de paramètres
        param_grid = (
            ParamGridBuilder()
            .addGrid(als.rank,      [10, 50, 100])
            .addGrid(als.regParam,  [0.01, 0.1, 1.0])
            .addGrid(als.maxIter,   [10, 20])
            .build()
        )

        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction",
        )

        cv = CrossValidator(
            estimator=als,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
            parallelism=4,   # cross-validation en parallèle
        )

        cv_model = cv.fit(train_df)
        self.model = cv_model.bestModel

        best_rank  = self.model.rank
        best_reg   = self.model._java_obj.parent().getRegParam()
        test_rmse  = evaluator.evaluate(self.model.transform(test_df))

        logger.info(f"Meilleurs paramètres → rank={best_rank}, regParam={best_reg:.3f}")
        logger.info(f"RMSE sur test       → {test_rmse:.4f}")
        return self.model

    # ─────────────────────────────────────────────
    # 4. ÉVALUATION
    # ─────────────────────────────────────────────

    def evaluate(self, test_df) -> dict:
        """
        Évalue le modèle sur l'ensemble de test.
        Métriques : RMSE, MAE
        """
        predictions = self.model.transform(test_df).na.drop()
        evaluator = RegressionEvaluator(labelCol="rating", predictionCol="prediction")

        rmse = evaluator.setMetricName("rmse").evaluate(predictions)
        mae  = evaluator.setMetricName("mae").evaluate(predictions)

        metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}
        logger.info(f"Évaluation : RMSE={rmse:.4f} | MAE={mae:.4f}")
        return metrics

    # ─────────────────────────────────────────────
    # 5. GÉNÉRATION DE RECOMMANDATIONS
    # ─────────────────────────────────────────────

    def recommend_for_user(self, user_id: int, n: int = 10):
        """
        Génère les N meilleures recommandations pour un utilisateur donné.
        Filtre les films déjà vus par l'utilisateur.
        """
        assert self.model is not None, "Entraîner le modèle d'abord"

        # Films déjà évalués par l'utilisateur
        seen_movies = (
            self.ratings_df
            .filter(col("userId") == user_id)
            .select("movieId")
        )

        # Tous les films − films vus
        all_movies = self.movies_df.select("movieId")
        unseen_movies = all_movies.join(seen_movies, "movieId", "left_anti")

        # Créer les paires utilisateur-film non vus
        user_df = self.spark.createDataFrame([(user_id,)], ["userId"])
        user_unseen = user_df.crossJoin(unseen_movies)

        # Prédire les notes
        predictions = (
            self.model.transform(user_unseen)
            .na.drop()
            .orderBy(col("prediction").desc())
            .limit(n)
            .join(self.movies_df, "movieId")
            .select("movieId", "title", "genres", "prediction")
        )

        return predictions

    def recommend_for_all_users(self, n: int = 10):
        """
        Génère les recommandations pour tous les utilisateurs en une passe.
        Utilise la méthode optimisée recommendForAllUsers de Spark.
        """
        assert self.model is not None, "Entraîner le modèle d'abord"
        recs = self.model.recommendForAllUsers(n)
        # Exploser les recommandations en lignes individuelles
        recs_exploded = (
            recs
            .withColumn("rec", explode("recommendations"))
            .select("userId", col("rec.movieId"), col("rec.rating").alias("predicted_rating"))
            .join(self.movies_df, "movieId")
        )
        return recs_exploded

    def find_similar_movies(self, movie_id: int, n: int = 10):
        """
        Trouve les films les plus similaires à un film donné.
        Utilise la similarité cosinus entre les vecteurs de facteurs items.
        
        Similarité cosinus : sim(a, b) = (a · b) / (‖a‖ × ‖b‖)
        """
        assert self.model is not None, "Entraîner le modèle d'abord"

        # Facteurs latents de tous les items
        item_factors = self.model.itemFactors

        # Vecteur du film cible
        target_vector = (
            item_factors
            .filter(col("id") == movie_id)
            .select("features")
            .collect()
        )
        if not target_vector:
            logger.warning(f"Film {movie_id} non trouvé dans le modèle")
            return None

        target_vec = target_vector[0]["features"]

        # Similarité cosinus via UDF
        from pyspark.sql.functions import udf
        from pyspark.sql.types import DoubleType
        import math

        def cosine_similarity(v1, v2):
            dot   = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)

        target_broadcast = self.spark.sparkContext.broadcast(target_vec)
        cosine_udf = udf(lambda v: cosine_similarity(v, target_broadcast.value), DoubleType())

        similar = (
            item_factors
            .filter(col("id") != movie_id)
            .withColumn("similarity", cosine_udf(col("features")))
            .orderBy(col("similarity").desc())
            .limit(n)
            .withColumnRenamed("id", "movieId")
            .join(self.movies_df, "movieId")
            .select("movieId", "title", "genres", "similarity")
        )
        return similar

    # ─────────────────────────────────────────────
    # 6. PERSISTANCE (HDFS)
    # ─────────────────────────────────────────────

    def save_model(self, path: str = None):
        """Sauvegarde le modèle entraîné sur HDFS."""
        assert self.model is not None
        save_path = path or f"{self.hdfs_path}/models/als_recommender"
        self.model.write().overwrite().save(save_path)
        logger.info(f"Modèle sauvegardé → {save_path}")

    def load_model(self, path: str = None):
        """Charge un modèle ALS précédemment sauvegardé depuis HDFS."""
        load_path = path or f"{self.hdfs_path}/models/als_recommender"
        self.model = ALSModel.load(load_path)
        logger.info(f"Modèle chargé depuis {load_path}")
        return self.model

    def stop(self):
        """Arrête la session Spark proprement."""
        if self.spark:
            self.spark.stop()
            logger.info("Session Spark terminée")
