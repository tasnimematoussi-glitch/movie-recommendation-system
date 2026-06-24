#!/usr/bin/env python3

import argparse
import logging
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, avg, count
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
import pyspark.sql.functions as F
import math

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ─────────────────────────────────────────────
# Session Spark locale (pas besoin de Hadoop)
# ─────────────────────────────────────────────

def create_spark():
    return (
        SparkSession.builder
        .appName("MovieRecommender-Local")
        .master("local[*]")                          # utilise tous les cœurs CPU
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8") # réduit pour mode local
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

# ─────────────────────────────────────────────
# Chargement des données (fichiers locaux)
# ─────────────────────────────────────────────

def load_data(spark, data_dir):
    logger.info(f"Chargement des données depuis : {data_dir}")

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

    ratings_path = os.path.join(data_dir, "ratings.csv")
    movies_path  = os.path.join(data_dir, "movies.csv")

    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"ratings.csv introuvable dans {data_dir}\n"
            f"Télécharge MovieLens sur : https://grouplens.org/datasets/movielens/latest/"
        )

    ratings_df = (
        spark.read.option("header", "true")
        .schema(ratings_schema)
        .csv(ratings_path)
        .cache()
    )
    movies_df = (
        spark.read.option("header", "true")
        .schema(movies_schema)
        .csv(movies_path)
    )

    n_ratings = ratings_df.count()
    n_users   = ratings_df.select("userId").distinct().count()
    n_movies  = ratings_df.select("movieId").distinct().count()
    avg_r     = ratings_df.select(avg("rating")).collect()[0][0]

    logger.info(f"  Évaluations : {n_ratings:,}")
    logger.info(f"  Utilisateurs: {n_users:,}")
    logger.info(f"  Films       : {n_movies:,}")
    logger.info(f"  Note moyenne: {avg_r:.2f} / 5.0")

    return ratings_df, movies_df

# ─────────────────────────────────────────────
# Prétraitement
# ─────────────────────────────────────────────

def preprocess(ratings_df):
    logger.info("Prétraitement...")
    df = ratings_df.dropDuplicates(["userId", "movieId"]).na.drop()

    # Garder les utilisateurs avec ≥ 5 évaluations (seuil bas pour le dataset small)
    user_counts = df.groupBy("userId").agg(count("rating").alias("n"))
    df = df.join(user_counts.filter(col("n") >= 5).select("userId"), "userId")

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_df = train_df.cache()
    test_df  = test_df.cache()

    logger.info(f"  Train : {train_df.count():,} | Test : {test_df.count():,}")
    return train_df, test_df

# ─────────────────────────────────────────────
# Entraînement ALS
# ─────────────────────────────────────────────

def train_model(train_df, rank=20, max_iter=10, reg_param=0.1):
    logger.info(f"Entraînement ALS — rank={rank}, maxIter={max_iter}, regParam={reg_param}")
    t0 = time.time()

    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        implicitPrefs=False,
        coldStartStrategy="drop",
        nonnegative=False,
    )
    model = als.fit(train_df)
    logger.info(f"Entraînement terminé en {time.time()-t0:.1f}s")
    return model

# ─────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────

def evaluate(model, test_df):
    preds = model.transform(test_df).na.drop()
    ev    = RegressionEvaluator(labelCol="rating", predictionCol="prediction")
    rmse  = ev.setMetricName("rmse").evaluate(preds)
    mae   = ev.setMetricName("mae").evaluate(preds)
    logger.info(f"  RMSE = {rmse:.4f}  |  MAE = {mae:.4f}")
    return rmse, mae

# ─────────────────────────────────────────────
# Recommandations
# ─────────────────────────────────────────────

def recommend_for_user(model, ratings_df, movies_df, user_id, n=10):
    logger.info(f"\nTop {n} recommandations pour l'utilisateur {user_id} :")
    logger.info("-" * 55)

    seen = ratings_df.filter(col("userId") == user_id).select("movieId")
    unseen = movies_df.select("movieId").join(seen, "movieId", "left_anti")

    spark = ratings_df.sparkSession
    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    pairs   = user_df.crossJoin(unseen)

    preds = (
        model.transform(pairs)
        .na.drop()
        .orderBy(col("prediction").desc())
        .limit(n)
        .join(movies_df, "movieId")
    )

    rows = preds.collect()
    if not rows:
        logger.warning("Aucune recommandation trouvée (utilisateur inconnu du modèle ?)")
        return

    for i, r in enumerate(rows, 1):
        genres = r["genres"].replace("|", ", ") if r["genres"] else "?"
        logger.info(f"  {i:2d}. [{r['prediction']:.2f}★]  {r['title']}  —  {genres}")

# ─────────────────────────────────────────────
# Films similaires
# ─────────────────────────────────────────────

def similar_movies(model, movies_df, movie_id, n=10):
    logger.info(f"\nFilms similaires au film ID={movie_id} :")
    logger.info("-" * 55)

    item_factors = model.itemFactors
    target = item_factors.filter(col("id") == movie_id).collect()
    if not target:
        logger.warning(f"Film {movie_id} non présent dans le modèle")
        return

    tv = target[0]["features"]

    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType

    def cosine(v):
        dot   = sum(a * b for a, b in zip(v, tv))
        n1    = math.sqrt(sum(a*a for a in v))
        n2    = math.sqrt(sum(b*b for b in tv))
        return float(dot / (n1 * n2)) if n1 and n2 else 0.0

    cos_udf = udf(cosine, DoubleType())

    rows = (
        item_factors
        .filter(col("id") != movie_id)
        .withColumn("similarity", cos_udf(col("features")))
        .orderBy(col("similarity").desc())
        .limit(n)
        .withColumnRenamed("id", "movieId")
        .join(movies_df, "movieId")
        .select("movieId", "title", "genres", "similarity")
        .collect()
    )

    for i, r in enumerate(rows, 1):
        logger.info(f"  {i:2d}. [sim={r['similarity']:.4f}]  {r['title']}")

# ─────────────────────────────────────────────
# Sauvegarde / chargement
# ─────────────────────────────────────────────

def save_model(model, path="./models/als_model"):
    """
    Sauvegarde le modèle ALS.
    Sur Windows sans winutils, on utilise un dossier temp Spark pour contourner
    les restrictions de permissions HDFS/Hadoop sur le système de fichiers local.
    """
    import tempfile, shutil
    try:
        # Essayer la sauvegarde Spark native
        os.makedirs(path, exist_ok=True)
        model.write().overwrite().save(path)
        logger.info(f"Modèle sauvegardé → {path}")
    except Exception:
        # Fallback : sauvegarder dans %TEMP% (pas de restrictions de permissions)
        tmp_path = os.path.join(tempfile.gettempdir(), "als_model")
        shutil.rmtree(tmp_path, ignore_errors=True)
        model.write().overwrite().save(tmp_path)
        # Copier vers la destination finale
        shutil.rmtree(path, ignore_errors=True)
        shutil.copytree(tmp_path, path)
        logger.info(f"Modèle sauvegardé → {path} (via temp)")

def load_model(path="./models/als_model"):
    model = ALSModel.load(path)
    logger.info(f"Modèle chargé depuis {path}")
    return model

# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Recommandation MovieLens — mode local")
    parser.add_argument("--data-dir",  required=True,        help="Dossier contenant ratings.csv et movies.csv")
    parser.add_argument("--mode",      default="full",
                        choices=["full", "train", "recommend", "similar"],
                        help="full | train | recommend | similar")
    parser.add_argument("--user-id",   type=int, default=1)
    parser.add_argument("--movie-id",  type=int, default=1)
    parser.add_argument("--top-n",     type=int, default=10)
    parser.add_argument("--rank",      type=int, default=20)
    parser.add_argument("--max-iter",  type=int, default=10)
    parser.add_argument("--reg-param", type=float, default=0.1)
    parser.add_argument("--model-dir", default="./models/als_model", help="Chemin de sauvegarde du modèle")
    return parser.parse_args()

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args  = parse_args()
    spark = create_spark()
    spark.sparkContext.setLogLevel("ERROR")

    logger.info("=" * 60)
    logger.info("Système de recommandation distribué — Mode local Spark")
    logger.info("=" * 60)

    try:
        ratings_df, movies_df = load_data(spark, args.data_dir)

        if args.mode in ("full", "train"):
            train_df, test_df = preprocess(ratings_df)

            logger.info("\n" + "=" * 60)
            logger.info("ENTRAÎNEMENT ALS")
            logger.info("=" * 60)
            model = train_model(train_df, args.rank, args.max_iter, args.reg_param)

            logger.info("\n" + "=" * 60)
            logger.info("ÉVALUATION")
            logger.info("=" * 60)
            evaluate(model, test_df)

            save_model(model, args.model_dir)

            if args.mode == "full":
                recommend_for_user(model, ratings_df, movies_df, args.user_id, args.top_n)
                similar_movies(model, movies_df, args.movie_id, args.top_n)

        elif args.mode == "recommend":
            model = load_model(args.model_dir)
            recommend_for_user(model, ratings_df, movies_df, args.user_id, args.top_n)

        elif args.mode == "similar":
            model = load_model(args.model_dir)
            similar_movies(model, movies_df, args.movie_id, args.top_n)

        logger.info("\n✓ Pipeline terminé avec succès")

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interruption utilisateur")
    except Exception as e:
        logger.error(f"Erreur : {e}", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()