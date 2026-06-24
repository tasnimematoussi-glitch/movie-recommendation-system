"""
API REST Flask — Système de recommandation
==========================================
Expose les recommandations ALS via des endpoints HTTP.
Le modèle Spark est chargé une fois au démarrage et réutilisé.
"""

from flask import Flask, jsonify, request, abort
from functools import lru_cache
import logging
import time
import os

# Import du moteur (adapté selon votre environnement)
# from src.recommendation_engine import RecommendationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ─────────────────────────────────────────────
# Initialisation du moteur (singleton)
# ─────────────────────────────────────────────

_engine = None

def get_engine():
    """Retourne l'instance singleton du moteur de recommandation."""
    global _engine
    if _engine is None:
        from src.recommendation_engine import RecommendationEngine
        _engine = RecommendationEngine(
            hdfs_path=os.getenv("HDFS_URL", "hdfs://localhost:9000")
        )
        model_path = os.getenv("MODEL_PATH", "hdfs://localhost:9000/models/als_recommender")
        _engine.load_model(model_path)
        _engine.load_movielens_data(os.getenv("DATA_PATH", "hdfs://localhost:9000/data/movielens/raw"))
        logger.info("Moteur de recommandation initialisé")
    return _engine


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — vérification que l'API est opérationnelle."""
    return jsonify({
        "status": "ok",
        "service": "Movie Recommendation API",
        "timestamp": time.time(),
    })


@app.route("/api/recommend/<int:user_id>", methods=["GET"])
def recommend_for_user(user_id: int):
    """
    Retourne les N meilleurs films recommandés pour un utilisateur.
    
    GET /api/recommend/42?n=10
    
    Réponse :
    {
      "user_id": 42,
      "recommendations": [
        {"movie_id": 318, "title": "Shawshank Redemption", "genres": "Drama", "predicted_rating": 4.87},
        ...
      ],
      "count": 10,
      "latency_ms": 123
    }
    """
    n = request.args.get("n", default=10, type=int)
    n = max(1, min(n, 50))   # clamp entre 1 et 50

    start = time.time()
    try:
        engine = get_engine()
        recs_df = engine.recommend_for_user(user_id, n=n)
        recs = recs_df.collect()
    except Exception as e:
        logger.error(f"Erreur recommandation utilisateur {user_id}: {e}")
        abort(500, description=str(e))

    latency_ms = round((time.time() - start) * 1000)

    return jsonify({
        "user_id": user_id,
        "recommendations": [
            {
                "movie_id":        r["movieId"],
                "title":           r["title"],
                "genres":          r["genres"],
                "predicted_rating": round(float(r["prediction"]), 3),
            }
            for r in recs
        ],
        "count":      len(recs),
        "latency_ms": latency_ms,
    })


@app.route("/api/similar/<int:movie_id>", methods=["GET"])
def similar_movies(movie_id: int):
    """
    Retourne les films les plus similaires à un film donné.
    Similarité basée sur les vecteurs de facteurs latents ALS.
    
    GET /api/similar/318?n=10
    """
    n = request.args.get("n", default=10, type=int)
    n = max(1, min(n, 50))

    start = time.time()
    try:
        engine = get_engine()
        similar_df = engine.find_similar_movies(movie_id, n=n)
        if similar_df is None:
            abort(404, description=f"Film {movie_id} non trouvé dans le modèle")
        similar = similar_df.collect()
    except Exception as e:
        logger.error(f"Erreur similarité film {movie_id}: {e}")
        abort(500, description=str(e))

    latency_ms = round((time.time() - start) * 1000)

    return jsonify({
        "movie_id":      movie_id,
        "similar_movies": [
            {
                "movie_id":   r["movieId"],
                "title":      r["title"],
                "genres":     r["genres"],
                "similarity": round(float(r["similarity"]), 4),
            }
            for r in similar
        ],
        "count":      len(similar),
        "latency_ms": latency_ms,
    })


@app.route("/api/movies/<int:movie_id>", methods=["GET"])
def movie_info(movie_id: int):
    """Retourne les informations sur un film spécifique."""
    try:
        engine = get_engine()
        movie = (
            engine.movies_df
            .filter(engine.movies_df.movieId == movie_id)
            .collect()
        )
    except Exception as e:
        abort(500, description=str(e))

    if not movie:
        abort(404, description=f"Film {movie_id} introuvable")

    m = movie[0]
    return jsonify({
        "movie_id": m["movieId"],
        "title":    m["title"],
        "genres":   m["genres"].split("|") if m["genres"] else [],
    })


@app.route("/api/stats", methods=["GET"])
def dataset_stats():
    """Retourne les statistiques globales du dataset et du modèle."""
    try:
        engine   = get_engine()
        n_ratings = engine.ratings_df.count()
        n_users  = engine.ratings_df.select("userId").distinct().count()
        n_movies = engine.movies_df.count()
        model_rank = engine.model.rank if engine.model else None
    except Exception as e:
        abort(500, description=str(e))

    return jsonify({
        "dataset": {
            "ratings": n_ratings,
            "users":   n_users,
            "movies":  n_movies,
        },
        "model": {
            "algorithm": "ALS (Alternating Least Squares)",
            "rank":      model_rank,
            "framework": "Apache Spark MLlib",
        },
    })


# ─────────────────────────────────────────────
# Gestion des erreurs
# ─────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "message": str(e)}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "message": str(e)}), 500


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Démarrage de l'API sur le port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
