"""
Gestionnaire Hadoop HDFS
========================
Gère l'ingestion et le stockage distribué du dataset MovieLens sur HDFS.
"""

import subprocess
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class HDFSManager:
    """
    Interface Python pour interagir avec Hadoop HDFS.
    Gère le chargement, la vérification et l'organisation des données.
    """

    def __init__(self, hdfs_url: str = "hdfs://localhost:9000",
                 hadoop_home: str = "/opt/hadoop"):
        self.hdfs_url  = hdfs_url
        self.hadoop_home = hadoop_home
        self.hdfs_cmd  = f"{hadoop_home}/bin/hdfs"

    def _run(self, *args, check: bool = True) -> subprocess.CompletedProcess:
        """Exécute une commande HDFS et retourne le résultat."""
        cmd = [self.hdfs_cmd, "dfs"] + list(args)
        logger.debug(f"HDFS cmd: {' '.join(cmd)}")
        return subprocess.run(cmd, capture_output=True, text=True, check=check)

    # ─────────────────────────────────────────────
    # Opérations répertoire
    # ─────────────────────────────────────────────

    def mkdir(self, hdfs_path: str):
        """Crée un répertoire sur HDFS (incluant les parents)."""
        result = self._run("-mkdir", "-p", hdfs_path, check=False)
        if result.returncode == 0:
            logger.info(f"Répertoire créé : {hdfs_path}")
        else:
            logger.warning(f"mkdir {hdfs_path} : {result.stderr.strip()}")

    def ls(self, hdfs_path: str) -> list:
        """Liste le contenu d'un répertoire HDFS."""
        result = self._run("-ls", hdfs_path, check=False)
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().split("\n")[1:]  # ignorer la ligne d'en-tête
        return [line.split()[-1] for line in lines if line]

    def exists(self, hdfs_path: str) -> bool:
        """Vérifie si un chemin existe sur HDFS."""
        result = self._run("-test", "-e", hdfs_path, check=False)
        return result.returncode == 0

    # ─────────────────────────────────────────────
    # Transfert de fichiers
    # ─────────────────────────────────────────────

    def upload(self, local_path: str, hdfs_path: str, overwrite: bool = False):
        """Transfère un fichier local vers HDFS."""
        if not Path(local_path).exists():
            raise FileNotFoundError(f"Fichier local introuvable : {local_path}")

        args = ["-put"]
        if overwrite:
            args.append("-f")
        args += [local_path, hdfs_path]

        self._run(*args)
        size_mb = Path(local_path).stat().st_size / 1_048_576
        logger.info(f"Chargé : {local_path} → {hdfs_path} ({size_mb:.1f} MB)")

    def download(self, hdfs_path: str, local_path: str):
        """Télécharge un fichier depuis HDFS vers le système local."""
        self._run("-get", hdfs_path, local_path)
        logger.info(f"Téléchargé : {hdfs_path} → {local_path}")

    # ─────────────────────────────────────────────
    # Gestion dataset MovieLens
    # ─────────────────────────────────────────────

    def setup_movielens(self, local_data_dir: str,
                        hdfs_base: str = "/data/movielens") -> str:
        """
        Prépare l'arborescence HDFS et charge le dataset MovieLens.
        
        Structure HDFS créée :
          /data/movielens/
            raw/       ← fichiers CSV bruts
            processed/ ← données nettoyées par Spark
          /models/
            als_recommender/ ← modèle ALS sauvegardé
          /output/
            recommendations/ ← recommandations générées
        """
        dirs = [
            f"{hdfs_base}/raw",
            f"{hdfs_base}/processed",
            "/models/als_recommender",
            "/output/recommendations",
        ]
        for d in dirs:
            self.mkdir(d)

        # Charger les fichiers CSV
        for filename in ("ratings.csv", "movies.csv", "tags.csv", "links.csv"):
            local_file = os.path.join(local_data_dir, filename)
            hdfs_file  = f"{hdfs_base}/raw/{filename}"
            if os.path.exists(local_file):
                if self.exists(hdfs_file):
                    logger.info(f"Déjà présent sur HDFS : {hdfs_file}")
                else:
                    self.upload(local_file, hdfs_file)
            else:
                logger.warning(f"Fichier absent localement : {local_file}")

        return f"{hdfs_base}/raw"

    def get_cluster_stats(self) -> dict:
        """Retourne les statistiques du cluster Hadoop (espace, DataNodes...)."""
        result = self._run("-df", "-h", check=False)
        stats = {"raw_output": result.stdout}

        # Parser l'espace disponible
        for line in result.stdout.splitlines():
            if self.hdfs_url in line or "hdfs" in line.lower():
                parts = line.split()
                if len(parts) >= 4:
                    stats.update({
                        "filesystem": parts[0],
                        "size":       parts[1],
                        "used":       parts[2],
                        "available":  parts[3],
                    })
        return stats

    def set_replication(self, hdfs_path: str, factor: int = 3):
        """Configure le facteur de réplication HDFS pour un fichier."""
        self._run("-setrep", "-w", str(factor), hdfs_path)
        logger.info(f"Réplication ×{factor} configurée pour {hdfs_path}")
