#!/usr/bin/env python3
"""
Téléchargement automatique du dataset MovieLens
================================================
Télécharge et décompresse MovieLens (version small ou 25M).
"""

import urllib.request
import zipfile
import os
import sys
import shutil
import logging

logger = logging.getLogger(__name__)

DATASETS = {
    "small": {
        "url":    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "folder": "ml-latest-small",
        "desc":   "100K évaluations, 9K films, 600 utilisateurs",
    },
    "25m": {
        "url":    "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "folder": "ml-25m",
        "desc":   "25M évaluations, 62K films, 162K utilisateurs",
    },
}


def download_movielens(version: str = "small", dest_dir: str = "/tmp") -> str:
    """
    Télécharge et décompresse le dataset MovieLens.
    
    Args:
        version : 'small' (100K, rapide) ou '25m' (25M, production)
        dest_dir: répertoire de destination local
    
    Returns:
        Chemin vers le dossier contenant les CSV
    """
    if version not in DATASETS:
        raise ValueError(f"Version inconnue : {version}. Choisir parmi {list(DATASETS)}")

    dataset = DATASETS[version]
    url    = dataset["url"]
    folder = dataset["folder"]

    zip_path  = os.path.join(dest_dir, f"{folder}.zip")
    data_path = os.path.join(dest_dir, folder)

    if os.path.isdir(data_path):
        logger.info(f"Dataset déjà présent : {data_path}")
        return data_path

    # Téléchargement avec barre de progression
    logger.info(f"Téléchargement MovieLens {version} — {dataset['desc']}")
    logger.info(f"URL : {url}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {downloaded/1_048_576:.1f}/{total_size/1_048_576:.1f} MB",
              end="", flush=True)

    urllib.request.urlretrieve(url, zip_path, reporthook=progress)
    print()  # nouvelle ligne après la barre

    # Décompression
    logger.info(f"Décompression vers {dest_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    # Nettoyage du zip
    os.remove(zip_path)
    logger.info(f"Dataset prêt : {data_path}")

    # Lister les fichiers
    for f in sorted(os.listdir(data_path)):
        size_mb = os.path.getsize(os.path.join(data_path, f)) / 1_048_576
        logger.info(f"  {f:<20s} {size_mb:8.2f} MB")

    return data_path


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Téléchargement MovieLens")
    parser.add_argument("--version", default="small", choices=["small", "25m"])
    parser.add_argument("--dest",    default="/tmp/movielens")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    path = download_movielens(args.version, args.dest)
    print(f"\nDataset disponible : {path}")
