import shutil
from pathlib import Path

import pandas as pd

# Script permettant de regrouper toutes les images et leurs labels dans un seul dossier.
# Initialement, elles sont séparés en train / val / test mais pas nécessairement comme souhaité


def gather_dataset_files(base_dir: str) -> pd.DataFrame:
    """Regroupe dans un dictionnaire des informations sur le nom, l'origine et le jour associé à chaque image"""
    base_path = Path(base_dir)
    data = []

    for split in ["train", "valid", "test"]:
        split_dir = base_path / split

        if not split_dir.exists():
            continue

        # rglob trouve les images récursivement
        for img_path in split_dir.rglob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Déduction du chemin du label YOLO (.txt)
            if "images" in img_path.parts:
                label_path = Path(
                    str(img_path).replace("images", "labels")
                ).with_suffix(".txt")
            else:
                label_path = img_path.with_suffix(".txt")

            # Extraction de la date (les 8 premiers caractères du nom, ex: '20220905')
            date_str = img_path.name[:8]

            data.append(
                {
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "label_path": str(label_path) if label_path.exists() else None,
                    "original_split": split,
                    "jour": date_str,
                }
            )

    return pd.DataFrame(data)


def copy_files_to_new_directory(df: pd.DataFrame, dest_dir: str):
    """Copie les images et les labels dans un nouveau dossier unifié."""
    dest_path = Path(dest_dir)
    images_dest = dest_path / "images"
    labels_dest = dest_path / "labels"

    # Création des dossiers cibles
    images_dest.mkdir(parents=True, exist_ok=True)
    labels_dest.mkdir(parents=True, exist_ok=True)

    print(f"Création des dossiers :\n- {images_dest}\n- {labels_dest}")
    print(f"Copie de {len(df)} images et labels en cours...")

    for _, row in df.iterrows():
        # Copie de l'image
        if pd.notna(row["image_path"]):
            shutil.copy2(row["image_path"], images_dest / row["image_name"])

        # Copie du label
        if pd.notna(row["label_path"]):
            label_name = Path(row["label_path"]).name
            shutil.copy2(row["label_path"], labels_dest / label_name)

    print("Copie terminée avec succès !")


if __name__ == "__main__":
    # Dossier source
    DATA_DIR = (
        "./data/Insect_Detect_detection.v4-insect_detect_416_1class.yolov5pytorch"
    )

    # Dossier cible
    MERGED_OUTPUT_DIR = "./data/merged_dataset"

    # Récupération des métadonnées
    df_dataset = gather_dataset_files(DATA_DIR)

    print(f"Total d'images recensées : {len(df_dataset)}")
    print("\nAperçu du DataFrame avec la nouvelle colonne 'jour' :")
    print(df_dataset[["image_name", "jour", "original_split"]].head())
    print("\n--------------------------------------------------\n")

    # Copie physique des fichiers
    copy_files_to_new_directory(df_dataset, MERGED_OUTPUT_DIR)
