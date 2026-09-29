import shutil
from pathlib import Path

import pandas as pd

# Script permettant de copier dans un nouveau dossier les images et labels provenant de jours avec au moins 50 images présentes.


def filter_and_copy_dataset(source_dir: str, dest_dir: str, min_images: int = 50):
    """Fonction principale permettant d'identifier de quel jour provient chaque image via le timestamp du nom,
    et de filtrer les jours avec 50 images ou plus"""
    source_path = Path(source_dir)
    images_source = source_path / "images"
    labels_source = source_path / "labels"

    dest_path = Path(dest_dir)
    images_dest = dest_path / "images"
    labels_dest = dest_path / "labels"

    # Lister les images du dataset fusionné et extraire les dates
    data = []
    if not images_source.exists():
        print(f"Erreur : Le dossier {images_source} n'existe pas. Vérifie le chemin.")
        return

    for img_path in images_source.glob("*.*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            date_str = img_path.name[
                :8
            ]  # Les 8 premiers caractères correspondent à la date
            label_path = labels_source / img_path.with_suffix(".txt").name

            data.append(
                {
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "label_path": str(label_path) if label_path.exists() else None,
                    "jour": date_str,
                }
            )

    df = pd.DataFrame(data)
    print(f"Nombre total d'images trouvées initialement : {len(df)}")

    # Identifier les jours ayant au moins 'min_images' images
    counts_per_day = df["jour"].value_counts()
    valid_days = counts_per_day[counts_per_day >= min_images].index

    # Filtrer le DataFrame pour ne garder que ces jours-là
    df_filtered = df[df["jour"].isin(valid_days)]

    print(f"\nJours conservés (>= {min_images} images) : {len(valid_days)} jours")
    print("Répartition par jour :")
    print(counts_per_day[valid_days])
    print(f"\nTotal d'images après filtrage : {len(df_filtered)}")

    # Copier les fichiers conservés dans le nouveau dossier
    images_dest.mkdir(parents=True, exist_ok=True)
    labels_dest.mkdir(parents=True, exist_ok=True)

    print(f"\nCopie vers {dest_dir} en cours...")
    for _, row in df_filtered.iterrows():
        # Copie de l'image
        shutil.copy2(row["image_path"], images_dest / row["image_name"])

        # Copie du label s'il existe
        if pd.notna(row["label_path"]):
            label_name = Path(row["label_path"]).name
            shutil.copy2(row["label_path"], labels_dest / label_name)

    print("Copie terminée avec succès !")


if __name__ == "__main__":
    # Chemin vers le dossier créé à l'étape précédente
    SOURCE_DIR = "./data/merged_dataset"

    # Chemin vers le dossier final de travail
    DEST_DIR = "./data/merged_filtered_dataset"

    filter_and_copy_dataset(SOURCE_DIR, DEST_DIR, min_images=50)
