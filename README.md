# Exercice technique : détection et comptage d'insectes sur des images via OpenCV

## Contexte et objectifs
Les images utilisées pour répondre à l'exercice sont téléchargeables via le lien suivant : https://zenodo.org/records/7725941/files/Insect_Detect_detection.v4-insect_detect_416_1class.yolov5pytorch.zip?download=1

L'objectif est de concevoir un système de comptage d'insectes sur des fleurs et plantes artificielles à partir d'images issues d'un piège photographique, en respectant deux contraintes majeures : utiliser uniquement des méthodes de traitement d'image classiques (OpenCV, pas de Deep Learning) et garantir que le code puisse tourner sur une machine standard.

**Temps total passé sur le projet :** Environ 10 heures.

## 1. Préparation des données
Initialement, les images sont réparties arbitrairement en 3 jeux (Train / Val / Test) avec dans chaque jeu des images provenant des mêmes jours. Pour construire une approche rigoureuse, la première étape a consisté à rassembler toutes les images et à les identifier par jour (script `merge_days.py`), puis à ne garder que les jours respectant la contrainte des 50 images minimum capturées (script `filter_days.py`) :
- **Regroupement et unification** : Rassemblement de l'ensemble des images et de leurs labels YOLO.
- **Filtrage qualitatif** : Application de la consigne en excluant les journées de captation contenant moins de 50 images. Un total de 7 journées sont conservées à l'issue de ce tri, pour un total de 1101 images.

## 2. Exploration des données préparées
Après obtention d'un jeu de données propre, j'ai entamé une phase d'exploration via des graphiques du nombre d'insectes par jour ou par heure, et des panneaux montrant les images de chaque jour afin d'anticiper d'éventuelles difficultés. Le notebook `data_visualization.ipynb` contient toutes ces étapes d'exploration.
- **Analyse visuelle** : L'observation des données filtrées par jour a permis de constater que l'environnement physique est constant au cours d'une journée, et change très peu d'une journée à l'autre. La luminosité est stable au cours de chaque journée, hormis quelques cas de changement de teinte à des moments précis. 

## 3. Stratégie de séparation
Même si dans la méthode retenue (cf. 4) aucun modèle ou algorithme n'est à proprement parler entraîné, la solution développée a nécessité des données d'ajustement pour optimiser au mieux ses paramètres de fonctionnement. De fait, pour éviter tout *data leakage* et évaluer la capacité de généralisation de la méthode, les données ont été séparées stratégiquement par journées :
- **Jeu d'ajustement** : Assimilable à un jeu d'entraînement ou de validation, il a été utilisé exclusivement pour la recherche et l'optimisation des hyperparamètres de l'algorithme. Il correspond à 4 des 7 jours du jeu de données et contient 702 images.
- **Jeu de test** : Strictement isolé jusqu'à la fin du projet, utilisé uniquement pour extraire les métriques de performance finales. Il correspond aux 3 journées restantes du jeu de données et contient 399 images.

## 4. Solution développée : Background Subtraction (MOG2)
Après recherche des solutions disponibles via OpenCV pour la tâche demandée, la **soustraction de fond adaptative (Background Subtractor MOG2)** a retenu mon attention. Ce modèle apprend itérativement à modéliser le fond de l'image (les fleurs artificielles, la plateforme) et isole ce qui est en mouvement ou nouveau (les insectes).

Le notebook `MOG2_training.ipynb` rassemble toutes les étapes de découverte et d'optimisation de cette approche sur les données d'ajustement.

**Pourquoi ce choix ?**
- Les images d'une même journée partagent un environnement et une luminosité similaires, seule la présence ou l'absence d'insectes les distingue.
- Le modèle MOG2 s'adapte dynamiquement aux changements de teinte repérés lors de l'analyse visuelle des données, limitant les erreurs sur le long terme.

**Comment ça fonctionne ?**
- Le modèle MOG2 (`cv2.createBackgroundSubtractorMOG2()`) passe sur chaque image d'une journée, et sauvegarde en mémoire à chaque passage ce qu'il considère comme le fond et ce qui a bougé sur l'image. Un masque avec du noir pour le fond et du gris / blanc pour ce qui est en mouvement est sauvegardé après le passage du MOG2. 
- Ces masques sont traités par une fonction d'extraction des contours (`cv2.findContours()`) qui y repère les tâches grises ou blanches, puis ces dernières sont filtrées par leur taille et comptées.

**Optimisation des paramètres clés :**
- **`learning_rate` et `var_threshold`** : Ces deux paramètres ont été optimisés sur le jeu d'ajustement afin de permettre la création des meilleurs masques fond / insecte possibles.
- **Filtrage par la taille (`min_area`, `max_area`)** : Paramètres appliqués sur les masques générés pour filtrer le bruit (pixels isolés) et les anomalies trop grandes.
- **Garde-fou (rejet d'images)** : Implémentation d'un plafond de détections (> 8 détections) basé sur l'exploration des données et le nombre observé maximum d'insectes par image, pour rejeter les images aberrantes.

**Limites intrinsèques de l'approche :**
- L'algorithme nécessite de lire les images de manière séquentielle pour construire son historique. Il est par conséquent impossible de traiter une image isolée de manière fiable.
- **L'effet "warm-up"** : Les toutes premières images d'une séquence (ou lors d'un changement brutal d'éclairage) sont mal évaluées car le modèle n'a pas encore stabilisé son fond. Ces erreurs initiales sont actuellement gérées (et exclues) indirectement grâce au garde-fou de détections maximales.

## 5. Évaluation des performances
*Note : Les performances présentées ici sont issues de l'évaluation stricte sur le jeu de test.*

Le notebook `MOG2_evaluation.ipynb` détaille l'évaluation des performances sur chacun des 3 jours du jeu de test (399 images au total). Moyenne réelle globale : 1.92 insectes par image.

### 5.1. Performances de comptage global

| Métrique | Valeur | Interprétation |
| :--- | :--- | :--- |
| **MAE** *(Mean Absolute Error)* | **0.66** | Erreur moyenne de comptage par image. |
| **WAPE** *(Weighted Absolute Percentage Error)* | **34.6 %** | Erreur globale pondérée par le volume total d'insectes. |
| **Taux de rejet** | **5.0 %** | Part des images écartées par le garde-fou (> 8 détections). |

**Fiabilité opérationnelle :** 
- Probabilité de comptage parfait (0 erreur) : **52.0 %**
- Probabilité d'erreur minime (≤ 1 erreur) : **90.5 %**

### 5.2. Précision spatiale (Bounding Boxes)
Compte tenu de la faible quantité d'insectes par image, de potentielles erreurs de comptage peuvent se compenser de manière fortuite. J'ai donc calculé l'**Intersection sur Minimum (Io-Min)** entre les boîtes réelles et prédites. Si l'Io-Min est supérieur à 0.5, on considère que la prédiction encadre bien l'objet réel.

*Note : Les faux négatifs incluent toutes les images rejetées par le garde-fou, ce qui pénalise mécaniquement le modèle mais rend compte de ses performances en conditions réelles.*

| Matrice de confusion | Total | | Métriques spatiales | Score |
| :--- | :--- | :--- | :--- | :--- |
| **Vrais Positifs** (Insectes bien détectés) | 559 | | **Précision globale** | **81.6 %** |
| **Faux Positifs** (Bruits / ombres détectés) | 126 | | **Rappel global** | **74.1 %** |
| **Faux Négatifs** (Insectes ratés + rejets) | 195 | | **F1-Score global** | **77.7 %** |

## 6. Conclusion : que vaut la méthode développée ?
La méthode développée constitue une très bonne *baseline* d'ingénierie classique pour répondre au problème posé (précision globale > 80 %). Elle est légère, rapide, explicable et répond parfaitement à la contrainte matérielle. Elle performe correctement dans un environnement statique où il n'y a pas de changement brusque de lumière.

En revanche, elle manque de compréhension sémantique. Le modèle détecte des "tâches en mouvement", mais ne "sait" pas ce qu'est un insecte. L'ombre d'un insecte volant ou un mouvement de la caméra générera des faux positifs. De plus, sa dépendance à un flux séquentiel d'images la rend inefficace pour traiter une image isolée. Ainsi, la probabilité que le modèle ne fasse absolument aucune erreur spatiale et quantitative sur une image donnée plafonne à 52 %.

Pour dépasser ces limites en conditions réelles sans passer par des modèles beaucoup plus lourds, une piste concrète serait l'hybridation : utiliser MOG2 pour extraire des propositions de zones d'intérêt ultra-rapidement, puis appliquer un classifieur très léger (comme un SVM entraîné sur des caractéristiques géométriques HOG) uniquement sur ces zones recadrées pour valider la nature de l'objet avant de confirmer le comptage.