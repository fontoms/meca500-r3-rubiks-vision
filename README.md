# Rubik's Cube Vision

Application de vision par ordinateur permettant de scanner un Rubik's Cube 3×3 à l'aide d'une webcam, d'identifier les couleurs de ses stickers et de transmettre son état à un solveur.

## 1. Présentation du projet

L'objectif est de développer un programme Python capable de reconnaître les six faces d'un Rubik's Cube à partir d'une webcam.

Le cube sera présenté **face par face devant la caméra**. Le programme devra détecter les neuf stickers visibles, reconnaître leurs couleurs et enregistrer progressivement les informations nécessaires pour reconstituer le cube complet.

La partie vision par ordinateur est indépendante du futur algorithme de résolution.

### Matériel

- Webcam : Logitech HD Pro Webcam C920.
- Ordinateur sous Windows.
- Environnement de développement : WSL2 avec Debian.
- Python.

### Technologies

- Python 3
- OpenCV : acquisition et traitement des images.
- NumPy : manipulation des données et des images.
- Matplotlib : visualisation et débogage.
- Pytest : tests automatisés.

La reconnaissance des couleurs reposera principalement sur le traitement d'image et l'espace colorimétrique HSV. Il n'est pas prévu d'utiliser un modèle d'intelligence artificielle pour cette première version.

---

## 2. Consignes de développement pour Claude Code

**Ces consignes sont prioritaires sur toute autre initiative de développement.**

Le projet doit être développé progressivement, fonctionnalité par fonctionnalité.

### Règle fondamentale : une fonctionnalité à la fois

Claude Code doit impérativement respecter le fonctionnement suivant :

1. Analyser la fonctionnalité à développer.
2. Expliquer brièvement son objectif et son fonctionnement.
3. Proposer une implémentation adaptée.
4. Développer uniquement cette fonctionnalité.
5. Ajouter les tests nécessaires.
6. Expliquer comment tester manuellement le résultat.
7. Présenter les résultats attendus.
8. S'arrêter et attendre la validation explicite de l'utilisateur.

**Claude Code ne doit jamais passer automatiquement à la fonctionnalité suivante.**

Il doit attendre que l'utilisateur confirme que la fonctionnalité actuelle est terminée et validée.

### Autres règles

- Ne pas développer plusieurs fonctionnalités dans une même étape.
- Ne pas anticiper les fonctionnalités suivantes.
- Ne pas créer du code inutilement complexe.
- Privilégier un code simple, modulaire et maintenable.
- Respecter les conventions Python (PEP 8).
- Ajouter des commentaires uniquement lorsqu'ils apportent une réelle valeur.
- Ne pas modifier du code existant sans raison.
- Ne pas remplacer une implémentation fonctionnelle sans justification.
- Utiliser des fonctions courtes avec des responsabilités clairement définies.
- Gérer les erreurs et les cas particuliers.
- Ne pas considérer une fonctionnalité comme terminée tant que ses tests ne sont pas concluants.

### À la fin de chaque fonctionnalité

Claude Code doit présenter :

- Les fichiers créés ou modifiés.
- Une explication du fonctionnement.
- Les tests réalisés.
- Les commandes permettant de reproduire les tests.
- Les éventuels problèmes restants.

Puis il doit s'arrêter.

---

## 3. Architecture cible

L'architecture doit rester simple et évoluer en fonction des besoins réels du projet.

```text
rubiks-vision/
│
├── src/
│   ├── camera.py
│   ├── capture.py
│   ├── detection.py
│   ├── grid.py
│   ├── color_recognition.py
│   ├── calibration.py
│   ├── cube.py
│   └── validation.py
│
├── config/
│   └── colors.json
│
├── tests/
│   ├── test_camera.py
│   ├── test_capture.py
│   ├── test_detection.py
│   ├── test_colors.py
│   └── test_cube.py
│
├── main.py
├── requirements.txt
└── README.md
```

Cette architecture est une cible indicative. Les fichiers doivent être créés au moment où leur fonctionnalité est développée, et non tous dès le début.

---

## 4. Découpage fonctionnel

Chaque fonctionnalité constitue une étape indépendante avec ses propres critères de validation.

### Fonctionnalité 1 — Connexion à la webcam

**Objectif :** permettre au programme d'ouvrir la webcam Logitech C920.

Travail attendu :
- Identifier la caméra disponible.
- Ouvrir le flux vidéo avec OpenCV.
- Vérifier que la caméra est correctement initialisée.
- Gérer les erreurs si la caméra est indisponible.
- Libérer correctement les ressources à la fermeture.

**Critères de validation :**
- La caméra est détectée.
- OpenCV arrive à ouvrir le flux vidéo.
- Le programme ne plante pas si la caméra est inaccessible.
- La caméra est correctement libérée après utilisation.

**Arrêt obligatoire :** ne pas commencer la capture d'images avant validation.

### Fonctionnalité 2 — Capture d'image

**Objectif :** récupérer une image exploitable depuis la webcam.

Travail attendu :
- Récupérer une image depuis le flux vidéo.
- Vérifier que l'image est valide.
- Afficher l'image avec OpenCV.
- Permettre de capturer une image à la demande.
- Enregistrer une image sur le disque pour les tests.

**Critères de validation :**
- L'image est correctement récupérée.
- L'image peut être affichée.
- L'image peut être enregistrée.
- Sa résolution et son format sont correctement identifiés.

**Arrêt obligatoire :** ne pas commencer la détection du cube avant validation.

### Fonctionnalité 3 — Détection du cube

**Objectif :** localiser le Rubik's Cube dans l'image.

Travail attendu :
- Analyser les contours de l'image.
- Rechercher une forme correspondant à la face du cube.
- Utiliser notamment les contours, Canny et `approxPolyDP` si pertinent.
- Identifier les quatre coins de la face.
- Afficher visuellement la zone détectée.

La détection devra progressivement être capable de gérer les variations de position et d'inclinaison.

**Critères de validation :**
- Le cube est détecté dans une image.
- Les quatre coins de la face sont correctement identifiés.
- La détection est visualisable.

**Arrêt obligatoire :** ne pas commencer le découpage en neuf cases avant validation.

### Fonctionnalité 4 — Extraction des neuf stickers

**Objectif :** isoler les neuf cases de la face du cube.

Travail attendu :
- Corriger la perspective de la face si nécessaire.
- Obtenir une vue frontale de la face.
- Div
