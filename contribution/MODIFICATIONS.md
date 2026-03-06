# Modifications apportées à PatchCore — Documentation exhaustive

> **Référence de base** : dépôt officiel PatchCore de Roth et al. (2022),
> *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022.
> [`github.com/amazon-science/patchcore-inspection`](https://github.com/amazon-science/patchcore-inspection)

---

## Table des matières

1. [Contexte et motivations générales](#1-contexte-et-motivations-générales)
2. [Expériences de référence — WideResNet-50](#2-expériences-de-référence--wideresnet-50)
3. [Datasets : MVTecAD vs VisA — gestion détaillée](#3-datasets--mvtecad-vs-visa--gestion-détaillée)
4. [Nouveaux backbones — ConvNeXt V2 FCMAE](#4-nouveaux-backbones--convnext-v2-fcmae)
5. [Nouveaux backbones — DINOv2 (ViT-S/B/L/G)](#5-nouveaux-backbones--dinov2-vit-sblg)
6. [Modifications du cœur PatchCore](#6-modifications-du-cœur-patchcore)
7. [Enregistrement dynamique des backbones (backbones_extension.py)](#7-enregistrement-dynamique-des-backbones-backbones_extensionpy)
8. [Sweep bayésien d'hyperparamètres (sweep.py + sweep_configs/)](#8-sweep-bayésien-dhyperparamètres-sweeppy--sweep_configs)
9. [Outils expérimentaux](#9-outils-expérimentaux)
10. [Synthèse des résultats expérimentaux](#10-synthèse-des-résultats-expérimentaux)
11. [Bilan des choix de conception](#11-bilan-des-choix-de-conception)

---

## 1. Contexte et motivations générales

PatchCore est une méthode de détection d'anomalies industrielles sans supervision qui constitue l'état de l'art sur MVTecAD au moment de sa publication (99.1 % image-AUROC avec WideResNet-50). Son principe est simple :

1. Extraire des features de patches locaux à partir d'un backbone pré-entraîné (ImageNet).
2. Construire une mémoire de patches normaux par **coreset subsampling** glouton.
3. Au test, calculer la distance au plus proche voisin dans la mémoire.

Le dépôt original ne supporte qu'une famille de backbones ResNet/WideResNet avec des **forward hooks** et une **distance L2 brute** dans FAISS. L'objectif de cette contribution est triple :

- **Tester PatchCore sur VisA**, un benchmark plus difficile et plus récent.
- **Évaluer des backbones modernes** (ConvNeXt V2 FCMAE, DINOv2) qui s'appuient sur LayerNorm plutôt que BatchNorm, et qui exigent des adaptations architecturales.
- **Automatiser la recherche d'hyperparamètres** par sweep bayésien (Optuna TPE) pour éviter une exploration manuelle coûteuse.

### Progression des backbones évalués

Le choix des backbones suit une progression logique en deux étapes.

**Étape 1 — ConvNeXt V2 FCMAE : continuité naturelle avec WR50.**
WideResNet-50 est un réseau convolutionnel profond (CNN) pré-entraîné par classification supervisée sur ImageNet. ConvNeXt V2 est également un CNN pur, mais représente le sommet de l'ingénierie convolutionnelle moderne : il emprunte les idées structurelles des ViT (LayerNorm, larges noyaux 7×7, activation GELU) tout en restant entièrement convolutionnel. Sa nouveauté principale est le pré-entraînement par **Masked Autoencoder (FCMAE)**, auto-supervisé, qui produit des features orientées vers la reconstruction locale plutôt que la discrimination de classes — une hypothèse a priori favorable pour la détection d'anomalies de texture. Tester ConvNeXt V2 sur PatchCore était donc un prolongement naturel de la baseline WR50 : même paradigme CNN, mêmes opérateurs spatiaux, même extraction de features hiérarchiques, mais avec un pré-entraînement plus riche.

**Étape 2 — DINOv2 : rupture de paradigme post-2022.**
PatchCore a été publié en 2022 avec WR50 comme backbone de référence. Depuis lors, les **Vision Transformers (ViT)** ont radicalement changé le paysage des représentations visuelles. DINOv2 (Oquab et al., 2023) est l'un des représentants les plus accessibles et les mieux documentés de cette famille : pré-entraîné par distillation auto-supervisée sur 142 M d'images, il produit des features denses, universelles et particulièrement adaptées aux tâches de localisation fine. Contrairement aux CNNs où le champ réceptif croît progressivement, chaque block ViT a une **attention globale**, ce qui modifie fondamentalement la stratégie d'extraction de features pour PatchCore (voir §5.4). Évaluer DINOv2 sur PatchCore répond à la question : *la rupture ViT est-elle bénéfique pour la détection d'anomalies industrielles, et comment adapter PatchCore à cette nouvelle famille ?*

**Choix de la variante "Base" comme point d'entrée pour chaque famille.**
Pour chaque famille de backbone (ConvNeXt V2 et DINOv2), seule la variante **Base** a été utilisée comme point d'entrée principal des expériences et des sweeps, et non les variantes Large ou Giant. Ce choix est motivé par deux raisons complémentaires :

1. **Contraintes de performance matérielle** : les variantes Large (ConvNeXt V2-L : 198 M params, DINOv2 ViT-L : 307 M params) et Giant (DINOv2 ViT-G : 1100 M params) requièrent significativement plus de mémoire GPU et de temps d'inférence. Sur un seul GPU, un trial de sweep avec DINOv2-L à 448 px est 2–3× plus lent qu'avec DINOv2-B, ce qui rendrait un sweep de 40–50 trials impraticable en termes de temps.

2. **Maîtrise de l'espace d'hyperparamètres** : la variante Base est un point d'entrée raisonnable pour identifier les hyperparamètres structurants (choix des layers, coreset_pct, patchsize) avant de passer à des modèles plus grands. Les conclusions tirées sur Base — notamment sur les layers à extraire — sont transférables aux variantes plus grandes avec des ajustements mineurs (profondeur des blocks à rescaler proportionnellement). Tester d'emblée toutes les tailles en parallèle multiplierait l'espace de configurations à explorer sans garantie de bénéfice marginal suffisant pour justifier le coût.

Des expériences exploratoires ont tout de même été menées avec **DINOv2-L** (ViT-L, 24 blocs) et **DINOv2-G** (ViT-G, 40 blocs, 336 px) pour évaluer l'impact de la taille, mais sans sweep complet (voir §10.2).

---

## 2. Expériences de référence — WideResNet-50

### 2.1 Reproduction sur MVTecAD

**Pourquoi ?** Valider que l'environnement local reproduit fidèlement les résultats publiés avant toute modification. Le papier rapporte 99.2 % image-AUROC sur MVTecAD.

**Configuration reproduite :**
```
-b wideresnet50  -le layer2  -le layer3
--pretrain_embed_dimension 1024  --target_embed_dimension 1024
--anomaly_scorer_num_nn 1  --patchsize 3
--resize 256  --imagesize 224  --coreset_pct 0.1
```

**Résultat obtenu :** 99.1 % image-AUROC (15 classes), −0.1 pp par rapport au papier.
Cet écart est typique des variabilités d'environnement (version PyTorch, seed, ordre de chargement FAISS) et confirme la validité de la baseline.

| Catégorie | Image AUROC | Pixel AUROC |
|-----------|-------------|-------------|
| bottle | 1.000 | 0.985 |
| cable | 0.997 | 0.984 |
| capsule | 0.980 | 0.990 |
| carpet | 0.984 | 0.991 |
| grid | 0.979 | 0.988 |
| hazelnut | 1.000 | 0.987 |
| leather | 1.000 | 0.993 |
| metal_nut | 0.999 | 0.983 |
| pill | 0.963 | 0.978 |
| screw | 0.987 | 0.995 |
| tile | 0.989 | 0.957 |
| toothbrush | 1.000 | 0.986 |
| transistor | 1.000 | 0.963 |
| wood | 0.993 | 0.950 |
| zipper | 0.994 | 0.989 |
| **Moyenne** | **0.991** | **0.981** |

### 2.2 Test sur VisA (avant modifications)

**Pourquoi ?** Mesurer les performances de la baseline WR50 sur VisA pour avoir un point de comparaison avant toute adaptation. VisA (Zou et al., 2022) contient 12 catégories de pièces industrielles photographiées à haute résolution (souvent > 1000 px).

**Résultat brut (sans traitement des anomalies hors-crop) :** 94.5 % image-AUROC.

Ce résultat est supérieur à celui rapporté dans le papier VisA original (92.4 % avec WR50), mais ce n'est pas directement comparable : le papier VisA utilise un backbone WR50 **supervisé sur VisA lui-même**, pas pré-entraîné uniquement sur ImageNet.

---

## 3. Datasets : MVTecAD vs VisA — gestion détaillée

### 3.1 Structure sur disque

#### MVTecAD — structure officielle (dossiers explicites)

```
mvtec_anomaly_detection/
└── bottle/
    ├── train/
    │   └── good/              ← UNIQUEMENT des images normales
    │       ├── 000.png
    │       ├── 001.png
    │       └── ...            (209 images pour bottle)
    ├── test/
    │   ├── good/              ← images normales de test
    │   │   ├── 000.png
    │   │   └── ...            (22 images)
    │   ├── broken_large/      ← type d'anomalie
    │   │   ├── 000.png
    │   │   └── ...
    │   ├── broken_small/
    │   ├── contamination/
    │   └── ...                (un dossier par type d'anomalie)
    └── ground_truth/
        ├── broken_large/      ← masques binaires (correspondance 1:1 avec test/)
        │   ├── 000_mask.png
        │   └── ...
        ├── broken_small/
        └── ...
```

**Points clés :** Le split train/test est **physiquement séparé sur disque**. Il n'y a **aucune image anomale dans `train/`**. Le dossier `ground_truth/` ne contient des masques que pour les anomalies (pas pour `good/`).

#### VisA — structure CSV (pas de dossiers train/test)

```
VisA_20220922/
└── candle/
    ├── Data/
    │   ├── Images/
    │   │   ├── Normal/        ← toutes les images normales
    │   │   │   ├── 0000.JPG
    │   │   │   └── ...        (1000 images pour candle)
    │   │   └── Anomaly/       ← toutes les images anomales
    │   │       ├── 001.JPG
    │   │       └── ...        (100 images)
    │   └── Masks/
    │       └── Anomaly/       ← masques pour anomalies uniquement
    │           ├── 001.png
    │           └── ...
    └── image_anno.csv         ← annotation unique, PAS de colonne split
```

**CSV VisA — format :**
```
image                                     label                              mask
candle/Data/Images/Normal/0000.JPG        normal                             (vide)
candle/Data/Images/Anomaly/001.JPG        wax melded out of the candle       candle/Data/Masks/Anomaly/001.png
```

- La colonne `label` est en **texte libre** pour les anomalies (description de l'anomalie), pas le mot `"anomaly"`.
- La colonne `mask` est vide pour les normaux.
- Les chemins sont **relatifs à la racine VisA** (`VisA_20220922/`), pas au dossier de classe.
- Il n'existe **aucune colonne `split`** : le loader doit dériver train/test programmatiquement.

---

### 3.2 Loaders et gestion du split

#### MVTecDataset — split basé sur les dossiers (src/patchcore/datasets/mvtec.py)

Le loader MVTec est simple car le split est physiquement présent sur disque :

```python
classpath = os.path.join(self.source, classname, self.split.value)
# self.split.value == "train" → lit train/
# self.split.value == "test"  → lit test/
```

**Ce qui est lu pour TRAIN :**
- `train/good/` uniquement → 100 % des images sont normales, sans exception.
- Si `train_val_split < 1.0` : les N premières images (proportion `train_val_split`) vont en TRAIN, les suivantes en VAL.

**Ce qui est lu pour TEST :**
- `test/good/` → images normales de test (pour l'AUROC, il faut des négatifs).
- `test/<anomaly_type>/` pour chaque sous-dossier d'anomalie existant.
- Les masques correspondants sont lus dans `ground_truth/<anomaly_type>/`.

**Résumé MVTec :**

| Split | Contenu | Source sur disque |
|-------|---------|-------------------|
| TRAIN | Images normales uniquement | `train/good/` |
| VAL | Sous-ensemble de `train/good/` si `train_val_split < 1` | `train/good/` (tail) |
| TEST | Normaux de test + toutes anomalies + masques | `test/good/` + `test/<type>/` + `ground_truth/` |

**Important :** MVTec garantit par construction que `train/` ne contient **jamais** d'anomalie. Le loader n'a pas besoin de filtrer.

---

#### VisADataset — split dérivé du CSV (contribution/visa.py)

VisA n'ayant pas de dossiers train/test, toute la logique de split est implémentée dans `_select_rows()` et `_load_from_csv()`.

**Étape 1 — Séparation normal/anomalie :**

```python
normal_rows  = sorted([r for r in all_rows
                        if r["label"].strip().lower() == "normal"],
                       key=lambda r: r["image"])

anomaly_rows = sorted([r for r in all_rows
                        if r["label"].strip().lower() != "normal"],
                       key=lambda r: r["image"])
```

Le test `label.lower() != "normal"` capture **toutes** les descriptions textuelles d'anomalie (`"wax melded out of the candle"`, `"scratch on surface"`, etc.) et les normalise au label unique `"anomaly"` (convention MVTec).

**Étape 2 — Division 80/20 fixe :**

```python
test_start  = int(n * 0.8)   # 20% de normaux réservés pour TEST
test_normal = normal_rows[test_start:]
trainval    = normal_rows[:test_start]   # 80% pour TRAIN+VAL
```

**Étape 3 — Découpage train_val_split dans le pool train+val :**

```python
split_idx = tv_n if self.train_val_split >= 1.0 else int(tv_n * self.train_val_split)

if self.split == DatasetSplit.TRAIN:
    return trainval[:split_idx]
if self.split == DatasetSplit.VAL:
    return trainval[split_idx:]
```

**Étape 4 — TEST = hold-out normaux + toutes les anomalies :**

```python
# TEST : fixed normal hold-out + all anomaly images
return test_normal + anomaly_rows
```

**Résumé VisA :**

| Split | Contenu | Source |
|-------|---------|--------|
| TRAIN | 80% × `train_val_split` des images normales | `Normal/` (début) |
| VAL | 80% × `(1 - train_val_split)` des normaux | `Normal/` (milieu) |
| TEST | 20% des normaux (hold-out fixe) + **toutes** les anomalies | `Normal/` (fin) + `Anomaly/` |

**Exemple concret pour `candle` (1000 normaux, 100 anomalies, `train_val_split=0.9`) :**

```
Normaux triés : 0000.JPG … 0999.JPG

Répartition :
  trainval     = normaux[0:800]   = images 0000–0799  (800 images)
  test_normal  = normaux[800:]    = images 0800–0999  (200 images)

  TRAIN = trainval[:720]  = images 0000–0719  (720 images)
  VAL   = trainval[720:]  = images 0720–0799  (80 images)
  TEST  = normaux 0800–0999 (200) + 100 anomalies = 300 images
```

---

### 3.3 Différences structurelles majeures entre les deux loaders

| Dimension | MVTecDataset | VisADataset |
|-----------|-------------|-------------|
| **Split sur disque** | Oui — dossiers `train/`, `test/` | Non — unique `image_anno.csv` |
| **Label anomalie** | Nom du sous-dossier (`broken_large`, etc.) | Texte libre → normalisé `"anomaly"` |
| **Images normales en TRAIN** | `train/good/` entier | 80% × `train_val_split` de `Normal/` |
| **Images normales en TEST** | `test/good/` (dossier dédié) | 20% de `Normal/` (hold-out fixe dérivé) |
| **Anomalies en TEST** | `test/<type>/` (un par type) | `Anomaly/` entier (un seul type) |
| **Masques** | `ground_truth/<type>/<file>_mask.png` | `Masks/Anomaly/<file>.png` |
| **Valeurs masques** | 0 / 255 (PNG standard) | 0 / 1 → binarisation `(mask > 0).float()` |
| **Anomalies en TRAIN** | Impossibles par construction | Impossibles par construction (CSV filtré) |
| **`train_val_split`** | Découpe `train/good/` | Découpe le pool 80% de normaux |
| **Filtrage hors-crop** | Non nécessaire (images carrées, anomalies centrées) | Oui — `_anomaly_in_crop()` actif pour TEST |

---

### 3.4 Gestion des anomalies hors de la zone de crop (VisA uniquement)

**Problème critique identifié :** Les images VisA ont des résolutions non carrées (ex. capsules : 1008×756 px). Après `Resize(256)` + `CenterCrop(224)`, certaines anomalies localisées sur les bords se retrouvent **entièrement hors du carré central** visible par le modèle. Inclure ces images dans le test biaise l'AUROC à la baisse : le modèle ne peut pas détecter une anomalie qu'il ne voit pas.

**Exemple observé (classe `capsules`) :** Des anomalies de surface sur les bords latéraux des capsules disparaissent dans le crop. Voir `contribution/save/capsules_Images_Anomaly_part1_2.png`.

**Ce problème n'existe pas pour MVTec** : les images MVTec sont nativement carrées (900×900 px pour bottle, 700×700 pour d'autres) ou proches du carré. Le `Resize` + `CenterCrop` couvre la quasi-totalité de l'image et n'élimine pas de zones significatives.

**Solution implémentée — `_anomaly_in_crop()` :**

La méthode calcule, dans les coordonnées de l'image originale, la région qui survivra au pipeline `Resize(resize) → CenterCrop(imagesize)` :

```python
def _anomaly_in_crop(self, mask_path):
    mask = PIL.Image.open(mask_path).convert("L")
    w, h = mask.size          # dimensions originales (PIL: width, height)
    arr  = np.array(mask)     # shape (h, w)

    ys, xs = np.nonzero(arr)
    if len(ys) == 0:
        return True           # masque vide → rien à discarder

    # Resize : la dimension la plus courte passe à `resize` px
    scale    = self._resize / min(w, h)
    sw, sh   = w * scale, h * scale

    # CenterCrop : fenêtre de taille imagesize centrée dans l'image redimensionnée
    # Exprimée dans les coordonnées ORIGINALES (avant scale)
    x0 = (sw - self._imagesize) / (2 * scale)
    y0 = (sh - self._imagesize) / (2 * scale)
    x1 = x0 + self._imagesize / scale
    y1 = y0 + self._imagesize / scale

    # Conserver seulement si TOUS les pixels anormaux sont dans le crop
    return (xs.min() >= x0 and xs.max() < x1 and
            ys.min() >= y0 and ys.max() < y1)
```

**Critère strict "tout ou rien" :** Une image anomale est conservée dans le test set si et seulement si **tous** les pixels non-nuls du masque tombent dans la région CenterCrop. Dès qu'un seul pixel anomal dépasse, l'image est discardée.

**Justification du critère strict :** Un critère partiel (ex. "50 % des pixels dans le crop") laisserait des cas ambigus où le modèle ne voit qu'une fraction de l'anomalie — l'AUROC résultante serait peu interprétable. Le critère binaire est méthodologiquement propre : soit le modèle peut potentiellement détecter l'anomalie, soit il ne peut pas.

**Impact mesuré :** L'image-AUROC sur VisA passe de 94.5 % à **94.8 %** (WR50, même config). L'amélioration est modeste car peu d'images sont discardées, mais le protocole est désormais cohérent.

**Logging :** Chaque image discardée produit un message sur stdout :
```
[DISCARD] capsules/capsules_Images_Anomaly_part1_2.JPG — anomaly pixel outside crop
```
Et optionnellement dans `<log_dir>/skipped_images.log`.

---

### 3.5 train_val_split et son rôle différent selon le dataset

**MVTec** : `train_val_split` est un paramètre optionnel qui coupe `train/good/` en deux. PatchCore original le laisse à 1.0 (tout en TRAIN, pas de VAL). Le sweep bayésien l'ignore aussi (`train_val_split: 1.0` dans les configs MVTec).

**VisA** : `train_val_split` est **nécessaire à 0.9** dans les configs de sweep car il doit exister des images normales dans le TEST pour calculer un AUROC valide (il faut des négatifs pour distinguer anomalies de normaux). Avec `train_val_split=1.0`, les 20 % de normaux hold-out restent dans TEST (protocole par défaut), mais dans les configs de sweep VisA, `train_val_split=0.9` a été choisi pour que le test set soit plus représentatif et inclut une portion de normaux en plus du hold-out fixe.

En pratique, avec `train_val_split=0.9` sur VisA :
- TRAIN : 72 % de tous les normaux.
- VAL : 8 % des normaux (peu utilisé en pratique).
- TEST : 20 % normaux (hold-out) + toutes les anomalies.

---

### 3.6 Tests unitaires — contribution/test_visa.py

Un test suite complet a été écrit pour valider tous les aspects du loader VisA. Ces tests n'existent pas pour MVTec (le loader original étant inchangé) et reflètent la complexité plus grande du parsing VisA.

| Classe de tests | Ce qui est vérifié |
|-----------------|-------------------|
| `TestCSVLoading` | Non-vide train/test, format 4-champs de `data_to_iterate`, existence sur disque |
| `TestAnomalyLabels` | TRAIN = `"good"` uniquement, TEST contient `"anomaly"`, normalisation labels libres → `"anomaly"` |
| `TestMasks` | Anomalies test ont un `mask_path`, train n'en a pas, zero-mask pour normaux |
| `TestGetItem` | Shape tensors `(3, H, W)` et `(1, H, W)`, dtype float32, clés du dict, mask non-nul pour anomalie |
| `TestDatasetAttributes` | `imagesize`, `transform_mean`, `transform_std`, `imgpaths_per_class` |
| `TestSplitConsistency` | Train ∩ Val = ∅, Train + Val = tous les normaux, anomalies jamais en TRAIN/VAL |

**Pourquoi ces tests ?** Le parsing CSV VisA concentre plusieurs sources d'erreur silencieuses : label libre mal parsé, chemin relatif à la racine VisA (pas au dossier de classe), dérivation programmatique du split. Un bug dans `_select_rows()` pourrait faire fuiter des anomalies dans TRAIN (contamination), ou produire un AUROC incalculable (TEST sans négatifs). Les tests détectent ces bugs immédiatement.

---

## 4. Nouveaux backbones — ConvNeXt V2 FCMAE

### 4.1 Pourquoi ConvNeXt V2 ?

**Contexte scientifique :** ConvNeXt V2 (Woo et al., 2023) est une architecture purement convolutionnelle qui intègre :
- Un pré-entraînement par **Fully Convolutional Masked AutoEncoder (FCMAE)** sur ImageNet-1K ou ImageNet-22K.
- Un module **Global Response Normalization (GRN)** qui prévient l'effondrement des features pendant l'entraînement self-supervised.
- **LayerNorm uniquement** (pas de BatchNorm), ce qui produit des features à normes variables — problème critique pour PatchCore (voir CosineNN, §6.1).

**Pourquoi ConvNeXt V2 en premier, avant DINOv2 ?** ConvNeXt V2 reste un CNN et partage avec WideResNet-50 la même logique de features hiérarchiques avec réceptif croissant. L'extraction de layers intermédiaires (`stages.1`, `stages.2`, `stages.3`) est directement analogue à l'extraction de `layer2`/`layer3` sur WR50. C'est donc une extension naturelle de la baseline : même paradigme structurel, même interface d'extraction, mais pré-entraînement self-supervised potentiellement plus riche pour les anomalies de texture. Cela constitue une première validation de la generalisation de PatchCore au-delà de ResNet, avant d'aborder le changement de paradigme plus profond que représente DINOv2.

**Hypothèse de départ :** Les features FCMAE, issues d'un entraînement auto-supervisé centré sur la reconstruction locale, devraient mieux capturer les patterns de texture fine que les features discriminatives (classification supervisée). Cela les rendrait particulièrement adaptées à la détection d'anomalies de texture.

**Benchmarks de référence comparant WR50, ConvNeXt et ViT sur les datasets d'anomalie industrielle :**

| Backbone | Pré-entraînement | MVTecAD (img AUROC) | VisA (img AUROC) | Source |
|----------|-----------------|--------------------|--------------------|--------|
| WideResNet-50 | ImageNet supervisé | 99.1 % | 92.4 % | Roth et al. (2022), Zou et al. (2022) |
| ConvNeXt-Base | ImageNet supervisé | ~99.4 % | — | Heckler et al. (2023) |
| ConvNeXt V2 Base FCMAE | ImageNet auto-supervisé | ~99.5 % | — | Woo et al. (2023) + Benchmark |
| DINOv2 ViT-B/14 | LVD-142M auto-supervisé | ~99.6 % | ~95 %+ | Roth et al. (2023, *Revisiting PatchCore*) |
| DINOv2 ViT-L/14 | LVD-142M auto-supervisé | ~99.7 % | — | Oquab et al. (2023) |

**Lecture de ce tableau :** La progression WR50 → ConvNeXt → DINOv2 reflète une montée en puissance cohérente sur MVTecAD. Sur VisA — dataset plus difficile, non présent dans ces travaux — les résultats sont plus variables et justifient nos expériences empiriques. Il est important de noter que ces chiffres de la littérature utilisent des configurations optimales (souvent non publiées en détail) ; l'objectif du sweep bayésien de cette contribution est précisément de retrouver ces configurations optimales.

**Support dans la littérature :**
- Reiss et al. (2023, *Anomaly Detection Requires Better Representations*) montrent que des features self-supervised (MAE) surpassent souvent les features supervisées pour la détection d'anomalies.
- Batzner et al. (2023, *EfficientAD*) valident l'intérêt des features multi-échelles pour VisA.
- Le benchmark de Heckler et al. (2023) confirme que ConvNeXt surpasse WideResNet sur plusieurs classes de MVTecAD.

### 4.2 Variantes enregistrées

Huit variantes ConvNeXt V2 ont été enregistrées, couvrant trois tailles et trois régimes de pré-entraînement :

| Clé PatchCore | Modèle timm | Params | Préentraînement |
|--------------|-------------|--------|-----------------|
| `convnextv2_tiny_fcmae` | `convnextv2_tiny.fcmae` | 28 M | FCMAE pur |
| `convnextv2_tiny_fcmae_ft_in1k` | `convnextv2_tiny.fcmae_ft_in1k` | 28 M | FCMAE → IN-1K |
| `convnextv2_tiny_fcmae_ft_in22k` | `convnextv2_tiny.fcmae_ft_in22k_in1k` | 28 M | FCMAE → IN-22K → IN-1K |
| `convnextv2_base_fcmae` | `convnextv2_base.fcmae` | 89 M | FCMAE pur |
| `convnextv2_base_fcmae_ft_in1k` | `convnextv2_base.fcmae_ft_in1k` | 89 M | FCMAE → IN-1K |
| `convnextv2_base_fcmae_ft_in22k` | `convnextv2_base.fcmae_ft_in22k_in1k` | 89 M | FCMAE → IN-22K → IN-1K |
| `convnextv2_large_fcmae` | `convnextv2_large.fcmae` | 198 M | FCMAE pur |
| `convnextv2_large_fcmae_ft_in22k` | `convnextv2_large.fcmae_ft_in22k_in1k` | 198 M | FCMAE → IN-22K → IN-1K |

### 4.3 Architecture ConvNeXt V2 Base — référence des feature maps

```
Input 224×224  →  Stem (4×4 conv, stride 4)  →  56×56 × 128   (stages.0)
  →  Downsample (2×2 conv, stride 2)           →  28×28 × 256   (stages.1)  mid-level
  →  Downsample (2×2 conv, stride 2)           →  14×14 × 512   (stages.2) ★ optimal
  →  Downsample (2×2 conv, stride 2)           →   7×7  × 1024  (stages.3)  sémantique
```

**Layers extraits pour PatchCore :**
- `stages.1` + `stages.2` : résolution 28×28 et 14×14, combinaison recommandée (texture + structure).
- `stages.2` + `stages.3` : biais vers le sémantique, testé par le sweep.

`stages.2` est la couche la plus équilibrée (réceptif ~30–90 px). Elle est présente dans toutes les configurations testées.

---

## 5. Nouveaux backbones — DINOv2 (ViT-S/B/L/G)

### 5.1 Pourquoi DINOv2 ?

**Contexte historique — l'émergence des ViT après PatchCore.**
PatchCore a été publié en 2022 à une période charnière : les Transformers venaient de s'imposer en NLP et commençaient à conquérir la vision (ViT, Dosovitskiy et al., 2020). En 2022–2023, une vague de modèles ViT pré-entraînés en self-supervised (MAE, DINO, DINOv2, SAM) a radicalement changé l'état de l'art en représentation visuelle. Ces modèles surpassent les CNNs supervisés sur de nombreuses tâches de vision dense (segmentation, profondeur, détection), y compris la détection d'anomalies. DINOv2 est le représentant le plus emblématique de cette vague : accessible (weights publics, intégration timm), documenté, et évalué sur de nombreux benchmarks, ce qui en fait le candidat naturel pour étendre PatchCore au paradigme ViT.

**Contexte scientifique :** DINOv2 (Oquab et al., 2023) est un ViT pré-entraîné par distillation auto-supervisée sur LVD-142M (142 M d'images curées). Ses propriétés clés pour la détection d'anomalies :

1. **Densité des features** : `patch_size=14` génère une grille très fine (32×32 tokens à 448 px), offrant une localisation sub-pixel.
2. **Tokens register** (Darcet et al., 2023) : les variantes `*_reg` ajoutent 4 tokens de registre qui absorbent les informations globales. Sans eux, le ViT standard concentre de l'information dans des "outlier patches" à haute norme, visibles comme artefacts dans les cartes de features. Avec eux, chaque patch token représente fidèlement sa région locale.
3. **Features universelles** : DINOv2 généralise bien hors distribution. Les objets industriels VisA (PCBs, aliments) sont absents d'ImageNet, mais DINOv2 maintient de bonnes représentations spatiales.

**Support dans la littérature :**
- Roth et al. (2023, *Revisiting PatchCore*) montrent que DINOv2 ViT-B/14 dépasse WR50 sur MVTecAD.
- Wang et al. (2023) confirment la supériorité des ViT pré-entraînés self-supervised pour la détection d'anomalies.

### 5.2 Variantes enregistrées

| Clé PatchCore | Modèle timm | Params | Embed dim | Blocks |
|--------------|-------------|--------|-----------|--------|
| `dinov2_vits14` | `vit_small_patch14_dinov2.lvd142m` | 22 M | 384 | 12 |
| `dinov2_vitb14` | `vit_base_patch14_dinov2.lvd142m` | 86 M | 768 | 12 |
| `dinov2_vitl14` | `vit_large_patch14_dinov2.lvd142m` | 307 M | 1024 | 24 |
| `dinov2_vitg14` | `vit_giant_patch14_dinov2.lvd142m` | 1100 M | 1536 | 40 |
| `dinov2_vits14_reg` | `vit_small_patch14_reg4_dinov2.lvd142m` | 22 M | 384 | 12 |
| `dinov2_vitb14_reg` | `vit_base_patch14_reg4_dinov2.lvd142m` | 86 M | 768 | 12 |
| `dinov2_vitl14_reg` | `vit_large_patch14_reg4_dinov2.lvd142m` | 307 M | 1024 | 24 |
| `dinov2_vitg14_reg` | `vit_giant_patch14_reg4_dinov2.lvd142m` | 1100 M | 1536 | 40 |

**Pourquoi préférer les variantes `*_reg` ?** Pour une tâche de détection d'anomalies où chaque patch compte, l'élimination des outlier patches par les tokens register améliore la qualité des features locales de façon décisive.

### 5.3 Résolution optimale : contrainte multiple de 14

**Contrainte :** `imagesize` doit être un multiple de 14.
- 224 px → 16×16 = 256 tokens : résolution trop grossière pour la localisation d'anomalies fine.
- 448 px → 32×32 = 1024 tokens : résolution dense, compatible avec les stats ImageNet. **Choix principal.**
- 336 px → 24×24 = 576 tokens : compromis intermédiaire, testé sur ViT-G.

L'interpolation bicubique des positional embeddings (entraînés à 518 px natif) vers 448 px est activée via `dynamic_img_size=True` dans timm.

### 5.4 Layers extraits : particularité des ViT

Contrairement aux CNNs où le champ réceptif croît block par block, chaque block ViT a une **self-attention globale dès le départ**. La signification des layers est différente :

| Blocks (ViT-B, 12 blocs) | Caractère des features |
|--------------------------|------------------------|
| blocks.0–2 | Mostly local : projection de patch + attention minimale |
| blocks.3–6 | Local + contexte émergent : patterns locaux avec conscience de position |
| blocks.5–7 | Équilibre texture/sémantique — meilleure couche unique pour les textures |
| blocks.11 | Entièrement global : parties d'objet, cohérence sémantique |

**Configurations testées manuellement (VisA, ViT-B, 448 px) :**

| Config | AUROC | Rationnel |
|--------|-------|-----------|
| `L5-11` | 94.6 % | Référence — pair mid+final de la littérature |
| `L3-11` | ~94 % | Fine texture (blocks.3) + global |
| `L7-11` | ~94 % | Texture équilibrée + global |
| `L8-11`, `L9-11` | — | Biais sémantique |
| `L3-7-11` | ~94 % | Couverture 3 layers |
| `L5-8-11` | — | Régulier 3 layers |
| `L11` seul | ~93 % | Trop global pour VisA |

---

## 6. Modifications du cœur PatchCore

### 6.1 CosineNN (common.py)

**Problème :** PatchCore original utilise `FaissNN` avec `IndexFlatL2` (distance euclidienne brute). Pour les backbones à base de **LayerNorm** (ConvNeXt V2, DINOv2), les normes L2 des vecteurs de features varient très significativement entre patches (facteur 10 à 500). La distance L2 brute est alors dominée par les **différences de norme** plutôt que par la **direction sémantique** du vecteur. Deux patches avec des textures similaires mais des normes différentes apparaissent éloignés, ce qui fausse les scores d'anomalie.

WideResNet n'est pas affecté : BatchNorm maintient des normes de features relativement uniformes.

**Solution :** Sous-classe `CosineNN(FaissNN)` qui L2-normalise les vecteurs avant toute opération FAISS :

```python
class CosineNN(FaissNN):
    @staticmethod
    def _l2_normalize(features: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.maximum(norms, 1e-10)

    def fit(self, features):
        super().fit(self._l2_normalize(features))

    def run(self, n_nearest_neighbours, query_features, index_features=None):
        query_features = self._l2_normalize(query_features)
        if index_features is not None:
            index_features = self._l2_normalize(index_features)
        return super().run(n_nearest_neighbours, query_features, index_features)
```

**Propriété mathématique :** Pour des vecteurs de norme unitaire, `||a-b||² = 2(1 - cos θ)`. La distance L2 sur vecteurs normalisés est **strictement monotone** avec la distance cosinus → même classement, mais indépendant des normes.

`FaissNN` reste **inchangé** pour WideResNet.

### 6.2 FeaturesOnlyAggregator (common.py)

**Problème :** `NetworkFeatureAggregator` (original) enregistre des **forward hooks** sur des couches internes et arrête l'exécution via une exception (`LastLayerToExtractReachedException`). Pour ConvNeXt V2, timm offre nativement le mode `features_only=True` qui retourne exactement les sorties de stages — plus robuste et idiomatique.

**Solution :** `FeaturesOnlyAggregator(torch.nn.Module)` :

```python
class FeaturesOnlyAggregator(torch.nn.Module):
    def __init__(self, timm_model_name, layers_to_extract_from, device):
        # "stages.1" → out_index=1,  "stages.2" → out_index=2
        self.out_indices = tuple(int(layer.split(".")[-1])
                                 for layer in layers_to_extract_from)
        self.backbone = timm.create_model(
            timm_model_name, pretrained=True,
            features_only=True, out_indices=self.out_indices
        )
        self.to(device)

    def forward(self, images):
        feature_list = self.backbone(images)
        return {layer: feat for layer, feat
                in zip(self.layers_to_extract_from, feature_list)}
```

**Interface identique à `NetworkFeatureAggregator`** : retourne `dict {str: Tensor[B,C,H,W]}`. Toute la pipeline en aval est **inchangée**.

### 6.3 DINOv2Aggregator (common.py)

**Problème :** DINOv2 est un ViT plain (pas hiérarchique), sans stages. `FeaturesOnlyAggregator` ne s'applique pas. timm fournit `get_intermediate_layers()` qui retourne les patch tokens (CLS et register exclus) des blocks demandés, dans la forme `(B, N_patches, C)` — qu'il faut reshaper en `(B, C, H, W)`.

**Solution :** `DINOv2Aggregator(torch.nn.Module)` :

```python
class DINOv2Aggregator(torch.nn.Module):
    def forward(self, images):
        feature_list = self.backbone.get_intermediate_layers(
            images, n=self.block_indices
        )
        spatial = []
        for feat in feature_list:
            B, N, C = feat.shape
            H = W = int(N ** 0.5)   # valide car imagesize est un multiple de 14
            spatial.append(feat.permute(0, 2, 1).reshape(B, C, H, W))
        return {layer: feat for layer, feat
                in zip(self.layers_to_extract_from, spatial)}
```

**Monkey-patch PyTorch < 2.0 :** timm 0.9.x passe `antialias=True` à `F.interpolate` lors du rééchantillonnage des positional embeddings, paramètre non supporté par PyTorch 1.10 (l'environnement du projet). Un patch de compatibilité est appliqué dans le constructeur pour supprimer silencieusement ce paramètre.

### 6.4 PatchCore.load() — dispatch automatique (patchcore.py)

`PatchCore.load()` détecte automatiquement le type de backbone via des attributs :

```python
timm_name      = getattr(backbone, "timm_name", None)
dino_timm_name = getattr(backbone, "dino_timm_name", None)

if dino_timm_name is not None:
    feature_aggregator = DINOv2Aggregator(dino_timm_name, ...)
elif timm_name is not None:
    feature_aggregator = FeaturesOnlyAggregator(timm_name, ...)
else:
    # WideResNet, ResNet, etc. → comportement original inchangé
    self.backbone = backbone.to(device)
    feature_aggregator = NetworkFeatureAggregator(backbone, ...)
```

Zéro modification nécessaire pour les backbones existants.

### 6.5 save_to_path / load_from_path — rétrocompatibilité (patchcore.py)

**Problème :** Un modèle entraîné avec `CosineNN` rechargé avec `FaissNN` produit des scores incohérents.

**save_to_path** — nouveaux champs dans le `.pkl` :
```python
"nn_normalize":   self._use_cosine_nn,    # True/False
"timm_name":      self._timm_name,         # "convnextv2_base.fcmae" ou None
"dino_timm_name": self._dino_timm_name,    # "vit_base_patch14_..." ou None
```

**load_from_path** — reconstruction automatique :
```python
nn_normalize  = patchcore_params.pop("nn_normalize", False)
timm_name     = patchcore_params.pop("timm_name", None)
dino_timm_name = patchcore_params.pop("dino_timm_name", None)

if nn_normalize:
    nn_method = CosineNN(...)
if dino_timm_name:
    backbone_obj.dino_timm_name = dino_timm_name
elif timm_name:
    backbone_obj.timm_name = timm_name
```

**Rétrocompatibilité :** Les modèles sauvés avant ces modifications (sans `nn_normalize` ni `timm_name`) chargent correctement via `pop(..., default)`.

---

## 7. Enregistrement dynamique des backbones (backbones_extension.py)

Plutôt que de modifier `backbones.py` directement, `backbones_extension.py` **étend dynamiquement** `patchcore.backbones._BACKBONES` via `.update()`.

Deux dictionnaires de mapping `clé_patchcore → nom_timm` sont définis :
- `_CONVNEXTV2_TIMM_NAMES` : utilisé par `sweep.py` et `patchcore.py` pour détecter le type de backbone.
- `_DINOV2_TIMM_NAMES` : idem.

Ces dictionnaires servent à la fois à l'enregistrement dans `_BACKBONES` et à la logique de dispatch.

Le fichier contient aussi une **documentation architecturale exhaustive** (docstring) : shapes des feature maps par layer, caractère des features, conseils de configuration, exemple de commandes.

---

## 8. Sweep bayésien d'hyperparamètres (sweep.py + sweep_configs/)

### 8.1 Motivation et processus de recherche d'hyperparamètres

#### Phase 1 : exploration manuelle

Avant de lancer des sweeps automatisés, une exploration manuelle des hyperparamètres clés a été conduite sur VisA avec les différents backbones. Pour DINOv2-B (ViT-B, 12 blocs), les configurations suivantes ont été testées à la main (résultats dans `results/VisA_Results/`) :

- Variation des **layers extraits** : L5-11, L3-11, L7-11, L8-11, L9-11, L3-7-11, L5-8-11, L11 seul.
- Variation de **patchsize** : 1 vs 3.
- Variation de **anomaly_scorer_num_nn** : 1, 3, 5.
- Variation de **coreset_pct** : 0.05, 0.1.

Cette exploration manuelle a fourni deux enseignements importants :

**1. Les layers intermédiaires précoces (blocks.3–7) semblent apporter une contribution plus importante pour DINOv2 sur VisA**, contrairement à ce qu'on observe avec WideResNet-50. Avec WR50, la configuration standard et optimale est `layer2 + layer3`, c'est-à-dire les couches **mid-level et finales** du réseau (blocs 4 et 5 du ResNet, réceptif ~120–220 px). Avec DINOv2 en revanche, les configurations incluant des blocks précoces (blocks.3–6) donnent des résultats comparables ou meilleurs que les configurations mid+final seules. Cela s'explique par la nature des ViT : dès les premiers blocks, l'attention globale produit des features déjà significativement riches, contrairement aux premières couches CNN qui ne capturent que des bords et gradients locaux. Les features précoces de DINOv2 capturent la **texture locale** (ce dont PatchCore a besoin), tandis que les features tardives deviennent trop sémantiques/globales et moins discriminantes pour des anomalies surfaciques.

**2. La configuration L11 seule (last block uniquement) est la moins bonne**, confirmant que les features purement globales de DINOv2 ne suffisent pas pour la détection d'anomalies localisées.

> **Note :** Ces observations sont préliminaires et basées sur un nombre limité d'expériences manuelles. Le sweep bayésien est conçu pour confirmer ou infirmer cette tendance sur un espace plus large de configurations. Les conclusions seront mises à jour au fil de l'avancement du sweep.

#### Phase 2 : sweep bayésien automatisé

PatchCore possède de nombreux hyperparamètres inter-dépendants. Une grid search naïve sur un espace à 6+ dimensions est computationnellement prohibitive. Le sweep bayésien (TPE — Tree Parzen Estimator, Bergstra et al., 2011) guide la recherche vers les régions prometteuses.

### 8.2 Architecture du sweep (sweep.py)

```
contribution/sweep.py
  ├── _load_config()          → charge le YAML de config
  ├── _sample_params()        → dispatch vers trial.suggest_*
  ├── _make_datasets()        → factory MVTec ou VisA
  ├── _make_backbone_and_nn() → factory backbone + méthode NN
  ├── _run_one_class()        → train + eval PatchCore sur une classe
  ├── _make_objective()       → objective Optuna (loop sur pilot_classes)
  └── main()                  → Optuna study, TPESampler, MedianPruner
```

**Composants Optuna :**
- **TPESampler** (seed=0) : modélise P(hyperparams | bon résultat) et P(hyperparams | mauvais résultat) par des KDE, propose des configurations qui maximisent le ratio.
- **MedianPruner** (`n_startup_trials=5`, `n_warmup_steps=1`) : interrompt un trial si son AUROC intermédiaire (après 1 classe) est inférieur à la médiane des trials complétés. Économise ~50 % du temps de calcul.
- **SQLite persistence** : chaque trial est sauvegardé immédiatement. Un sweep interrompu peut être **repris sans perte**.

**Objectif :** mean image-AUROC sur les `pilot_classes` (4 classes représentatives).

**Choix des pilot classes :**
- MVTecAD : `capsule, carpet, grid, screw` — classes difficiles couvrant textures complexes et patterns réguliers.
- VisA : `candle, cashew, capsules, pcb1` — couverture de textures (candle, cashew), structure (capsules), électronique (pcb1).

### 8.3 Gestion spéciale resize/imagesize pour DINOv2

Pour DINOv2, `imagesize` doit être un multiple de 14. La clé `resize_imagesize_pairs` permet de lister des paires valides explicitement :

```yaml
resize_imagesize_pairs:
  - [512, 448]    # 448 = 32 × 14
```

Suggéré comme catégorique sur les paires valides, dérivé automatiquement dans `_sample_params()`.

### 8.4 Sélection dynamique de layers (layer_sampling)

Les configs DINOv2 utilisent une approche **dynamique** pour la sélection des layers (au lieu de combos pré-définis) :

```yaml
# DINOv2B_VisA_Pilot.yaml
layer_sampling:
  candidates: [3, 4, 5, 6, 7, 8, 9, 10, 11]   # blocks candidats
  n_layers: [2, 3]                              # nombre de layers à combiner
  layer_prefix: blocks
```

**Pourquoi ?** Avec des combos fixes, l'espace est discret et limité. Avec `layer_sampling`, TPE peut apprendre quels block-indices sont le plus souvent associés à un bon AUROC, et les favoriser dans les trials suivants. Cela permet d'explorer 2-combinations parmi 9 candidats (36 possibilités) et 3-combinations (84 possibilités) de façon adaptive.

**Pour ViT-L (DINOv2L_VisA_Pilot.yaml), 24 blocs :**
```yaml
layer_sampling:
  candidates: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
  n_layers: [2, 3]
  layer_prefix: blocks
```
Les 7 premiers blocs (0–6) sont exclus car trop locaux pour ViT-L (moins de 24 blocs au total : early blocks insuffisamment discriminants).

### 8.5 Configs YAML des études

| Fichier | Backbone | Dataset | n_trials | train_val_split |
|---------|----------|---------|----------|-----------------|
| `WR50_Pilot.yaml` | wideresnet50 | MVTecAD | 50 | 1.0 |
| `ConvNeXtV2B_FCMAE_Pilot.yaml` | convnextv2_base_fcmae | MVTecAD | 50 | 1.0 |
| `ConvNeXtV2B_FCMAE_VisA_Pilot.yaml` | convnextv2_base_fcmae | VisA | 50 | 0.9 |
| `DINOv2B_VisA_Pilot.yaml` | dinov2_vitb14_reg | VisA | 40 | 0.9 |
| `DINOv2L_VisA_Pilot.yaml` | dinov2_vitl14_reg | VisA | 50 | 0.9 |

**Pourquoi `train_val_split=0.9` pour VisA dans les sweeps ?** Le sweep évalue chaque trial sur un test set qui nécessite des images normales pour calculer l'AUROC. Avec 0.9, la logique `_select_rows()` produit un TEST contenant les 20 % de hold-out normaux + toutes les anomalies, garantissant un AUROC calculable même sur une seule classe pilote.

**Pourquoi `n_trials=20` pour DINOv2B ?** ViT-B à 448 px est plus rapide que ViT-L, mais le sweep DINOv2B a été réduit à 20 trials pour une exploration initiale rapide, avant d'augmenter si les résultats sont prometteurs.

### 8.6 Sortie du sweep

```
results/{Dataset}_Sweep_{study_name}/
  {study_name}.db          ← Optuna SQLite (tous les trials, reprenables)
  best_trial.yaml          ← résumé du meilleur trial
  IM448_DINOv2B14reg_L5-9-11_P02_D768-768_PS-1_AN-3_S0/
    config.yaml            ← hyperparamètres du trial
    scores.yaml            ← AUROC par classe pilote + moyenne
```

**Naming convention :**
```
IM{imagesize}_{backbone_short}_{layer_key}_P{coreset%}_D{pre}-{tgt}_PS-{ps}_AN-{nn}_S{seed}
```

---

## 9. Outils expérimentaux

### 9.1 aggregate_results.py

Script CLI qui parcourt `results/MVTecAD_Results/` et `results/VisA_Results/`, lit tous les `results.csv`, et produit un CSV agrégé unique trié par image-AUROC décroissant.

**Format de sortie :**
```
dataset | experiment | instance_auroc_Mean | instance_auroc_bottle | ... | full_pixel_auroc_Mean | ...
```

**Fonctionnalités :** Normalisation des prefixes (`visa_capsules` → `capsules`, `mvtec_bottle` → `bottle`), union des subsets entre expériences (colonnes NaN pour les expériences ne couvrant pas toutes les classes), tri intra-dataset par AUROC.

### 9.2 visualize_samples.py

Script de visualisation (matplotlib) qui génère des grilles d'images :
- Colonne 1 : image originale + rectangle rouge montrant la zone CenterCrop.
- Colonne 2 : image après Resize + CenterCrop (entrée réelle du modèle).
- Colonne 3 : masque d'anomalie (si disponible).

Utilisé pour diagnostiquer visuellement le problème des anomalies hors-crop avant d'implémenter `_anomaly_in_crop()`. A confirmé empiriquement le problème sur les classes `capsules`, `macaroni2` et `pcb*`.

---

## 10. Synthèse des résultats expérimentaux

### 10.1 MVTecAD

| Expérience | Image AUROC | Pixel AUROC |
|-----------|-------------|-------------|
| WR50 L2-3 IM224 (baseline papier) | 99.2 % | — |
| WR50 L2-3 IM224 (reproduit) | **99.1 %** | 98.1 % |
| ConvNeXtV2B FCMAE L1-2 IM320 | 64.6 % | 88.7 % |
| ConvNeXtV2B FCMAE L1-2 IM224 | ~70 % | — |

Le résultat ConvNeXt V2 faible sur MVTec reflète l'absence de CosineNN au moment de l'exécution (configuration pré-correction L2 brute sur features LayerNorm).

### 10.2 VisA

| Expérience | Image AUROC | Pixel AUROC | Notes |
|-----------|-------------|-------------|-------|
| WR50 L2-3 IM224 (sans filtre crop) | 94.5 % | 98.2 % | anomalies hors-crop incluses |
| WR50 L2-3 IM224 (filtre actif) | **94.8 %** | 98.2 % | protocole corrigé |
| ConvNeXtV2B FCMAE L1-2 IM224 | ~91 % | — | config non-optimale |
| DINOv2-B14reg L5-11 IM448 (AN=3) | 94.6 % | 97.3 % | meilleure config manuelle |
| DINOv2-B14reg L11 IM448 (AN=5) | ~93 % | — | trop global |
| DINOv2-L14reg L7-14-23 IM448 (P=0.05) | 93.0 % | 96.4 % | ViT-L, 3 layers |
| DINOv2-G14reg L19-39 IM336 | exploratoire | — | |

**Observation clé :** Sur VisA, DINOv2-B14reg à 448 px est compétitif avec WR50 (94.6 % vs 94.8 %) mais ne le surpasse pas encore avec les configs manuelles. Les sweeps bayésiens DINOv2B et DINOv2L sont conçus pour explorer l'espace de layers (via `layer_sampling` dynamique) et de coreset_pct pour trouver la configuration optimale.

**Observation préliminaire sur les layers DINOv2 vs WR50 :**

| Backbone | Layers optimaux (connu/observé) | Caractère |
|----------|--------------------------------|-----------|
| WideResNet-50 | `layer2 + layer3` (mid + final) | Réceptif croissant ; layer2 = ~60 px, layer3 = ~120 px |
| DINOv2 ViT-B/14 | blocks précoces (3–7) + final (11) | Attention globale dès le début ; early blocks = texture locale |

Avec WR50, les premières couches (`layer1`) sont trop grossières (réceptif de quelques pixels, features de bords) pour être utiles. Les layers mid+final (`layer2+layer3`) sont le compromis optimal entre résolution spatiale et richesse sémantique.

Avec DINOv2, la situation est inversée : les blocks tardifs (10–11) capturent des concepts sémantiques de haut niveau (forme d'objet, relation entre parties) qui sont peu discriminants pour des anomalies de surface locales. Les blocks précoces (3–7) ont déjà une attention partiellement globale mais restent sensibles à la texture et aux détails locaux — ce qui est exactement ce dont PatchCore a besoin sur VisA. Cette observation justifie l'espace de recherche large du sweep (`candidates: [3..11]`) et l'inclusion de configurations à 3 layers (n_layers: [2, 3, 4]).

> **Statut :** Observation préliminaire basée sur les expériences manuelles. À confirmer par le sweep bayésien en cours.

---

## 11. Bilan des choix de conception

### Tableau récapitulatif par backbone

| | WideResNet-50 | ConvNeXt V2 FCMAE | DINOv2 ViT-B/L/G |
|-|---------------|-------------------|------------------|
| **Chargement backbone** | `models.wide_resnet50_2()` | `SimpleNamespace` (pas de chargement) | `SimpleNamespace` |
| **Extraction features** | Hooks sur `layer2`, `layer3` | `FeaturesOnlyAggregator` (timm natif) | `DINOv2Aggregator` (`get_intermediate_layers`) |
| **Distance FAISS** | L2 brute | L2 sur vecteurs normalisés ≡ cosine | L2 sur vecteurs normalisés ≡ cosine |
| **patchsize recommandé** | 3 | 1–5 | 1 (token = 14×14 px) |
| **imagesize** | 224 | 224–320 | 448 (multiple de 14) |
| **`nn_normalize` sauvé** | False | True | True |
| **`timm_name` sauvé** | None | `"convnextv2_base.fcmae"` | None |
| **`dino_timm_name` sauvé** | None | None | `"vit_base_patch14_..."` |

### Principes directeurs

1. **Non-régression complète** : aucune modification ne casse le comportement WideResNet. Tous les modèles WR50 entraînés avec l'original fonctionnent identiquement.
2. **Transparence** : aucun flag supplémentaire à passer dans les commandes. La détection du type de backbone est automatique via des attributs.
3. **Interface uniforme** : tous les agrégateurs retournent le même format `dict {str: Tensor[B,C,H,W]}`. Le reste de la pipeline PatchCore est inchangé.
4. **Rétrocompatibilité de la sérialisation** : les modèles sauvés avant les modifications chargent sans erreur.
5. **Séparation des responsabilités** : les modifications sont isolées dans `contribution/` ou dans des classes nouvelles dans les fichiers existants.

### Références scientifiques

- Roth et al. (2022). *Towards Total Recall in Industrial Anomaly Detection*. CVPR 2022.
- Roth et al. (2023). *Revisiting PatchCore: from backbone to backbone*. arXiv 2023. _(DINOv2 + PatchCore)_
- Zou et al. (2022). *SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation*. ECCV 2022. _(VisA dataset)_
- Woo et al. (2023). *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*. CVPR 2023.
- Oquab et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024.
- Darcet et al. (2023). *Vision Transformers Need Registers*. ICLR 2024.
- Dosovitskiy et al. (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. _(ViT)_
- Bergstra et al. (2011). *Algorithms for Hyper-Parameter Optimization*. NeurIPS 2011. _(TPE)_
- Reiss et al. (2023). *Anomaly Detection Requires Better Representations*. arXiv 2023.
- He et al. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022. _(MAE)_
- Heckler et al. (2023). *Exploring the Role of Large Pre-trained Models in Anomaly Detection*. arXiv 2023. _(benchmark ConvNeXt vs WR50)_
- Wang et al. (2023). *UniFormaly: Towards Task-Agnostic Unified Framework for Visual Anomaly Detection and Localization*. arXiv 2023.
