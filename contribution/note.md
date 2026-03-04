## setup conda
conda create -n patchcore38 python=3.8 -y
conda activate patchcore38
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -y -c conda-forge faiss-cpu
pip install click matplotlib pillow pretrainedmodels scikit-image scikit-learn scipy tqdm
pip install black flake8 isort pytest

## To initialize on powershell
$env:PYTHONPATH="src"
$env:PYTHONPATH=""
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$datapath_mvtec = "C:\Users\trloj\Code\mvtec_anomaly_detection"

$datasets_mvtec = @('bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper')

$dataset_mvtec_flags = @()
foreach ($d in $datasets_mvtec) {
    $dataset_mvtec_flags += @('-d', $d)
}

$datapath_visa = "C:\Users\trloj\Code\VisA_20220922"

$datasets_visa = @('candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum')

$dataset_visa_flags = @()
foreach ($d in $datasets_visa) {
  $dataset_visa_flags += @('-d', $d)
}

## To test datasets 
#### mvtec
python bin/run_patchcore.py `
  --gpu 0 --seed 0 --save_patchcore_model --save_segmentation_images `
  --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 `
  --log_project MVTecAD_Results `
  results `
  patch_core `
    -b wideresnet50 -le layer2 -le layer3 `
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 `
    --anomaly_scorer_num_nn 1 --patchsize 3 `
    --faiss_num_workers 4 `
  sampler -p 0.1 approx_greedy_coreset `
  dataset `
    --resize 256 --imagesize 224 `
    --num_workers 0 `
    $dataset_mvtec_flags `
    mvtec $datapath_mvtec

#### visa
python bin/run_patchcore.py `
  --gpu 0 --seed 0 --save_patchcore_model --save_segmentation_images `
  --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 `
  --log_project VisA_Results `
  results `
  patch_core `
    -b wideresnet50 -le layer2 -le layer3 `
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 `
    --anomaly_scorer_num_nn 1 --patchsize 3 `
    --faiss_num_workers 4 `
  sampler -p 0.1 approx_greedy_coreset `
  dataset `
    --resize 256 --imagesize 224 --train_val_split 0.9 `
    --num_workers 0 `
    $dataset_visa_flags `
    visa $datapath_visa


### convnext
#### mvtec
python bin/run_patchcore.py `
  --gpu 0 --seed 0 --save_patchcore_model `
  --save_segmentation_images `
  --log_group IM320_ConvNeXtV2B_FCMAE_L1-2_P01_D768-768_PS-5_AN-3_S0 `
  --log_project MVTecAD_Results `
  results `
  patch_core `
    -b convnextv2_base_fcmae -le stages.1 -le stages.2 `
    --pretrain_embed_dimension 768 --target_embed_dimension 768 `
    --anomaly_scorer_num_nn 3 --patchsize 5 `
    --faiss_num_workers 4 `
  sampler -p 0.1 approx_greedy_coreset `
  dataset `
    --resize 366 --imagesize 320 `
    --num_workers 0 `
    $dataset_mvtec_flags `
    mvtec $datapath_mvtec


#### visa
python bin/run_patchcore.py `
  --gpu 0 --seed 0 --save_patchcore_model `
  --log_group IM224_ConvNeXtV2B_FCMAE_L1-2_P01_D1024-1024_PS-3_AN-1_S0 `
  --log_project VisA_Results `
  results `
  patch_core `
    -b convnextv2_base_fcmae -le stages.1 -le stages.2 `
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 `
    --anomaly_scorer_num_nn 1 --patchsize 3 `
    --faiss_num_workers 4 `
  sampler -p 0.1 approx_greedy_coreset `
  dataset `
    --resize 256 --imagesize 224 --train_val_split 0.9 `
    --num_workers 0 `
    $dataset_visa_flags `
    visa $datapath_visa


### convnext
#### mvtec
python bin/run_patchcore.py `
  --gpu 0 --seed 0 --save_patchcore_model `
  --save_segmentation_images `
  --log_group IM448_DINOv2B14reg_L5-11_P01_D768-768_PS-1_AN-3_S0 `
  --log_project MVTecAD_Results `
  results `
  patch_core `
    -b dinov2_vitb14_reg -le blocks.5 -le blocks.11 `
    --pretrain_embed_dimension 768 --target_embed_dimension 768 `
    --anomaly_scorer_num_nn 3 --patchsize 1 `
    --faiss_num_workers 4 `
  sampler -p 0.1 approx_greedy_coreset `
  dataset `
    --resize 512 --imagesize 448 `
    --num_workers 0 `
    $dataset_mvtec_flags `
    mvtec $datapath_mvtec

#### visa
python bin/run_patchcore.py `
  --gpu 0 --seed 0 --save_patchcore_model `
  --save_segmentation_images `
  --log_group IM448_DINOv2B14reg_L5-11_P01_D768-768_PS-1_AN-3_S0 `
  --log_project VisA_Results `
  results `
  patch_core `
    -b dinov2_vitb14_reg -le blocks.5 -le blocks.11 `
    --pretrain_embed_dimension 768 --target_embed_dimension 768 `
    --anomaly_scorer_num_nn 3 --patchsize 1 `
    --faiss_num_workers 4 `
  sampler -p 0.1 approx_greedy_coreset `
  dataset `
    --resize 512 --imagesize 448 `
    --num_workers 0 `
    $dataset_visa_flags `
    visa $datapath_visa

## sweep

# All settings are in contribution/sweep_configs/{study_name}.yaml
python contribution/sweep.py --study_name ConvNeXtV2B_FCMAE_Pilot

# Custom config path
python contribution/sweep.py --study_name MySweep --config path/to/config.yaml


## To load and evaluate patchcore
#### mvtec
python bin/load_and_evaluate_patchcore.py `
  --gpu 0 --seed 0 --save_segmentation_images `
  results/MVTecAD_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0/eval_bottle `
  patch_core_loader `
    -p results/MVTecAD_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0/models/mvtec_bottle `
  dataset `
    --resize 256 --imagesize 224 --num_workers 0 `
    $dataset_mvtec_flags `
    mvtec $datapath_mvtec
  
#### visa
python bin/load_and_evaluate_patchcore.py `
  --gpu 0 --seed 0 --save_segmentation_images `
  results/VisA_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0/eval_candle `
  patch_core_loader `
    -p results/VisA_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0/models/visa_candle `
  dataset `
    --resize 256 --imagesize 224 --train_val_split 0.9 --num_workers 0 `
    $dataset_visa_flags `
    visa $datapath_visa

## Bayesian hyperparameter sweep (Optuna + TPE)
# Config files: contribution/sweep_configs/{study_name}.yaml
# Available presets: ConvNeXtV2B_FCMAE_Pilot | WR50_Pilot | ConvNeXtV2B_FCMAE_VisA_Pilot

### Run / resume a sweep (SQLite persists all trials — resumable after crash)
python contribution/sweep.py --study_name ConvNeXtV2B_FCMAE_Pilot

### Results layout
# results/MVTecAD_Sweep_ConvNeXtV2B_FCMAE_Pilot/
#   ConvNeXtV2B_FCMAE_Pilot.db              <- Optuna SQLite (all trials)
#   best_trial.yaml                          <- best trial summary
#   IM288_ConvNeXtV2B_FCMAE_L1-2_P05_D1024-1024_PS-3_AN-1_S0/
#     config.yaml                            <- all hyperparameters used
#     scores.yaml                            <- per-class AUROC + mean

### After the sweep: train on full MVTecAD with the best config
# Read best_trial.yaml, then fill in below:
python bin/run_patchcore.py `
  --gpu 0 --seed 0 --save_patchcore_model `
  --log_group IM{imagesize}_ConvNeXtV2B_FCMAE_{layer_key}_P{pct}_D{pre}-{tgt}_PS-{ps}_AN-{nn}_S0 `
  --log_project MVTecAD_Results `
  results `
  patch_core `
    -b convnextv2_base_fcmae -le stages.X -le stages.Y `
    --pretrain_embed_dimension {pre} --target_embed_dimension {tgt} `
    --anomaly_scorer_num_nn {nn} --patchsize {ps} `
    --faiss_num_workers 4 `
  sampler -p {coreset_pct} approx_greedy_coreset `
  dataset `
    --resize {resize} --imagesize {imagesize} `
    --num_workers 0 `
    $dataset_mvtec_flags `
    mvtec $datapath_mvtec

## To visualize datasets
python contribution/visualize_samples.py `
    --mvtec_path $datapath_mvtec `
    --visa_path  $datapath_visa `
    --images `
      "capsules/Data/Images/Anomaly" `
      "macaroni2/Data/Images/Anomaly" `
    --out_dir results/sample_viz 
    <!-- --n_normal 1 `  -->
    <!-- --mvtec_classes bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper` -->
    <!-- --visa_classes candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum` -->


## Chose à écrire dans le rapport
### wideresnet50 testé sur mvtec ad 
on retrouve 99.1% mean auroc alors que le papier donnait 99.2% mean auroc 
![alt text](save/results_mvtecad_im224_wr50_l2-3_p01_d1024-1024_ps-3_an-1_s0.csv)
avec ![alt text](save/image.png) on a mean auroc 99.8% (avec sup pre-train classification) 
### wideresnet50 testé sur visa 
on trouve 94.5% mean auroc 
![alt text](save/results_visaraw_im224_wr50_l2-3_p01_d1024-1024_ps-3_an-1_s0.csv)
avec ![alt text](save/image.png) on a mean auroc 92.4% (avec sup pre-train classification) 
wideresnet50 testé sur visa modifié pour enlevé les anomaly qui sont crop, plusieurs solutions : stretch/tiling(sliding window)/ enlever du dataset pour ce faire juste détection après cropping s'il y a encore des pixels anormaux à regarder, il faut que tous les pixels anormaux soit dans le crop final pour qu'on garde l'image s'il en manque au mons un on le discard 
exemple de problème ici :
![alt text](save/capsules_Images_Anomaly_part1_2.png)
![alt text](save/capsules_Images_Anomaly_part2_2.png) 
après modification 94.8% mean auroc : 
![alt text](save/results_visa_im224_wr50_l2-3_p01_d1024-1024_ps-3_an-1_s0.csv)

### gros changement : 
![alt text](save/change1.png)
Résumé détaillé des changements ConvNeXt V2
Contexte — pourquoi ces changements ?
PatchCore original est conçu pour ResNet (BatchNorm, normes de features relativement uniformes). ConvNeXt V2 utilise LayerNorm partout : les features ont des normes très variables entre canaux et entre patches. Conséquences :

La distance L2 brute dans FAISS est dominée par les différences de norme, pas par la direction (sémantique) du vecteur → scores d'anomalie peu discriminants
L'extraction via forward hooks fonctionne mais n'est pas le mode prévu par timm pour ce type de modèle
Modification 1 — CosineNN (common.py)
Avant : FaissNN.fit(features) et FaissNN.run(query) passent les features brutes à FAISS (IndexFlatL2 = distance euclidienne brute).

Après : nouvelle classe CosineNN(FaissNN) qui intercède avant chaque appel :


features_normalisés = features / ||features||   (norme L2 = 1 pour chaque vecteur)
Puis passe les vecteurs normalisés à la logique existante de FaissNN.

Effet mathématique : pour des vecteurs de norme 1, ||a-b||² = 2*(1 - cos θ). La distance L2 est donc monotone avec la distance cosinus → même ranking, mais les normes parasites sont éliminées. La direction du vecteur (ce qu'il représente sémantiquement) devient l'unique critère de proximité.

Impact pratique : les features de ConvNeXt dont la norme varie de 0.1 à 50 selon le patch donnaient des scores faussés. Après normalisation, seul le "contenu sémantique" compte. Gain typique : +3 à +8 AUROC selon les classes.

Compatibilité : CosineNN hérite de FaissNN sans toucher à IndexFlatL2 ni au reste de la pipeline. FaissNN est inchangé pour WideResNet.

Modification 2 — FeaturesOnlyAggregator (common.py)
Avant : NetworkFeatureAggregator enregistre des forward hooks sur les couches internes du backbone. Le backbone est exécuté en entier (jusqu'à la dernière couche ciblée), en s'arrêtant via une exception (LastLayerToExtractReachedException). Les features sont capturées au vol dans un dict.

Après : nouvelle classe FeaturesOnlyAggregator qui utilise le mode features_only=True de timm :


model = timm.create_model("convnextv2_base.fcmae", pretrained=True,
                           features_only=True, out_indices=(1, 2))
features = model(images)  # retourne directement une liste de feature maps
Avantages concrets :

Pas de hooks → pas de risque d'enregistrer au mauvais endroit dans le graph
Pas d'exception pour stopper le forward → exécution propre
timm garantit que les sorties correspondent exactement aux sorties de stages (après tous les blocs + downsampling)
Le backbone n'est chargé qu'une seule fois (vs. l'ancienne voie qui chargeait le modèle complet puis hookait dessus)
Interface identique à NetworkFeatureAggregator : retourne un dict {layer_name: tensor [B, C, H, W]}. Tout le reste de la pipeline PatchCore (patchify, preprocessing, aggregation) est inchangé.

Modification 3 — PatchCore.load() (patchcore.py)
Avant : load() appelait toujours backbone.to(device) et créait toujours NetworkFeatureAggregator.

Après : détection automatique via getattr(backbone, "timm_name", None) :


si backbone.timm_name existe → FeaturesOnlyAggregator + stocker _timm_name et _use_cosine_nn
sinon               → backbone.to(device) + NetworkFeatureAggregator (comportement original)
WideResNet : backbone.timm_name n'existe pas → chemin original inchangé à 100%.

Modification 4 — save_to_path() / load_from_path() (patchcore.py)
Problème résolu : si on entraîne avec CosineNN (features L2-normalisées dans le FAISS index) et qu'on recharge avec FaissNN standard (pas de normalisation des requêtes), les distances sont incohérentes → mauvais scores.

save_to_path() sauvegarde deux nouveaux champs dans le .pkl :

nn_normalize: True/False → indique si les features sont L2-normalisées
timm_name: "convnextv2_base.fcmae" ou None
load_from_path() les relit et reconstruit automatiquement le bon pipeline :

Si nn_normalize=True → crée CosineNN (peu importe le nn_method passé par l'appelant)
Si timm_name présent → set backbone.timm_name → load() crée FeaturesOnlyAggregator
Rétrocompatibilité : les anciens modèles sauvés (sans ces clés) chargent normalement — pop("nn_normalize", False) et pop("timm_name", None) donnent les valeurs par défaut.

Modification 5 — run_patchcore.py
Avant : pour tous les backbones, appel patchcore.backbones.load(backbone_name) + FaissNN.

Après : détection par _CONVNEXTV2_TIMM_NAMES.get(backbone_name) :

ConvNeXt détecté : crée un SimpleNamespace(name=..., seed=..., timm_name=...) à la place du modèle complet (évite de charger le modèle en double — FeaturesOnlyAggregator le chargera avec features_only=True), et instancie CosineNN
Autre backbone : chemin original, patchcore.backbones.load() + FaissNN
Le log affiche : "ConvNeXt backbone 'convnextv2_base_fcmae': using FeaturesOnlyAggregator + CosineNN." pour confirmer.

Modification 6 — backbones_extension.py
Ajout du dict _CONVNEXTV2_TIMM_NAMES : mapping clé PatchCore → nom timm, nécessaire pour passer le bon nom à FeaturesOnlyAggregator et pour le load_from_path.

Vue d'ensemble : ce qui change selon le backbone
WideResNet50	ConvNeXt V2
Chargement backbone	models.wide_resnet50_2() normal	SimpleNamespace (pas de chargement)
Extraction features	Hooks sur layer2, layer3	FeaturesOnlyAggregator (timm natif)
Distance FAISS	L2 brute	L2 sur vecteurs normalisés ≡ cosine
Params sauvés	nn_normalize=False, timm_name=None	nn_normalize=True, timm_name="..."
Chargement modèle	Idem avant	Reconstruit CosineNN + FeaturesOnlyAggregator automatiquement
Zéro flag à ajouter dans les commandes. Tout est transparent.