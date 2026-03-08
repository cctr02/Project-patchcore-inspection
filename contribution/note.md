## setup conda
# Reproduit exactement l'env patchcore38 (versions figées depuis `conda env export`)

conda create -n patchcore38 python=3.8.20 -y
conda activate patchcore38

# PyTorch + CUDA (builds CUDA 11.3 officiels — ordre des channels important)
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3.1 -c pytorch -c conda-forge

# FAISS (version conda-forge, pas la version pip)
conda install -y -c conda-forge faiss-cpu=1.8.0

# Toutes les dépendances pip (versions figées)
pip install -r requirements.txt

# Outils de développement (optionnel)
pip install -r requirements_dev.txt

## To initialize on powershell
$env:PYTHONPATH="src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$datapath_mvtec = "C:\Users\trloj\Code\mvtec_anomaly_detection"

$datasets_mvtec = @('bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper')

$dataset_mvtec_flags = @()
foreach ($d in $datasets_mvtec) {
    $dataset_mvtec_flags += @('-d', $d)
}

$datapath_visa = "C:\Users\cedri\PythonProject\VisA_20220922"

$datasets_visa = @('candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum')

$dataset_visa_flags = @()
foreach ($d in $datasets_visa) {
  $dataset_visa_flags += @('-d', $d)
}

## To test datasets 
Modèles : 
- wideresnet50
- convnextv2_base_fcmae
- dinov2_vitb14_reg
- dinov2_vitl14_reg

python bin/run_patchcore.py `
  --gpu 0 --seed 0 --save_patchcore_model `
  --save_segmentation_images `
  --log_group IM448_DINOv2B14reg_L11_P01_D768-768_PS-1_AN-5_S0 `
  --log_project VisA_Results `
  results `
  patch_core `
    -b dinov2_vitb14_reg -le blocks.11 `
    --pretrain_embed_dimension 768 --target_embed_dimension 768 `
    --anomaly_scorer_num_nn 5 --patchsize 1 `
    --faiss_num_workers 4 `
  sampler -p 0.1 approx_greedy_coreset `
  dataset `
    --resize 512 --imagesize 448 `
    --num_workers 0 `
    $dataset_visa_flags `
    visa $datapath_visa
## sweep

# All settings are in contribution/sweep_configs/{study_name}.yaml
# Reprend automatiquement si interrompu (SQLite garde les trials complétés)
python contribution/sweep.py --study_name DINOv2B_VisA_Pilot2

# Tout afficher, tous les trials (PYTHONIOENCODING requis sur Windows pour les barres █)
PYTHONIOENCODING=utf-8 conda run -n patchcore38 --no-capture-output python contribution/inspect_sweep.py --study_name DINOv2B_VisA_Pilot2 --all

# Pointer directement vers un .db
PYTHONIOENCODING=utf-8 conda run -n patchcore38 --no-capture-output python contribution/inspect_sweep.py --db results/VisA_Sweep_DINOv2B_VisA_Pilot2/DINOv2B_VisA_Pilot2.db

## Aggregate_results
python contribution/aggregate_results.py

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
    --resize 256 --imagesize 224 --num_workers 0 `
    $dataset_visa_flags `
    visa $datapath_visa

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
