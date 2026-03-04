## To initialize on powershell
$env:PYTHONPATH="src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

## To test datasets 
### (without gpu) : 
python bin/run_patchcore.py `
  --gpu -1 --seed 0 --save_patchcore_model `
  --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 `
  --log_project MVTecAD_Results `
  results `
  patch_core `
    -b wideresnet50 -le layer2 -le layer3 `
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 `
    --anomaly_scorer_num_nn 1 --patchsize 3 `
    --faiss_num_workers 0 `
  sampler -p 0.1 approx_greedy_coreset `
  dataset `
    --resize 256 --imagesize 224 `
    --num_workers 0 `
    -d bottle -d cable -d capsule -d carpet -d grid -d hazelnut `
    -d leather -d metal_nut -d pill -d screw -d tile -d toothbrush `
    -d transistor -d wood -d zipper `
    mvtec C:/Users/cedri/PythonProject/mvtec_anomaly_detection



###########
python bin/run_patchcore.py `
  --gpu -1 --seed 0 --save_patchcore_model `
  --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 `
  --log_project VisA_Results `
  results `
  patch_core `
    -b wideresnet50 -le layer2 -le layer3 `
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 `
    --anomaly_scorer_num_nn 1 --patchsize 3 `
    --faiss_num_workers 0 `
  sampler -p 0.1 approx_greedy_coreset `
  dataset `
    --resize 256 --imagesize 224 --train_val_split 0.9 `
    --num_workers 0 `
    -d candle -d capsules -d cashew -d chewinggum -d fryum `
    -d macaroni1 -d macaroni2 -d pcb1 -d pcb2 -d pcb3 -d pcb4 -d pipe_fryum `
    visa C:/Users/cedri/PythonProject/VisA_20220922

### (with gpu) :
python bin/run_patchcore.py `
  --gpu 0 --seed 0 --save_patchcore_model `
  --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 `
  --log_project MVTecAD_Results `
  results `
  patch_core `
    -b wideresnet50 -le layer2 -le layer3 `
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 `
    --anomaly_scorer_num_nn 1 --patchsize 3 `
    --faiss_on_gpu --faiss_num_workers 4 `
  sampler -p 0.1 approx_greedy_coreset `
  dataset `
    --resize 256 --imagesize 224 `
    --num_workers 0 `
    -d bottle -d cable -d capsule -d carpet -d grid -d hazelnut `
    -d leather -d metal_nut -d pill -d screw -d tile -d toothbrush `
    -d transistor -d wood -d zipper `
    mvtec C:/Users/cedri/PythonProject/mvtec_anomaly_detection

##############
python bin/run_patchcore.py `
  --gpu 0 --seed 0 --save_patchcore_model `
  --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 `
  --log_project VisA_Results `
  results `
  patch_core `
    -b wideresnet50 -le layer2 -le layer3 `
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 `
    --anomaly_scorer_num_nn 1 --patchsize 3 `
    --faiss_on_gpu --faiss_num_workers 4 `
  sampler -p 0.1 approx_greedy_coreset `
  dataset `
    --resize 256 --imagesize 224 --train_val_split 0.9 `
    --num_workers 0 `
    -d candle -d capsules -d cashew -d chewinggum -d fryum `
    -d macaroni1 -d macaroni2 -d pcb1 -d pcb2 -d pcb3 -d pcb4 -d pipe_fryum `
    visa C:/Users/cedri/PythonProject/VisA_20220922
