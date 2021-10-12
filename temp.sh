set -ex
CUDA_VISIBLE_DEVICES=0,1 python data/retrieval_workshop_baseline_dataset.py --dataroot dataset/train_data --name workshop_baseline_notexture_pretrain_v1 --model retrieval_workshop_baseline_pretrain --dataset_mode retrieval_workshop_baseline --niter 10 --niter_decay 20 --crop_size 256 --fine_size 256 --num_threads 8 --lr 0.01 --batch_size 32  --gpu_ids 0,1 


