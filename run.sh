# for resnet
CUDA_VISIBLE_DEVICES=2 python main.py --epoch 164 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct 10 --model-num 5 --gate-type 3 --name cmcl-for-cmcl


