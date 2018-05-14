# for densenet
# CUDA_VISIBLE_DEVICES=0 python main.py --epoch 300 --batch-size 64 -ct 100

# for resnet
#CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --batch-size 64 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 10 --model-num 2 --gate-type 3 --name mcl-twostep-3 --resume result/dcl-cifar-10-mcl-twostep-1/checkpoint.pth --evaluate

CUDA_VISIBLE_DEVICES=1 python main.py --epoch 200 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name mcl-focal-loss-not-detach