# for densenet
# CUDA_VISIBLE_DEVICES=0 python main.py --epoch 300 --batch-size 64 -ct 100

# for resnet
#CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --batch-size 64 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 10 --model-num 2 --gate-type 3 --name mcl-twostep-3 --resume result/dcl-cifar-10-mcl-twostep-1/checkpoint.pth --evaluate

#CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --batch-size 64 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name unknwon

#cifar10 - gateonly - based on mcl
#CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --batch-size 64 --lr 0.03 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name mcl-gateonly-feature-as-input --resume result/dcl-cifar-100-mn-5-gt-3-mcl-twostep-6/checkpoint.pth

#CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --batch-size 128 --lr 0.03 --momentum 0.9 --wd 1e-4 -ct 10 --model-num 5 --gate-type 3 --name mcl-gateonly-regular-1 --resume result/dcl-cifar-10-mn-5-gt-3-cmcl-for-cmcl/checkpoint.pth

#CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --batch-size 128 --lr 0.03 --momentum 0.9 --wd 1e-4 -ct 10 --model-num 5 --gate-type 3 --name mcl-gateonly-regular-1 --resume result/dcl-cifar-10-mn-5-gt-3-ie-for-cmcl/checkpoint.pth


CUDA_VISIBLE_DEVICES=2 python main.py --epoch 200 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name mcl-gates-together-overlap3

