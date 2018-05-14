# for densenet
# CUDA_VISIBLE_DEVICES=0 python main.py --epoch 300 --batch-size 64 -ct 100

# for resnet
#CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --batch-size 64 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 10 --model-num 2 --gate-type 3 --name mcl-twostep-3 --resume result/dcl-cifar-10-mcl-twostep-1/checkpoint.pth --evaluate

#CUDA_VISIBLE_DEVICES=1 python main.py --epoch 200 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name mcl-twostep-4 --evaluate --resume result/dcl-cifar-100-mn-5-gt-3-cmcl-gateonly-regular-entropy15/checkpoint.pth
#CUDA_VISIBLE_DEVICES=1 python main.py --epoch 200 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name mcl-twostep-4 --evaluate --resume result/dcl-cifar-100-mn-5-gt-3-mcl-twostep-penalty-neg-overconfident-bugfix-penaltynum-2/checkpoint.pth

#CUDA_VISIBLE_DEVICES=1 python main.py --epoch 200 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name mcl-twostep-4 --evaluate --resume result/dcl-cifar-100-mn-5-gt-3-mcl-twostep-4/checkpoint.pth

#mcl overlap 3
#CUDA_VISIBLE_DEVICES=1 python main.py --epoch 200 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name mcl-twostep-4 --evaluate --resume result/dcl-cifar-100-mn-5-gt-3-mcl-overlap3-2/checkpoint.pth

#ie
#CUDA_VISIBLE_DEVICES=1 python main.py --epoch 200 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name mcl-twostep-4 --evaluate --resume result/dcl-cifar-100-mn-5-gt-3-ie-kl/checkpoint.pth

#cmcl overlap 3
#CUDA_VISIBLE_DEVICES=1 python main.py --epoch 200 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name mcl-twostep-4 --evaluate --resume result/dcl-cifar-100-mn-5-gt-3-cmcl-overlap3-1/checkpoint.pth

#mcl overlap 2
#CUDA_VISIBLE_DEVICES=1 python main.py --epoch 200 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name mcl-twostep-4 --evaluate --resume result/dcl-cifar-100-mn-5-gt-3-mcl-overlap-2/checkpoint.pth

#mcl overlap3 fixed-uni
CUDA_VISIBLE_DEVICES=1 python main.py --epoch 200 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name unknown --evaluate --resume result/dcl-cifar-100-mn-5-gt-3-mcl-overlap3-fixed-uni/checkpoint.pth