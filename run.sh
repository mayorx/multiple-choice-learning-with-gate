# for densenet
# CUDA_VISIBLE_DEVICES=0 python main.py --epoch 300 --batch-size 64 -ct 100

# for resnet
#CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --batch-size 64 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 10 --model-num 2 --gate-type 3 --name mcl-twostep-3 --resume result/dcl-cifar-10-mcl-twostep-1/checkpoint.pth --evaluate

#CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --batch-size 64 --lr 0.1 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name unknwon

#CUDA_VISIBLE_DEVICES=2 python main.py --epoch 200 --batch-size 128 --lr 0.03 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name cmcl-gateonly-regular-entropy15 --resume result/dcl-cifar-100-mn-5-gt-3-mcl-entropy-15/checkpoint.pth
#CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --batch-size 5 --lr 0.03 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name mcl-gateonly-gateentropy --resume result/dcl-cifar-100-mn-5-gt-3-mcl-twostep-penalty-neg-overconfident-bugfix-penaltynum-3/checkpoint-epoch-0.pth

#mcl + overlap3
#CUDA_VISIBLE_DEVICES=2 python main.py --epoch 200 --batch-size 128 --lr 0.03 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name mcl-overlap3-3 --resume result/dcl-cifar-100-mn-5-gt-3-mcl-twostep-4/checkpoint.pth

#ie
#CUDA_VISIBLE_DEVICES=1 python main.py --epoch 200 --batch-size 128 --lr 0.03 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name ie-kl --resume result/dcl-cifar-100-mn-5-gt-3-ie3/checkpoint.pth

#cmcl + overlap3
#CUDA_VISIBLE_DEVICES=2 python main.py --epoch 200 --batch-size 128 --lr 0.03 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name cmcl-overlap3-1 --resume result/dcl-cifar-100-mn-5-gt-3-mcl-entropy-17/checkpoint.pth

#mcl (regular)
#CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --batch-size 128 --lr 0.03 --momentum 0.9 --wd 5e-4 -ct 100 --model-num 5 --gate-type 3 --name mcl-regular-1 --resume result/dcl-cifar-100-mn-5-gt-3-mcl-twostep-penalty-neg-overconfident-bugfix-penaltynum-3/checkpoint-epoch-0.pth



