# for densenet
# CUDA_VISIBLE_DEVICES=0 python main.py --epoch 300 --batch-size 64 -ct 100

# for resnet
CUDA_VISIBLE_DEVICES=1 python main.py --epoch 200 --batch-size 128 --lr 0.03 --momentum 0.9 --wd 5e-4 -ct 10 --model-num 1 --gate-type 3 --name ie-bagging-no-aug-num-1-vgg


