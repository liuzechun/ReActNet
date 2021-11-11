clear
mkdir log
# 128 epoch setting: larger learning rate, similar performance to 256 epoch
python3 train.py --data=/datasets/imagenet --batch_size=256 --learning_rate=1.25e-3 --epochs=128 --weight_decay=1e-5 | tee -a log/training.txt
# 256 epoch setting: longer training, similar performance to 128 epoch
# python3 train.py --data=/datasets/imagenet --batch_size=256 --learning_rate=5e-4 --epochs=256 --weight_decay=1e-5 | tee -a log/training.txt
