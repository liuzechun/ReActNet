clear
python3 train.py --data=/datasets/imagenet --batch_size=256 --learning_rate=5e-4 --epochs=256 --weight_decay=1e-5 | tee -a log/training.txt
