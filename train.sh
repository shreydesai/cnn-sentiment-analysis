for dataset in mr sst mpqa; do
    echo Training ${dataset} dataset
    for embedding in random w2v glove nb; do
        python3 train.py --name=cnn_${dataset}_${embedding} --dataset=${dataset} --epochs=30 --batch=32 --lr=1e-4 --reg=0 --edims=300 --etype=${embedding}
    done
done

sudo shutdown