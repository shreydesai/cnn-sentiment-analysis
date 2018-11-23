# CNN Sentiment Analysis

Sentiment analysis using convolutional neural networks.

## Datasets

This project uses the Movie Reviews, SST-2, and MPQA datasets. These preprocessed datasets can be obtained from the [Academia Sinica NLP Lab](https://github.com/AcademiaSinicaNLPLab/sentiment_dataset). Further, we use several types of word embeddings -- Word2Vec, GloVe, and Numberbatch.

Use `python3 create_datasets.py` to build the datasets from their raw formats.

## Training

Use `train.sh` to train all the models on each dataset and embedding type. For more fine-grained training, refer to the individual flags provided in `train.py`. For example, the following command can be used to train a CNN on the Movie Reviews corpus using Word2Vec embeddings:

```python
python3 train.py --name=cnn_mr_rand --dataset=mr --epochs=20 --batch=32 --lr=1e-4 --reg=1e-3 --edims=300 --etype=w2v
```

By default, models will be checkpointed according to their validation accuracies in `checkpoints/`. The training loss, training accuracy, validation loss, and validation accuracy statistics will be logged in `stats/`. `mkdir checkpoints stats` can create these directories if they do not already exist.