import tensorflow as tf
from OtherUtils import load_vocab
from tqdm import tqdm
import numpy as np
import json


def medicalw2ner(filepath, train_tfrecordfilepath, dev_tfrecordfilepath):
    char_dict = load_vocab("OriginalFiles/vocab.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        all_datas = json.load(f)

    writer_train = tf.io.TFRecordWriter(train_tfrecordfilepath)
    writer_dev = tf.io.TFRecordWriter(dev_tfrecordfilepath)

    label_dict = {
        'NONE': 0,
        'NNW': 1,
        'TREATMENT': 2,
        'BODY': 3,
        'SIGNS': 4,
        'CHECK': 5,
        'DISEASE': 6
    }

    m_samples_train = 0
    m_samples_dev = 0

    for data in tqdm(all_datas):
        text = data["context"]
        seqlen = len(text)

        if seqlen > 150:
            continue

        sent2id = [101]
        sent2id += [char_dict.get(char, char_dict["[UNK]"]) for char in text]
        sent2id += [102]

        spanMatrix = np.zeros([seqlen, seqlen], np.int32)

        for se, label in data["span_posLabel"].items():
            startid, endid = se.split(";")
            spanMatrix[int(endid), int(startid)] = label_dict[label]
            for id in range(int(startid), int(endid)):
                spanMatrix[id, id + 1] = label_dict['NNW']

        sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[t])) for t in sent2id]

        span_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[t])) for t in spanMatrix.flatten()]

        seq_example = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(feature_list={
                'sen': tf.train.FeatureList(feature=sen_feature),
                'span': tf.train.FeatureList(feature=span_feature),
            })
        )

        serialized = seq_example.SerializeToString()

        if np.random.random() > 0.1:
            writer_train.write(serialized)
            m_samples_train += 1
        else:
            writer_dev.write(serialized)
            m_samples_dev += 1

    print("训练集样本数：", m_samples_train)
    print("验证集样本数：", m_samples_dev)


if __name__ == "__main__":
    medicalw2ner(
        "OriginalFiles/train_span.txt",
        "TFRecordFiles/train_span.tfrecord",  # 7016
        "TFRecordFiles/dev_span.tfrecord",  # 730
    )
