'''
    bert for ner with tf2.0
    bert通过transformers加载
    自定义训练过程
    两个不同的学习率
    技巧使用：
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

import tensorflow as tf
from tensorflow import keras
from transformers import TFBertModel, BertTokenizer, BertConfig
from transformers.optimization_tf import WarmUp, AdamWeightDecay
from data.OtherUtils import load_vocab

import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=10, help='Epochs during training')
parser.add_argument('--lr', type=float, default=1.0e-5, help='Initial learing rate')
parser.add_argument('--eps', type=float, default=1.0e-6, help='epsilon')
parser.add_argument('--label_num', type=int, default=7, help='number of ner labels')
parser.add_argument('--per_save', type=int, default=int(np.ceil(6938 / 2)), help='save model per num')
parser.add_argument('--check', type=str, default='model/w2nerss', help='The path where model saved')
parser.add_argument('--mode', type=str, default='train0', help='The mode of train or predict as follows: '
                                                               'train0: begin to train or retrain'
                                                               'tran1:continue to train'
                                                               'predict: predict')
params = parser.parse_args()


def single_example_parser(serialized_example):
    sequence_features = {
        'sen': tf.io.FixedLenSequenceFeature([], tf.int64),
        'span': tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    seqlen = tf.shape(sen)[0] - 2

    span = tf.reshape(sequence_parsed['span'], [seqlen, seqlen])

    return {"sen": sen,
            "span": span,
            }


def dataloader(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=1000, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=False) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def focal_loss(y_true, y_pred, gamma=2.0):
    """
    Focal Loss 针对样本不均衡
    :param y_true: 样本标签 B*N*N
    :param y_pred: 预测值（softmax） B*N*N*n_class
    :return: focal loss
    """

    batch_size = tf.shape(y_true)[0]
    seq_len = tf.shape(y_true)[1]
    n_class = tf.shape(y_pred)[-1]

    softmax = tf.reshape(y_pred, [-1])

    labels = tf.reshape(y_true, [-1])
    labels = tf.range(0, batch_size * seq_len * seq_len) * n_class + labels

    prob = tf.gather(softmax, labels)

    weight = tf.pow(tf.subtract(1., prob), gamma)
    loss = -tf.multiply(weight, tf.math.log(prob))

    loss = tf.reshape(loss, [batch_size, seq_len, seq_len])

    return loss


class Mask(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, sen, **kwargs):
        sequencemask = tf.cast(tf.greater(sen, 0), tf.int32)

        return tf.reduce_sum(sequencemask, axis=-1) - 2


class BERT(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BERT, self).__init__(**kwargs)

        config = BertConfig.from_pretrained("hfl/chinese-roberta-wwm-ext", output_hidden_states=True)
        self.bert = TFBertModel.from_pretrained("hfl/chinese-roberta-wwm-ext", config=config)

    def call(self, inputs, **kwargs):
        return self.bert(input_ids=inputs,
                         token_type_ids=tf.zeros_like(inputs),
                         attention_mask=tf.cast(tf.greater(inputs, 0), tf.int32)
                         )[0]


class SplitSequence(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SplitSequence, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return x[:, 1:-1]


class W2NER(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(W2NER, self).__init__(**kwargs)

    def build(self, input_shape):
        self.project = keras.layers.Dense(params.label_num,
                                          kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                                          dtype=tf.float32,
                                          name='project')

        super(W2NER, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # x: B*N*768,span: B*N*N seqlen: B
        x, span, seqlen = inputs

        N = tf.shape(x)[1]

        # B*N*N*768
        x1 = tf.tile(tf.expand_dims(x, 2), [1, 1, N, 1])
        x2 = tf.transpose(x1, perm=[0, 2, 1, 3])

        # B*N*N*768
        xx = x1 * x2

        # B*N*N*7
        logits = self.project(xx)

        # B*N*N*7
        softmax = tf.nn.softmax(logits)

        # B*N*N
        loss = focal_loss(span, softmax)

        # B*N*N
        predict = tf.argmax(softmax, axis=-1, output_type=tf.int32)

        # B*N
        val = tf.sequence_mask(seqlen, maxlen=N)

        # B*N*N
        val1 = tf.tile(tf.expand_dims(val, axis=1), [1, N, 1])
        val2 = tf.tile(tf.expand_dims(val, axis=2), [1, 1, N])

        # B*N*N
        val = tf.logical_and(val1, val2)

        # B*N*N
        loss = loss * tf.cast(val, tf.float32)

        # 1
        seqlen2sum = tf.reduce_sum(tf.square(tf.cast(seqlen, tf.float32)))

        # 1
        loss = tf.reduce_sum(loss) / seqlen2sum

        # B*N*N
        predict = predict * tf.cast(val, tf.int32)

        # 1
        accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, span), tf.float32) * tf.cast(val, tf.float32)) / seqlen2sum

        ###########################################  TP TN FP  #########################################################

        # 是实体，预测是相同实体
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predict, span), tf.greater(span, 0)), tf.float32))

        # 是实体，预测不是实体或者不是相同实体
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(predict, span), tf.greater(span, 0)), tf.float32))

        # 不是实体，预测是实体
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_and(tf.greater(predict, 0),
                                                                 tf.less_equal(span, 0)),
                                                  val),
                                   tf.float32))

        return predict, tp, tn, fp, loss, accuracy

def querycheck(predict):
    sys.stdout.write('\n')
    sys.stdout.flush()
    labels = ['NONE', 'NNW', "TREATMENT", "BODY", "SIGNS", "CHECK", "DISEASE"]

    for i, pre in enumerate(predict):
        sen = sentences[i]
        print(sen)
        lensen = len(sen)

        for prepre in pre:
            print(prepre)

        for j in range(lensen):
            for k in range(0, j + 1):
                if pre[j, k] > 1:
                    print(k, ";", j, " : ", labels[pre[j, k]], " , ", sen[k:j + 1])


@tf.function(experimental_relax_shapes=True)
def train_step(data, model, optimizerbert, optimizerw2ner):
    with tf.GradientTape(persistent=True) as tape:
        _, tp, tn, fp, loss, accuracy = model(data, training=True)

    w2ner_trainable_variables = [v for v in model.trainable_variables if v.name.startswith("w2ner")]
    bert_trainable_variables = [v for v in model.trainable_variables if not v.name.startswith("w2ner")]

    gradientsw2ner = tape.gradient(loss, w2ner_trainable_variables)
    optimizerw2ner.apply_gradients(zip(gradientsw2ner, w2ner_trainable_variables))

    gradientsbert = tape.gradient(loss, bert_trainable_variables)
    optimizerbert.apply_gradients(zip(gradientsbert, bert_trainable_variables))

    return tp, tn, fp, loss, accuracy


@tf.function(experimental_relax_shapes=True)
def dev_step(data, model):
    _, tp, tn, fp, loss, accuracy = model(data, training=False)

    return tp, tn, fp, loss, accuracy


class USER:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def build_model(self, summary=True):
        sen = keras.layers.Input(shape=[None], name='sen', dtype=tf.int32)

        span = keras.layers.Input(shape=[None, None], name='span', dtype=tf.int32)

        seqlen = Mask(name="mask")(sen)

        sequence_output = BERT(name="bert")(sen)

        sequence_split = SplitSequence(name="splitsequence")(sequence_output)

        predict = W2NER(name="w2ner")(inputs=(sequence_split, span, seqlen))

        model = keras.Model(inputs=[sen, span], outputs=predict)

        if summary:
            model.summary()

        return model

    def train(self):
        history = {
            "loss": [],
            "acc": [],
            "precision": [],
            "recall": [],
            "F1": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_F1": []
        }

        model = self.build_model()

        if params.mode == 'train1':
            model.load_weights(params.check + '/w2ner.h5')

        decay_schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=params.lr,
                                                                    decay_steps=params.epochs * params.per_save,
                                                                    end_learning_rate=0.0,
                                                                    power=1.0,
                                                                    cycle=False)

        warmup_schedule = WarmUp(initial_learning_rate=params.lr,
                                 decay_schedule_fn=decay_schedule,
                                 warmup_steps=1 * params.per_save,
                                 )

        optimizerbert = AdamWeightDecay(learning_rate=warmup_schedule,
                                        weight_decay_rate=0.01,
                                        epsilon=1.0e-6,
                                        global_clipnorm=1.0,
                                        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        decay_schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=100.0 * params.lr,
                                                                    decay_steps=params.epochs * params.per_save,
                                                                    end_learning_rate=0.0,
                                                                    power=1.0,
                                                                    cycle=False)

        warmup_schedule = WarmUp(initial_learning_rate=100.0 * params.lr,
                                 decay_schedule_fn=decay_schedule,
                                 warmup_steps=1 * params.per_save,
                                 )

        optimizerw2ner = AdamWeightDecay(learning_rate=warmup_schedule,
                                         weight_decay_rate=0.01,
                                         epsilon=1.0e-6,
                                         global_clipnorm=1.0,
                                         exclude_from_weight_decay=["bias"])

        train_dataloader = dataloader(['data/TFRecordFiles/train_span.tfrecord'],
                                      single_example_parser,
                                      params.batch_size,
                                      padded_shapes={"sen": [-1],
                                                     "span": [-1, -1],
                                                     },
                                      buffer_size=100 * params.batch_size)

        dev_dataloader = dataloader(['data/TFRecordFiles/dev_span.tfrecord'],
                                    single_example_parser,
                                    params.batch_size,
                                    padded_shapes={"sen": [-1],
                                                   "span": [-1, -1],
                                                   },
                                    buffer_size=100 * params.batch_size,
                                    shuffle=False)
        F1_max = 0.0

        for epoch in range(params.epochs):
            tp = []
            tn = []
            fp = []

            loss = []
            acc = []

            loss_av = 0.0
            acc_av = 0.0
            precision = 0.0
            recall = 0.0
            F1 = 0.0

            for batch, data in enumerate(train_dataloader):
                tp_, tn_, fp_, loss_, accuracy_ = train_step(data, model, optimizerbert, optimizerw2ner)

                tp.append(tp_)
                tn.append(tn_)
                fp.append(fp_)
                loss.append(loss_)
                acc.append(accuracy_)

                loss_av = np.mean(loss)
                acc_av = np.mean(acc)

                tpsum = np.sum(tp)
                tnsum = np.sum(tn)
                fpsum = np.sum(fp)
                precision = tpsum / (tpsum + fpsum + params.eps)
                recall = tpsum / (tpsum + tnsum + params.eps)
                F1 = 2.0 * precision * recall / (precision + recall + params.eps)

                completeratio = batch / params.per_save
                total_len = 20
                rationum = int(completeratio * total_len)
                if rationum < total_len:
                    ratiogui = "=" * rationum + ">" + "." * (total_len - 1 - rationum)
                else:
                    ratiogui = "=" * total_len

                # 每一步都是显示经过累积后的各项指标，而不是每个batch的各项的指标
                print(
                    '\rEpoch %d/%d %d/%d [%s] -loss: %.6f -acc:%6.1f -precision:%6.1f -recall:%6.1f -F1:%6.1f' % (
                        epoch + 1, params.epochs, batch + 1, params.per_save,
                        ratiogui,
                        loss_av, 100.0 * acc_av,
                        100.0 * precision, 100.0 * recall, 100.0 * F1,
                    ), end=""
                )

            history["loss"].append(loss_av)
            history["acc"].append(acc_av)
            history["precision"].append(precision)
            history["recall"].append(recall)
            history["F1"].append(F1)

            loss_val, acc_val, precision_val, recall_val, F1_val = self.dev_train(dev_dataloader, model)

            print(" -val_loss: %.6f -val_acc:%6.1f -val_precision:%6.1f -val_recall:%6.1f -val_F1:%6.1f\n" % (
                loss_val, 100.0 * acc_val,
                100.0 * precision_val, 100.0 * recall_val, 100.0 * F1_val))

            history["val_loss"].append(loss_val)
            history["val_acc"].append(acc_val)
            history["val_precision"].append(precision_val)
            history["val_recall"].append(recall_val)
            history["val_F1"].append(F1_val)

            if F1_val > F1_max:
                model.save_weights(params.check + '/w2ner.h5')
                F1_max = F1_val

        with open(params.check + "/history.txt", "w", encoding="utf-8") as fw:
            fw.write(str(history))

    def dev_train(self, dev_dataloader, model):
        tp = []
        tn = []
        fp = []

        loss = []
        acc = []

        for batch, data in enumerate(dev_dataloader):
            tp_, tn_, fp_, loss_, accuracy_, = dev_step(data, model)

            tp.append(tp_)
            tn.append(tn_)
            fp.append(fp_)
            loss.append(loss_)
            acc.append(accuracy_)

        loss_av = np.mean(loss)
        acc_av = np.mean(acc)

        tp_sum = np.sum(tp)
        tn_sum = np.sum(tn)
        fp_sum = np.sum(fp)

        precision = tp_sum / (tp_sum + fp_sum + params.eps)
        recall = tp_sum / (tp_sum + tn_sum + params.eps)
        F1 = 2.0 * precision * recall / (precision + recall + params.eps)

        return loss_av, acc_av, precision, recall, F1

    def predict(self):
        model = self.build_model(summary=False)
        model.load_weights(params.check + '/w2ner.h5')

        BN = tf.shape(sent)
        B, N = BN[0], BN[1] - 2

        predict, _, _, _, _, _ = model.predict([sent, tf.ones([B, N, N], tf.int32)], verbose=0)

        querycheck(predict)

    def test(self):
        model = self.build_model()
        model.load_weights(params.check + '/w2ner.h5')

        dev_data = dataloader(['data/TFRecordFiles/dev_span.tfrecord'],
                              single_example_parser,
                              1,
                              padded_shapes={"sen": [-1],
                                             "span": [-1, -1],
                                             },
                              buffer_size=100 * params.batch_size,
                              shuffle=False)

        labels = ['NONE', 'NNW', "TREATMENT", "BODY", "SIGNS", "CHECK", "DISEASE"]

        tp, tn, fp = 0.0, 0.0, 0.0

        fw = open(params.check + "/log.txt", "w", encoding="utf-8")

        for batch, data in enumerate(dev_data):
            print("\r%d" % (batch), end="")

            sen = data["sen"][0][1:-1]
            span = data["span"][0]

            fw.write("句子：\n")
            fw.write("".join([char_inverse_dict[s] for s in sen.numpy()]) + "\n\n")

            fw.write("span_real：\n")
            span = span.numpy()

            for i in range(len(span)):
                for j in range(i + 1):
                    if span[i, j] > 0:
                        fw.write("%d;%d;%s\n" % (j, i, labels[span[i, j]]))
            fw.write("\n")

            span_predict_, tp_, tn_, fp_, _, _ = model.predict(data, verbose=0)

            fw.write("span_pred：\n")
            for i in range(len(span)):
                for j in range(i + 1):
                    if span_predict_[0, i, j] > 0:
                        fw.write("%d;%d;%s\n" % (j, i, labels[span_predict_[0, i, j]]))

            fw.write("\n")

            fw.write("TP: %d TN: %d FP: %d\n" % (tp_, tn_, fp_))
            fw.write("----------------------------------------------\n")

            tp += tp_
            tn += tn_
            fp += fp_

        precision = tp / (tp + fp + params.eps)
        recall = tp / (tp + tn + params.eps)
        F1 = 2.0 * precision * recall / (precision + recall + params.eps)

        fw.write('\nprecision: %.4f recall: %.4f F1: %.4f\n\n' % (precision, recall, F1))

        sys.stdout.write('\nprecision: %.4f recall: %.4f F1: %.4f\n\n' % (precision, recall, F1))
        sys.stdout.flush()

    def plothistory(self):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        with open(params.check + "/history.txt", "r", encoding="utf-8") as fr:
            history = fr.read()
            history = eval(history)

        gs = gridspec.GridSpec(2, 6)
        plt.subplot(gs[0, 1:3])
        plt.plot(history["loss"])
        plt.plot(history["val_loss"])
        plt.grid()
        plt.title('loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='best', prop={'size': 4})

        plt.subplot(gs[0, 3:5])
        plt.plot(history["acc"])
        plt.plot(history["val_acc"])
        plt.grid()
        plt.title('acc')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='best', prop={'size': 4})

        plt.subplot(gs[1, :2])
        plt.plot(history["precision"])
        plt.plot(history["val_precision"])
        plt.grid()
        plt.title('precision')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='best', prop={'size': 4})

        plt.subplot(gs[1, 2:4])
        plt.plot(history["recall"])
        plt.plot(history["val_recall"])
        plt.grid()
        plt.title('recall')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='best', prop={'size': 4})

        plt.subplot(gs[1, 4:])
        plt.plot(history["F1"])
        plt.plot(history["val_F1"])
        plt.grid()
        plt.title('F1')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='best', prop={'size': 4})

        plt.suptitle("Model Metrics")

        plt.tight_layout()
        plt.savefig("w2nerss_PRF.jpg", dpi=500, bbox_inches="tight")


if __name__ == '__main__':
    if not os.path.exists(params.check):
        os.makedirs(params.check)

    user = USER()

    if params.mode == "plot":
        user.plothistory()
    else:
        ner_dict = {
            'NONE': 0,
            'NNW': 1,
            'TREATMENT-B': 2,
            'BODY-B': 3,
            'SIGNS-B': 4,
            'CHECK-B': 5,
            'DISEASE-B': 6
        }
        ner_inverse_dict = {v: k for k, v in ner_dict.items()}

        char_dict = load_vocab("data/OriginalFiles/vocab.txt")
        char_inverse_dict = {v: k for k, v in char_dict.items()}

        sentences = [
            '左侧粗隆间骨折。',
            '心肺腹查体未见异常。',
            '主因右髋部摔伤后疼痛肿胀。',
            '入院后完善各项检查，给予右下肢持续皮牵引。'
        ]
        # sentences = [
        #     '1.患者老年女性，88岁；2.既往体健，否认药物过敏史。3.患者缘于5小时前不慎摔伤，伤及右髋部。伤后患者自感伤处疼痛，呼我院120接来我院，查左髋部X光片示：左侧粗隆间骨折。',
        #     '患者精神状况好，无发热，诉右髋部疼痛，饮食差，二便正常，查体：神清，各项生命体征平稳，心肺腹查体未见异常。',
        #     '女性，88岁，农民，双滦区应营子村人，主因右髋部摔伤后疼痛肿胀，活动受限5小时于2016-10-29；11：12入院。',
        #     '入院后完善各项检查，给予右下肢持续皮牵引，应用健骨药物治疗，患者略发热，查血常规：白细胞数12.18*10^9/L，中性粒细胞百分比92.00%。',
        #     '1患者老年男性，既往有高血压病史5年，血压最高达180/100mmHg，长期服用降压药物治疗，血压控制欠佳。'
        # ]

        m_samples = len(sentences)

        sent = []
        leng = []
        for sentence in sentences:
            sentence = sentence.lower()
            leng.append(len(sentence))

            sen2id = [char_dict['[CLS]']] + [
                char_dict[word] if word in char_dict.keys() else char_dict['[UNK]']
                for word in sentence.lower()] + [char_dict['[SEP]']]
            sent.append(sen2id)

        max_len = np.max(leng)
        for i in range(m_samples):
            if leng[i] < max_len:
                pad = [char_dict['[PAD]']] * (max_len - leng[i])
                sent[i] += pad

        sent = tf.constant(sent)

        if params.mode.startswith('train'):
            user.train()

            user.plothistory()
        elif params.mode == "test":
            user.test()
        elif params.mode == "predict":
            user.predict()
