import numpy as np
import math

from consts import NONE, PAD, TRIGGERS_WEIGHTS, ARGUMENTS_WEIGHTS


def build_vocab(labels, BIO_tagging=True):
    all_labels = [PAD, NONE]
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label


def get_trigger_loss_weights(triggers):
    if triggers[2].startswith('B-'):
        w = TRIGGERS_WEIGHTS[0:2]
        for i in range(2, len(TRIGGERS_WEIGHTS)):
            w.append(TRIGGERS_WEIGHTS[i])
            w.append(TRIGGERS_WEIGHTS[i])
    else:
        w = TRIGGERS_WEIGHTS

    w = [math.sqrt(x) for x in w]
    s = sum(w)
    w = [x / s for x in w]
    w = [x * len(w) * 50 for x in w]
    return w


def get_arg_loss_weights(arguments):
    if arguments[2].startswith('B-'):
        nw = ARGUMENTS_WEIGHTS[0:2]
        for i in range(2, len(ARGUMENTS_WEIGHTS)):
            nw.append(ARGUMENTS_WEIGHTS[i])
            nw.append(ARGUMENTS_WEIGHTS[i])
    else:
        nw = ARGUMENTS_WEIGHTS

    pw = [1 - x for x in nw]
    pw = [math.sqrt(x) for x in pw]
    nw = [math.sqrt(x) for x in nw]
    sw = [p + n for p, n in zip(pw, nw)]
    pw = [p / s for p, s in zip(pw, sw)]
    return pw


def calc_metric(y_true, y_pred):
    """
    :param y_true: [(tuple), ...]
    :param y_pred: [(tuple), ...]
    :return:
    """
    num_proposed = len(y_pred)
    num_gold = len(y_true)

    y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true_set:
            num_correct += 1

    print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 1.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 1.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1


def find_triggers(labels):
    """
    :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
    """
    result = []
    labels = [label.split('-') for label in labels]

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result.append([i, i + 1, labels[i][1]])

    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I':
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result]


# To watch performance comfortably on a telegram when training for a long time
def report_to_telegram(text, bot_token, chat_id):
    try:
        import requests
        requests.get('https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}'.format(bot_token, chat_id, text))
    except Exception as e:
        print(e)
