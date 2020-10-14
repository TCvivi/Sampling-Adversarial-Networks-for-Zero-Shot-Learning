# encoding: utf-8
import torch
import numpy as np
from sklearn.metrics import accuracy_score


def compute_acc(discriminator, dataset, batch_size=128, opt1='gzsl', opt2='test_seen'):
    test_loader = dataset.get_loader(opt2, batch_size=batch_size)
    att = dataset.data['att'].cuda()
    if opt1 == 'gzsl':
        search_space = torch.arange(att.shape[0])
    if opt1 == 'zsl':
        search_space = dataset.data['unseen_label']
    att = att[search_space].unsqueeze(0).repeat([batch_size, 1, 1])  # (B, test_class, 85)
    pred_label = []
    true_label = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.cuda(), labels.cuda()

            features = features.unsqueeze(1)
            features = features.repeat(1, search_space.shape[0], 1)
            score = discriminator(features, att)  # (B, n, 1)
            pred = torch.argmax(score, dim=1)
            pred = search_space[pred]
            pred_label = np.append(pred_label, pred.cpu().numpy())
            true_label = np.append(true_label, labels.cpu().numpy())

    pred_label = np.array(pred_label, dtype='int')
    true_label = np.array(true_label, dtype='int')
    acc = 0

    unique_label = np.unique(true_label)
    for i in unique_label:
        idx = np.nonzero(true_label == i)[0]
        acc += accuracy_score(true_label[idx], pred_label[idx])
    acc = acc / unique_label.shape[0]

    return acc


# topk test
# def topk(model, dataset, opt='unseen'):
#     att = dataset.data['att']
#     if opt == 'unseen':
#         test_data = dataset.data['test_unseen']
#         label = dataset.data['test_unseen_label']  # (5685, 1)
#         test_class = dataset.data['unseen_label']  # 10
#     if opt == 'seen':
#         test_data = dataset.data['test_seen']
#         label = dataset.data['test_seen_label']
#         test_class = dataset.data['seen_label']
#
#     att_num = test_class.shape[0]  # the num of test attribute
#     top1, top10, top50, top100 = 0, 0, 0, 0
#     for class_i in test_class:
#         att_i = att[class_i].repeat(test_data.shape[0], 1)  # (5685, 85)
#         score = model(test_data, att_i)  # (5685, 1)
#         sorted_score, idx = torch.sort(score, dim=0, descending=True)
#         top1 = top1 + (label[idx[0]] == class_i).sum().item()
#         top10 = top10 + (label[idx[:10]] == class_i).sum().item() / 10
#         top50 = top50 + (label[idx[:50]] == class_i).sum().item() / 50
#         top100 = top100 + (label[idx[:100]] == class_i).sum().item() / 100
#
#     return [top1 / att_num, top10 / att_num, top50 / att_num, top100 / att_num]

# def recallK(model, dataset, opt='unseen'):
#     att = dataset.data['att']
#     if opt == 'unseen':
#         test_data = dataset.data['test_unseen']
#         label = dataset.data['test_unseen_label']  # (5685, 1)
#         test_class = dataset.data['unseen_label']  # 10
#     if opt == 'seen':
#         test_data = dataset.data['test_seen']
#         label = dataset.data['test_seen_label']
#         test_class = dataset.data['seen_label']
#
#     att_num = test_class.shape[0]  # the num of test classes
#     r1, r5, r10, r20, r50 = 0, 0, 0, 0, 0
#     for class_i in test_class:
#         att_i = att[class_i].repeat(test_data.shape[0], 1)  # (5685, 85)
#         score = model(test_data, att_i)  # (5685, 1)
#         sorted_score, idx = torch.sort(score, dim=0, descending=True)
#         r1 = r1 + bool((label[idx[0]] == class_i).sum().item())
#         r5 = r5 + bool((label[idx[:5]] == class_i).sum().item())
#         r10 = r10 + bool((label[idx[:10]] == class_i).sum().item())
#         r20 = r20 + bool((label[idx[:20]] == class_i).sum().item())
#         r50 = r50 + bool((label[idx[:50]] == class_i).sum().item())
#     return [r1 / att_num, r5 / att_num, r10 / att_num, r20 / att_num, r50 / att_num]
#
