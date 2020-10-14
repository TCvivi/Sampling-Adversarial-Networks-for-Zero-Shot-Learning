import numpy as np
import torch
from torch import optim
import torch.nn as nn
from sklearn.metrics import accuracy_score


def train(discriminator, generator, classifier, dataset, d_lr=0.0001, g_lr=0.0001, c_lr=0.0001, batch_size=64, alpha=1.0, epochs=100):
    att = dataset.data['att']
    # new label of seen for classification loss trainning
    seen_att = att[dataset.data['seen_label']]
    seen_att = seen_att.unsqueeze(0).repeat([batch_size, 1, 1])
    new_label = torch.zeros(att.shape[0]).long().cuda()
    new_label[dataset.data['seen_label']] = torch.arange(seen_att.shape[1]).cuda() 
    # new label of unseen for classifier trainning
    unseen_att = att[dataset.data['unseen_label']]
    unseen_att = unseen_att.unsqueeze(0).repeat([batch_size, 1, 1])
    new_unseen_label = torch.zeros(att.shape[0]).long().cuda()
    new_unseen_label[dataset.data['unseen_label']] = torch.arange(unseen_att.shape[1]).cuda() 
    # feature data
    train_data = dataset.data['train']
    train_loader = dataset.get_loader('train', batch_size) 
    test_loader = dataset.get_loader('test_unseen', batch_size)
    cross_entropy = nn.CrossEntropyLoss()
    # optimizer
    D_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr, weight_decay=0.0001)
    G_optimizer = optim.Adam(generator.parameters(), lr=g_lr, weight_decay=0.0001)
    C_optimizer = optim.Adam(classifier.parameters(), lr=c_lr, weight_decay=0.0001)
    D_scheduler = optim.lr_scheduler.StepLR(D_optimizer, 80, gamma=0.5)
    G_scheduler = optim.lr_scheduler.StepLR(G_optimizer, 80, gamma=0.5)
    C_scheduler = optim.lr_scheduler.StepLR(C_optimizer, 80, gamma=0.5)
    best_zsl = 0
    best_seen = 0
    best_unseen = 0
    best_H = 0
    for epoch in range(epochs):
        print("Epoch {}/{}...".format(epoch + 1, epochs))
        D_loss = []
        G_loss = []
        C_loss = []
        for i, (features, labels) in enumerate(train_loader):

        #####   train  the discriminator  #####
            if i % 1 == 0:
                D_optimizer.zero_grad()
                _, pos_score = discriminator(features, att[labels]) # (64)
                pos_score = torch.mean(pos_score)
#                 pos_score = torch.exp(pos_score) / (1 + torch.exp(pos_score))

                neg_feature = dataset.neg_sample(generator, labels, train_data, att[labels])
                repeat_att = att[labels].unsqueeze(1).repeat([1, dataset.sample_size, 1])  # (64, sample_size, 2048)
                _, neg_score = discriminator(neg_feature, repeat_att)
                neg_score = torch.mean(neg_score)
#                 neg_score = torch.exp(neg_score) / (1 + torch.exp(neg_score))

                # the part of classification
                cls_features = features.unsqueeze(1).repeat([1, seen_att.shape[1], 1]) # (64, 40, 2048)
                _,cls_score = discriminator(cls_features, seen_att) # (64, 40)
                cls_label = new_label[labels]
                cls_loss = cross_entropy(cls_score, cls_label)

                d_loss = -(torch.log(pos_score - neg_score + 1)) + alpha * cls_loss
                # print(pos_score.item())
                # print(neg_score.item())
                # break
#                 d_loss = -(torch.log(pos_score) + torch.log(1-neg_score)) + alpha * cls_loss
                d_loss.backward()
                D_optimizer.step()
                D_scheduler.step(epoch)
                D_loss.append(d_loss.item())

        #####    train the generator #####
            G_optimizer.zero_grad()
            neg_feature, prob = dataset.sample(generator, train_data, att[labels])
            repeat_att = att[labels].unsqueeze(1).repeat([1, dataset.sample_size, 1])  # (64, sample_size, 2048)
            _,score = discriminator(neg_feature, repeat_att)
            reward = torch.log(1 + torch.exp(score))
            # reward = reward - reward.mean()
            g_loss = -torch.mean(torch.log(prob) * reward)
            g_loss.backward()
            G_optimizer.step()
            G_scheduler.step(epoch)
            G_loss.append(g_loss.item())
        
        # classifier    
        for features, labels in test_loader:
            features, labels = features.cuda(), labels.cuda()
            att_embed, _ = discriminator(features, att[labels])
            pred = classifier(att_embed)
            cls_label = new_unseen_label[labels]
            c_loss = cross_entropy(pred, cls_label)
            c_loss.backward()
            C_optimizer.step()
            C_scheduler.step(epoch)
            C_loss.append(c_loss.item())
        # test
        D_zsl_acc = compute_acc(discriminator, dataset, opt1='zsl', opt2='test_unseen')
        D_seen_acc = compute_acc(discriminator, dataset, opt1='gzsl', opt2='test_seen')
        D_unseen_acc = compute_acc(discriminator, dataset, opt1='gzsl', opt2='test_unseen')
        D_harmonic_mean = (2 * D_seen_acc * D_unseen_acc) / (D_seen_acc + D_unseen_acc)
        print("D_Loss: {:.4f}".format(np.mean(D_loss)),
              "zsl_acc: {:.4f}".format(D_zsl_acc),
              "seen_acc: {:.4f}".format(D_seen_acc),
              "unseen_acc: {:.4f}".format(D_unseen_acc),
              "harmonic_mean: {:.4f}".format(D_harmonic_mean)
              )
        G_zsl_acc = compute_gen_acc(generator, dataset, opt1='zsl', opt2='test_unseen')
        G_seen_acc = compute_gen_acc(generator, dataset, opt1='gzsl', opt2='test_seen')
        G_unseen_acc = compute_gen_acc(generator, dataset, opt1='gzsl', opt2='test_unseen')
        G_harmonic_mean = (2 * G_seen_acc * G_unseen_acc) / (G_seen_acc + G_unseen_acc)
        print("G_Loss: {:.4f}".format(np.mean(G_loss)),
              "zsl_acc: {:.4f}".format(G_zsl_acc),
              "seen_acc: {:.4f}".format(G_seen_acc),
              "unseen_acc: {:.4f}".format(G_unseen_acc),
              "harmonic_mean: {:.4f}".format(G_harmonic_mean)
              )
        C_zsl_acc = compute_cls_acc(classifier, dataset)
        print("C_Loss: {:.4f}".format(np.mean(C_loss)),
              "zsl_acc: {:.4f}".format(C_zsl_acc))
        best_zsl = D_zsl_acc if D_zsl_acc > best_zsl else best_zsl
        if D_harmonic_mean > best_H:
            best_H = D_harmonic_mean
            best_seen = D_seen_acc
            best_unseen = D_unseen_acc
    print('Best_zsl： {}, seen: {}，unseen: {}, Best_H: {}'.format(best_zsl, best_seen, best_unseen, best_H))

def train_sep(discriminator, generator, dataset, d_lr=0.0001, g_lr=0.0001, batch_size=64, alpha=1.0, epochs=100):
    att = dataset.data['att']
    seen_att = att[dataset.data['seen_label']]
    seen_att = seen_att.unsqueeze(0).repeat([batch_size, 1, 1])
    new_label = torch.zeros(att.shape[0]).long().cuda()
    new_label[dataset.data['seen_label']] = torch.arange(seen_att.shape[1]).cuda() # new label for classification trainning
    train_data = dataset.data['train']
    train_loader = dataset.get_loader('train', batch_size)
    test_loader = dataset.get_loader(opt2, batch_size)
    cross_entropy = nn.CrossEntropyLoss()
    D_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr, weight_decay=0.0001)
    G_optimizer = optim.Adam(generator.parameters(), lr=g_lr, weight_decay=0.0001)
    D_scheduler = optim.lr_scheduler.CosineAnnealingLR(D_optimizer, 30)
    G_scheduler = optim.lr_scheduler.CosineAnnealingLR(G_optimizer, 30)
    best_zsl = 0
    best_seen = 0
    best_unseen = 0
    best_H = 0
    for epoch in range(epochs):
        print("Epoch {}/{}...".format(epoch + 1, epochs))
        D_loss = []
        G_loss = []
        for features, labels in train_loader:

        #####   train  the discriminator  #####
            D_optimizer.zero_grad()
            pos_score = torch.mean(discriminator(features, att[labels]))  # (64)
            # neg_feature, prob = dataset.sample(generator, train_data, att[labels])
            neg_feature = dataset.neg_sample(generator, labels, train_data, att[labels])
            repeat_att = att[labels].unsqueeze(1).repeat([1, dataset.sample_size, 1])  # (64, sample_size, 2048)
            neg_score = torch.mean(discriminator(neg_feature, repeat_att))
            # the part of classification
            cls_features = features.unsqueeze(1).repeat([1, seen_att.shape[1], 1]) # (64, 40, 2048)
            cls_score = discriminator(cls_features, seen_att) # (64, 40)
            cls_label = new_label[labels]
            cls_loss = cross_entropy(cls_score, cls_label)
            cls_loss.backward(retain_graph=True)
            d_loss = -(torch.log(pos_score - neg_score + 0.15)) #+ alpha * cls_loss
            d_loss.backward()
            D_optimizer.step()
            D_scheduler.step(epoch)
            D_loss.append(d_loss.item())

        #####    train the generator #####
            for _ in range(1):
                G_optimizer.zero_grad()
                neg_feature, prob = dataset.sample(generator, train_data, att[labels])
                repeat_att = att[labels].unsqueeze(1).repeat([1, dataset.sample_size, 1])  # (64, sample_size, 2048)
                reward = torch.log(1 + torch.exp(discriminator(neg_feature, repeat_att)))
                # reward = reward - reward.mean()
                g_loss = -torch.mean(torch.log(prob) * reward)
                g_loss.backward()
                G_optimizer.step()
                G_scheduler.step(epoch)
                G_loss.append(g_loss.item())

        # test
        D_zsl_acc = compute_acc(discriminator, dataset, opt1='zsl', opt2='test_unseen')
        D_seen_acc = compute_acc(discriminator, dataset, opt1='gzsl', opt2='test_seen')
        D_unseen_acc = compute_acc(discriminator, dataset, opt1='gzsl', opt2='test_unseen')
        D_harmonic_mean = (2 * D_seen_acc * D_unseen_acc) / (D_seen_acc + D_unseen_acc)
        print("D_Loss: {:.4f}".format(np.mean(D_loss)),
              "zsl_acc: {:.4f}".format(D_zsl_acc),
              "seen_acc: {:.4f}".format(D_seen_acc),
              "unseen_acc: {:.4f}".format(D_unseen_acc),
              "harmonic_mean: {:.4f}".format(D_harmonic_mean)
              )
        G_zsl_acc = compute_gen_acc(generator, dataset, opt1='zsl', opt2='test_unseen')
        G_seen_acc = compute_gen_acc(generator, dataset, opt1='gzsl', opt2='test_seen')
        G_unseen_acc = compute_gen_acc(generator, dataset, opt1='gzsl', opt2='test_unseen')
        G_harmonic_mean = (2 * G_seen_acc * G_unseen_acc) / (G_seen_acc + G_unseen_acc)
        print("G_Loss: {:.4f}".format(np.mean(G_loss)),
              
              "seen_acc: {:.4f}".format(G_seen_acc),
              "unseen_acc: {:.4f}".format(G_unseen_acc),
              "harmonic_mean: {:.4f}".format(G_harmonic_mean)
              )
        best_zsl = D_zsl_acc if D_zsl_acc > best_zsl else best_zsl
        if D_harmonic_mean > best_H:
            best_H = D_harmonic_mean
            best_seen = D_seen_acc
            best_unseen = D_unseen_acc
    print('Best_zsl： {}, seen: {}，unseen: {}, Best_H: {}'.format(best_zsl, best_seen, best_unseen, best_H))



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
            _,score = discriminator(features, att)  # (B, n, 1)
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

def compute_gen_acc(generator, dataset, batch_size=128, opt1='gzsl', opt2='test_seen'):
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
            score = generator(features, att)  # (B, n, 1)
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

def compute_cls_acc(classifier, dataset, batch_size=128):
    test_loader = dataset.get_loader('test_unseen', batch_size=batch_size)
    att = dataset.data['att'].cuda()
    search_space = dataset.data['unseen_label']
    att = att[search_space].unsqueeze(0).repeat([batch_size, 1, 1])  # (B, test_class, 85)
    pred_label = []
    true_label = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.cuda(), labels.cuda()
            pred = classifier(features)
            pred = torch.argmax(pred, dim=1)
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
