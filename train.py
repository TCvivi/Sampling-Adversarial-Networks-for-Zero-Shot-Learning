import numpy as np
import torch
from torch import optim
import torch.nn as nn

def train_disciminator(discriminator, generator, dataset, lr=0.0001, batch_size=32, epoch=0):
    att = dataset.data['att']
    train_data = dataset.data['train']
    train_loader = dataset.get_loader('train', batch_size)
    optimizer = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    loss_list = []

    for features, labels in train_loader:
        optimizer.zero_grad()

        # calculate the positive score
        pos_socre = torch.mean(discriminator(features, att[labels]))

        # calculate the negative score
        neg_feature, _prob = dataset.sample(generator, train_data, att[labels])  # (64, 3, 2048)
        repeat_att = att[labels].unsqueeze(1).repeat([1, dataset.sample_size, 1])
        neg_score = torch.mean(discriminator(neg_feature, repeat_att))

        d_loss = -(torch.log(pos_socre - neg_score + 2))
        d_loss.backward()
        optimizer.step()
        scheduler.step(epoch)

        loss_list.append(d_loss.item())
    return np.mean(loss_list)


def train_generator(discriminator, generator, dataset, lr=0.0001, batch_size=32, epoch=0):
    att = dataset.data['att']
    train_data = dataset.data['train']
    train_loader = dataset.get_loader('train', batch_size)
    optimizer = optim.Adam(generator.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    loss_list = []

    for features, labels in train_loader:
        optimizer.zero_grad()

        neg_feature, prob = dataset.sample(generator, train_data, att[labels])  # (64, 3, 2048)
        repeat_att = att[labels].unsqueeze(1).repeat([1, dataset.sample_size, 1])
        reward = torch.log(1 + torch.exp(discriminator(neg_feature, repeat_att)))
        g_loss = -torch.mean(torch.log(prob) * reward)

        g_loss.backward()
        optimizer.step()
        scheduler.step(epoch)

        loss_list.append(g_loss.item())
    return np.mean(loss_list)

def train_together(discriminator, generator, dataset, lr=0.0001, batch_size=32, epoch=0):
    att = dataset.data['att']
    seen_att = att[dataset.data['seen_label']]
    new_label = torch.zeros(att.shape[0]).long().cuda()
    new_label[dataset.data['seen_label']] = torch.arange(seen_att.shape[0]).cuda()
    seen_att = seen_att.unsqueeze(0).repeat([batch_size, 1, 1])
    train_data = dataset.data['train']
    train_loader = dataset.get_loader('train', batch_size)

    cross_entropy = nn.CrossEntropyLoss()
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=0.001)
    G_optimizer = optim.Adam(generator.parameters(), lr=lr, weight_decay=0.001)
    D_loss_list = []
    G_loss_list = []

    for features, labels in train_loader:
        # train the discriminator
        D_optimizer.zero_grad()
        pos_score = torch.mean(discriminator(features, att[labels]))  # (64)

        # calculate the negative score
        neg_feature = dataset.neg_sample(generator, labels, train_data, att[labels])  # (64, 3, 2048)

        repeat_att = att[labels].unsqueeze(1).repeat([1, dataset.sample_size, 1])
        neg_score = discriminator(neg_feature, repeat_att)
        neg_score = torch.mean(neg_score)

        d_loss = -(torch.log(pos_score - neg_score + 0.2))

        features = features.unsqueeze(1).repeat([1, seen_att.shape[1], 1])  # (32, 40, 2048)
        # sigma = torch.tensor([10]).float().cuda()
        # cos_score = sigma*F.cosine_similarity(classifier, features, dim=-1) # (32, 40)
        cos_score = discriminator(features, seen_att)
        labels = new_label[labels]
        cls_loss = cross_entropy(cos_score, labels)
        d_loss = d_loss + 0.5*cls_loss
        d_loss.backward()
        D_optimizer.step()
        D_loss_list.append(d_loss.item())

        # train the generator
        for i in range(2):
            G_optimizer.zero_grad()

            neg_feature, prob = dataset.sample(generator, train_data, att[labels])  # (64, 3, 2048)
            repeat_att = att[labels].unsqueeze(1).repeat([1, dataset.sample_size, 1])
            reward = torch.log(1 + torch.exp(discriminator(neg_feature, repeat_att)))
            reward = reward - reward.mean()
            g_loss = -torch.mean(torch.log(prob) * reward) # - torch.mean(reward[:, 0] - reward[:, 1])

            g_loss.backward()
            G_optimizer.step()
            G_loss_list.append(g_loss.item())

    return [np.mean(D_loss_list), np.mean(G_loss_list)]

def train(discriminator, generator, dataset, d_lr=0.0001, g_lr=0.0001, batch_size=64, epochs=0):
    att = dataset.data['att']
    train_data = dataset.data['trian']
    train_loader =  dataset.get_loader('train', batch_size)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr, weight_decay=0.0001)
    G_optimizer = optim.Adam(generator.parameters(), lr=g_lr, weight_decay=0.0001)
    D_scheduler = optim.lr_scheduler.CosineAnnealingLR(D_optimizer, 20)
    G_scheduler = optim.lr_scheduler.CosineAnnealingLR(G_optimizer, 20)

    for epoch in range(epochs):
        D_loss = []
        G_loss = []
        for features, labels in train_loader:
            D_optimizer.zero_grad()
            G_optimizer.zero_grad()

            # sample the negfeature
            neg_feature, prob = dataset.sample(generator, train_data, att[labels])
            repeat_att = att[labels].unsqueeze(1).repeat([1, dataset.sample_size, 1])#(64, sample_size, 2048)

            #####   train  the discriminator  #####
            pos_score = torch.mean(discriminator(features, att[labels])) #(64)
            neg_score = torch.mean(discriminator(neg_feature, repeat_att))
            d_loss  = -(torch.log(pos_score - neg_score + 0.1))
            d_loss.backward()
            D_optimizer.step()
            D_scheduler.step(epoch)
            D_loss.append(d_loss.item())

            #####    train the generator #####
            reward = torch.log(1 + torch.exp(discriminator(neg_feature, repeat_att)))
            g_loss = -torch.mean(torch.log(prob) * reward)
            g_loss.backward()
            G_optimizer.step()
            G_scheduler.step(epoch)
            G_loss.append(g_loss.item())

        print("Epoch {}/{}...".format(epoch + 1, epochs))
