#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import scipy.io
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import argparse

"""
让数据集用eposide training的方法载入
"""
class DataLoader(Data.Dataset):
    def __init__(self, feats, atts, labels, ways=16, shots=4):
        self.ways = ways
        self.shots = shots

        self.feats = feats
        self.atts = atts
        self.labels = labels
        self.classes = np.unique(labels.numpy())

    def __getitem__(self, index):
        select_feats = torch.zeros(self.ways * self.shots, self.feats.shape[-1])
        select_atts = torch.zeros(self.ways * self.shots, self.atts.shape[-1])
        select_labels = torch.zeros(self.ways * self.shots).long()
        true_labels = torch.zeros(self.ways * self.shots).long()
        selected_classes = np.random.choice(self.classes, self.ways, False)

        for i in range(self.ways):
            idx = torch.nonzero(self.labels == selected_classes[i]).squeeze()
            select_instances = np.random.choice(idx.numpy(), self.shots, False)
            select_instances = torch.from_numpy(select_instances)
            select_feats[i * self.shots:i * self.shots + self.shots] = self.feats[select_instances]
            select_atts[i * self.shots:i * self.shots + self.shots] = self.atts[select_instances]
            select_labels[i * self.shots:i * self.shots + self.shots] = i
            true_labels[i * self.shots:i * self.shots + self.shots] = self.labels[select_instances]

        return select_feats, select_atts, select_labels, true_labels

    def __len__(self):
        return self.__size



"""
载入数据集
"""
class Dataset:
    def __init__(self, data_dir='../ZSL_data', dataset='AWA1', ways=32, shots=4, att='original_att'):
        res101 = scipy.io.loadmat(data_dir + '/' + dataset + '/res101.mat')
        att_spilts = scipy.io.loadmat(data_dir + '/' + dataset + '/att_splits.mat')
        features = res101['features'].T
        labels = res101['labels'].astype(int).squeeze() - 1

        # spilt the features and labels
        trainval_loc = att_spilts['trainval_loc'].squeeze() - 1  # minus 1 for matlab is from 1 ranther than from 0
        test_seen_loc = att_spilts['test_seen_loc'].squeeze() - 1
        test_unseen_loc = att_spilts['test_unseen_loc'].squeeze() - 1

        # convert to torch tensor
        train = torch.from_numpy(features[trainval_loc]).float()
        train_label = torch.from_numpy(labels[trainval_loc])
        test_seen = torch.from_numpy(features[test_seen_loc]).float()
        test_seen_label = torch.from_numpy(labels[test_seen_loc])
        test_unseen = torch.from_numpy(features[test_unseen_loc]).float()
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc])
        att = torch.from_numpy(att_spilts[att].T).float()

        train_att = att[train_label]

        # get the labels of seen classes and unseen classes
        seen_label = test_seen_label.unique()
        unseen_label = test_unseen_label.unique()

        # form the data as a dictionary
        self.data = {'train': train, 'train_label': train_label,
                     'test_seen': test_seen, 'test_seen_label': test_seen_label,
                     'test_unseen': test_unseen, 'test_unseen_label': test_unseen_label,
                     'att': att,
                     'train_att': train_att,
                     'seen_label': seen_label,
                     'unseen_label': unseen_label}
        self.ways = ways
        self.shots = shots
        self.feature_size = train.shape[1]
        self.att_size = att.shape[1]

    def get_loader(self, opt='train'):
        dataset = DataLoader(self.data[opt], self.data['train_att'],
                             self.data['train_label'], ways=self.ways, shots=self.shots)
        return dataset


"""
Discriminator 和 Generator 的代码一样
"""
class Discriminator(nn.Module):
    def __init__(self, hidden_size=1600, att_size=85):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(att_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2048)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, att):
        att = F.relu(self.fc1(att))
        att = F.relu(self.fc2(att))
        att = F.normalize(att, p=2, dim=-1, eps=1e-12)
        return att

class Generator(nn.Module):
    def __init__(self, hidden_size=1600, att_size=85):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(att_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2048)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)

    def forward(self, att):
        att = F.relu(self.fc1(att))
        att = F.relu(self.fc2(att))
        att = F.normalize(att, p=2, dim=-1, eps=1e-12)
        return att


"""
调整学习率的一个函数
"""
def adjust_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.1  * param_group['lr']


"""
在训练Generator时用到的采样函数，没有剔除正样本
"""
def sample(G, data, pos_att, pool_size=100, sample_size=3, g_sigma=10):
    data = data.cuda()
    data = F.normalize(data, p=2, dim=-1, eps=1e-12)
    batch_size = pos_att.shape[0]
    feature_size = data.shape[1]
    neg_pool = torch.randint(low=0, high=data.shape[0], size=[batch_size * pool_size])
    neg_pool = data[neg_pool].reshape(batch_size, pool_size, feature_size)  # (128, 100, 2048)
    pos_att = G(pos_att).unsqueeze(1)
    g_score = 20 * torch.bmm(neg_pool, pos_att.transpose(1, 2)).squeeze()
    prob = F.softmax(g_score, dim=-1)
    idx = torch.multinomial(prob, sample_size)
    # _, idx = torch.sort(prob, dim=-1, descending=True)
    # idx = idx[:, :sample_size]
    neg_prob = torch.zeros([batch_size, sample_size]).cuda()
    neg_feature = torch.zeros([batch_size, sample_size, feature_size]).cuda()
    for i in range(batch_size):
        neg_prob[i] = prob[i, idx[i]]
        neg_feature[i] = neg_pool[i, idx[i]]
    return [neg_feature, neg_prob]

"""
在训练Discriminator时用到的采样函数，剔除了正样本
"""
def neg_sample(G, dataset, pos_att, pos_label, pool_size=100, sample_size=3, g_sigma=10):
    data = dataset.data['train'].cuda()
    data = F.normalize(data, p=2, dim=-1, eps=1e-12)
    labels = dataset.data['train_label'].cuda()
    batch_size = pos_label.shape[0]
    feature_size = data.shape[1]
    rand_pool = torch.randint(low=0, high=labels.shape[0],
                              size=[2 * batch_size * pool_size])
    rand_pool = rand_pool.reshape(batch_size, 2 * pool_size)
    neg_pool = torch.zeros([batch_size, pool_size]).long().cuda()
    for i in range(batch_size):
        neg_idx = torch.nonzero(labels[rand_pool[i]] != pos_label[i])
        neg_pool[i] = rand_pool[i][neg_idx][:pool_size].squeeze()
    neg_pool = neg_pool.reshape(-1)
    neg_pool = data[neg_pool].reshape(batch_size, pool_size, feature_size)  # (128, 100, 2048)
    pos_att = G(pos_att).unsqueeze(1)
    g_score = 20 * torch.bmm(neg_pool, pos_att.transpose(1, 2)).squeeze()
    prob = torch.softmax(g_score, dim=-1)
    idx = torch.multinomial(prob, sample_size)
    # _, idx = torch.sort(g_score, dim=-1, descending=True)
    # idx = idx[:, :sample_size]
    neg_feature = torch.zeros([batch_size, sample_size, feature_size]).cuda()
    for i in range(batch_size):
        neg_feature[i] = neg_pool[i, idx[i]]
    return neg_feature


def compute_acc(net, dataset, opt1='gzsl', opt2='test_seen'):
    att = dataset.data['att'].cuda()
    if opt1 == 'gzsl':
        search_space = torch.arange(att.shape[0])
    if opt1 == 'zsl':
        search_space = dataset.data['unseen_label']
    att = att[search_space]
    feature = dataset.data[opt2].cuda()
    feature = F.normalize(feature, p=2, dim=-1, eps=1e-12)
    true_label = dataset.data[opt2 + '_label']

    with torch.no_grad():
        att_feature = net(att)
        score = torch.mm(feature , att_feature.transpose(0, 1))
        pred = torch.argmax(score, dim=-1)
        pred = search_space[pred]
        pred_label = pred.cpu().numpy()
        true_label = true_label.numpy()

        acc = 0
        unique_label = np.unique(true_label)
        for i in unique_label:
            idx = np.nonzero(true_label == i)[0]
            acc += accuracy_score(true_label[idx], pred_label[idx])
        acc = acc / unique_label.shape[0]

    return acc


def train(D, G, dataset, d_lr=0.00002, g_lr=0.00002, wt_decay=0.0001,\
          alpha=1, beta=1, pool_size=100, sample_size=3, epochs=1000, t1=10.0, t2=10.0):

    d_sigma = torch.tensor(t1, requires_grad=True, device='cuda')
    g_sigma = torch.tensor(t2, requires_grad=True, device='cuda')
    train_data = dataset.data['train']
    train_loader = dataset.get_loader('train')
    ways = train_loader.ways
    shots = train_loader.shots

    D_optimizer = optim.Adam([{'params':D.parameters()},
                              {'params': d_sigma}], lr=d_lr, weight_decay=wt_decay)

    G_optimizer = optim.Adam([{'params':G.parameters()},
                              {'params': g_sigma}], lr=g_lr, weight_decay=wt_decay)
    D_scheduler = optim.lr_scheduler.MultiStepLR(D_optimizer, milestones=[100,  500], gamma=0.5)
    G_scheduler = optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[100,  500], gamma=0.5)
    cross_entropy = nn.CrossEntropyLoss()
    D_best_zsl, G_best_zsl = 0, 0
    D_best_seen, G_best_seen = 0, 0
    D_best_unseen, G_best_unseen = 0, 0
    D_best_H, G_best_H = 0, 0

    for epoch in range(epochs):
        if epoch < 10:
            adjust_lr(D_optimizer, epoch)
            adjust_lr(G_optimizer, epoch)
        D_loss = []
        G_loss = []

        for i in range(200):
            batch_feature, batch_att, batch_label, true_label = train_loader.__getitem__(i)
            batch_feature, batch_att, batch_label, true_label = batch_feature.cuda(), batch_att.cuda(), batch_label.cuda(), true_label.cuda()
            batch_feature = F.normalize(batch_feature, p=2, dim=-1, eps=1e-12)
            idx = torch.arange(0, shots * ways, shots)
            unique_batch_att = batch_att[idx]  # (32, 85)

            ######### train the discriminator  ###########
            D_optimizer.zero_grad()
            d_att_feature = D(batch_att)
            pos_score = torch.bmm(batch_feature.unsqueeze(1), d_att_feature.unsqueeze(1).transpose(1,2)).squeeze()
            neg_feature = neg_sample(G, dataset, batch_att, true_label,
                                     pool_size=pool_size, sample_size=sample_size, g_sigma=g_sigma) #(128, 3, 2048)
            neg_score = torch.bmm(neg_feature, d_att_feature.unsqueeze(1).transpose(1,2)).squeeze()
            neg_score = torch.mean(neg_score, dim=-1) #(128)
            d_gan_loss = - torch.mean(torch.log(pos_score) + torch.log(1 - beta*neg_score))

            """
            计算discriminator的分类误差。
            整个SAN只有两个网络，判别器和生成器，并不存在一个叫分类器的网络，只是把训练集的40个85维度的语义向量
            通过判别器/生成器映射到2048维，然后计算每个feature与这40个fake feature的cosine距离，再计算交叉熵，
            就得到了分类误差。（和rethink那篇论文方式一样）
            """

            d_cls_score = d_sigma * torch.mm(batch_feature, D(unique_batch_att).transpose(0, 1))
            d_cls_loss = cross_entropy(d_cls_score, batch_label)

            d_loss = d_gan_loss + alpha * d_cls_loss #### loss由两部分组成，gan的loss 和 分类的loss
            d_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1)
            D_optimizer.step()
            D_loss.append(d_loss.item())

            ########   train the generator  ########
            G_optimizer.zero_grad()
            neg_feature, prob = sample(G, train_data, batch_att,
                                       pool_size=pool_size, sample_size=sample_size, g_sigma=g_sigma) #(128, 3, 2048), (128, 3)
            reward = torch.log(torch.bmm(neg_feature, d_att_feature.unsqueeze(1).transpose(1, 2))).squeeze()
            reward = reward - reward.mean()
            g_gan_loss = -torch.mean(torch.log(prob) * reward)

            """计算generator的分类误差 （方法和discriminator的完全一样）"""
            g_cls_score = g_sigma * torch.mm(batch_feature, G(unique_batch_att).transpose(0, 1))
            g_cls_loss = cross_entropy(g_cls_score, batch_label)

            g_loss = g_gan_loss + alpha * g_cls_loss #### loss由两部分组成，gan的loss 和 分类的loss
            g_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1)
            G_optimizer.step()
            G_loss.append(g_loss.item())

        D_scheduler.step()
        G_scheduler.step()

        ########## testing #############
        if (epoch + 1) % 1 == 0:
            print("Epoch {}/{}...".format(epoch + 1, epochs))
            D_zsl_acc = compute_acc(D, dataset, opt1='zsl', opt2='test_unseen')
            D_seen_acc = compute_acc(D, dataset, opt1='gzsl', opt2='test_seen')
            D_unseen_acc = compute_acc(D, dataset, opt1='gzsl', opt2='test_unseen')
            D_harmonic_mean = (2 * D_seen_acc * D_unseen_acc) / (D_seen_acc + D_unseen_acc)
            print("D_Loss: {:.4f}".format(np.mean(D_loss)),
                  "zsl_acc: {:.4f}".format(D_zsl_acc),
                  "seen_acc: {:.4f}".format(D_seen_acc),
                  "unseen_acc: {:.4f}".format(D_unseen_acc),
                  "harmonic_mean: {:.4f}".format(D_harmonic_mean)
                  )
            G_zsl_acc = compute_acc(G, dataset, opt1='zsl', opt2='test_unseen')
            G_seen_acc = compute_acc(G, dataset, opt1='gzsl', opt2='test_seen')
            G_unseen_acc = compute_acc(G, dataset, opt1='gzsl', opt2='test_unseen')
            G_harmonic_mean = (2 * G_seen_acc * G_unseen_acc) / (G_seen_acc + G_unseen_acc)
            print("G_Loss: {:.4f}".format(np.mean(G_loss)),
                  "zsl_acc: {:.4f}".format(G_zsl_acc),
                  "seen_acc: {:.4f}".format(G_seen_acc),
                  "unseen_acc: {:.4f}".format(G_unseen_acc),
                  "harmonic_mean: {:.4f}".format(G_harmonic_mean)
                  )
            D_best_zsl = D_zsl_acc if D_zsl_acc > D_best_zsl else D_best_zsl
            G_best_zsl = G_zsl_acc if G_zsl_acc > G_best_zsl else G_best_zsl
            if D_harmonic_mean > D_best_H:
                D_best_H = D_harmonic_mean
                D_best_seen = D_seen_acc
                D_best_unseen = D_unseen_acc
                # torch.save(D, 'D_AWA1.pkl')
            if G_harmonic_mean > G_best_H:
                G_best_H = G_harmonic_mean
                G_best_seen = G_seen_acc
                G_best_unseen = G_unseen_acc
                # torch.save(G, 'G_AWA1.pkl')

    print('D_Best_zsl： {}, D_seen: {}，D_unseen: {}, D_Best_H: {}'.format(D_best_zsl, D_best_seen, D_best_unseen,
                                                                         D_best_H))

    print('G_Best_zsl： {}, G_seen: {}，G_unseen: {}, G_Best_H: {}'.format(G_best_zsl, G_best_seen, G_best_unseen,
                                                                         G_best_H))

def main(opt):
    dataset = Dataset(data_dir=opt.data_dir, dataset=opt.dataset, ways=opt.ways, shots=opt.shots, att=opt.att)
    D = Discriminator(opt.hidden_size, dataset.att_size).cuda()
    G = Generator(opt.hidden_size, dataset.att_size).cuda()
    train(D, G, dataset, d_lr=opt.d_lr, g_lr=opt.g_lr, alpha=opt.alpha, wt_decay=opt.wt_decay, beta=opt.beta,
          pool_size=opt.pool_size, sample_size=opt.sample_size, epochs=opt.epochs, t1=opt.t1, t2=opt.t2)

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('please use GPU!')
        exit()
    parser = argparse.ArgumentParser()
    """
    实验的超参数
    """
    parser.add_argument('--dataset', default='AWA1')
    parser.add_argument('--data_dir', default='../ZSL_data')
    parser.add_argument('--att', default='original_att')
    parser.add_argument('--hidden_size', type=int, default=1600)
    parser.add_argument('--d_lr', type=float, default=0.00005)
    parser.add_argument('--g_lr', type=float, default=0.00005)
    parser.add_argument('--wt_decay', type=float, default=0.0001)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--t1', type=float, default=20.0)
    parser.add_argument('--t2', type=float, default=20.0)
    parser.add_argument('--pool_size', type=int, default=100)
    parser.add_argument('--sample_size', type=int, default=3)
    parser.add_argument('--ways', type=int, default=32)
    parser.add_argument('--shots', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)

    opt = parser.parse_args()
    main(opt)



