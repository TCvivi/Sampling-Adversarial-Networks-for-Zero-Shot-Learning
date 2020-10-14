import torch
import scipy.io
import torch.utils.data as Data
import torch.nn.functional as F


class Dataset:
    def __init__(self, data_dir='../ZSL_MINE/data/GBU', dataset='CUB', sample_size=3, pool_size=100):
        self.dataset = dataset
        self.sample_size = sample_size
        self.pool_size = pool_size

        # load the data
        res101 = scipy.io.loadmat(data_dir + '/' + dataset + '/res101.mat')
        att_spilts = scipy.io.loadmat(data_dir + '/' + dataset + '/att_splits.mat')
        features = res101['features'].T
        labels = res101['labels'].astype(int).squeeze() - 1

        # spilt the features and labels
        trainval_loc = att_spilts['trainval_loc'].squeeze() - 1  # minus 1 for matlab is from 1 ranther than from 0
        test_seen_loc = att_spilts['test_seen_loc'].squeeze() - 1
        test_unseen_loc = att_spilts['test_unseen_loc'].squeeze() - 1


        # convert to torch tensor
        train = torch.from_numpy(features[trainval_loc]).float().cuda()
        train_label = torch.from_numpy(labels[trainval_loc]).cuda()
        test_seen = torch.from_numpy(features[test_seen_loc]).float().cuda()
        test_seen_label = torch.from_numpy(labels[test_seen_loc]).cuda()
        test_unseen = torch.from_numpy(features[test_unseen_loc]).float().cuda()
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc]).cuda()
        att = torch.from_numpy(att_spilts['original_att'].T).float().cuda()
        features = torch.from_numpy(features).float().cuda()
        labels = torch.from_numpy(labels).cuda()
        # att = torch.from_numpy(att_spilts['att'].T).float().cuda()

        # get the labels of seen classes and unseen classes
        seen_label = test_seen_label.unique().cuda()
        unseen_label = test_unseen_label.unique().cuda()

        # form the data as a dictionary
        self.data = {'train': train, 'train_label': train_label,
                     'test_seen': test_seen, 'test_seen_label': test_seen_label,
                     'test_unseen': test_unseen, 'test_unseen_label': test_unseen_label,
                     'att': att, 'features': features, 'labels': labels,
                     'seen_label': seen_label,
                     'unseen_label': unseen_label}

        # add some atribute of dataset
        self.seen_class = seen_label.shape[0]
        self.unseen_class = unseen_label.shape[0]
        self.feature_size = train.shape[1]
        self.att_size = att.shape[1]

    def get_loader(self, opt='train', batch_size=32):
        data = Data.TensorDataset(self.data[opt], self.data[opt + '_label'])
        data_loader = Data.DataLoader(data, batch_size=batch_size, drop_last=True, shuffle=True)

        return data_loader

    # sample the negative samples
    def sample(self, generator, data, att):
        batch_size = att.shape[0]
        neg_pool = torch.randint(low=0, high=data.shape[0],
                                 size=[batch_size * self.pool_size])  # (data.shape[0])
        neg_pool = data[neg_pool].reshape(batch_size, self.pool_size, self.feature_size)  # （64, 100, 2048)
        repeat_att = att.unsqueeze(1).repeat([1, self.pool_size, 1])  # (64, 100, 85)
        g_score = generator(neg_pool, repeat_att)  # (64, 100)
        prob = F.softmax(g_score, dim=1)  # (64, 100)
        idx = torch.multinomial(prob, self.sample_size)  # (64, 3)
        neg_prob = torch.zeros([batch_size, self.sample_size]).cuda()  # (64, 3)
        neg_feature = torch.zeros([batch_size, self.sample_size, self.feature_size]).cuda()  # (64, 3, 2048)
        for i in range(batch_size):
            neg_prob[i] = prob[i, idx[i]]
            neg_feature[i] = neg_pool[i, idx[i]]
        return [neg_feature, neg_prob]
    
    def neg_sample(self, generator, pos_label, data, att):
        labels = self.data['train_label']
        batch_size = pos_label.shape[0]
        rand_pool = torch.randint(low=0, high=labels.shape[0],
                              size=[self.pool_size*2]) # 采集2倍于pool_size的样本，然后剔除其中的正样本。
        neg_pool = torch.zeros([batch_size, self.pool_size]).long().cuda() # (64, 100)
        for i in range(batch_size):
            neg_pool[i] = torch.nonzero(labels[rand_pool] != pos_label[i])[:100].squeeze()
        neg_pool = data[neg_pool].reshape(batch_size, self.pool_size, self.feature_size)  # （64, 100, 2048) 剔除了正样本
        repeat_att = att.unsqueeze(1).repeat([1, self.pool_size, 1])  # (64, 100, 85)
        g_score = generator(neg_pool, repeat_att)  # (64, 100)
        prob = F.softmax(g_score, dim=1)  # (64, 100)
        idx = torch.multinomial(prob, self.sample_size)  # (64, 3)
        # neg_prob = torch.zeros([batch_size, self.sample_size]).cuda()  # (64, 3)
        neg_feature = torch.zeros([batch_size, self.sample_size, self.feature_size]).cuda()  # (64, 3, 2048)
        for i in range(batch_size):
            # neg_prob[i] = prob[i, idx[i]]
            neg_feature[i] = neg_pool[i, idx[i]]
        return neg_feature






    def show_inf(self):
        print('dataset: {}, seen class: {}, unseen class {}'.format(self.dataset, self.seen_class, self.unseen_class))
        print('feature size: {}, attribute size： {}'.format(self.feature_size, self.att_size))


