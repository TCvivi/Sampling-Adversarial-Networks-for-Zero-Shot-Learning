# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):  # the structure is artotally same with Generator
    def __init__(self, feature_size=2048, att_size=85, t1=10.0):
        super(Discriminator, self).__init__()
        self.sigma = nn.Parameter(torch.Tensor([t1]), requires_grad=True)
        self.fc1 = nn.Linear(att_size, 1600)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(1600, feature_size)
    def forward(self, feature, att):  # feature: , att:
        att_embed = torch.relu(self.fc1(att))
        att_embed = torch.relu(self.fc2(att_embed))
        # score = F.cosine_similarity(att, feature, dim=-1)
        score = self.sigma * F.cosine_similarity(att_embed, feature, dim=-1)

        return att_embed, score

class Generator(nn.Module):
    def __init__(self, feature_size=2048, att_size=85, t2=10.0):
        super(Generator, self).__init__()
        self.sigma = nn.Parameter(torch.Tensor([t2]), requires_grad=True)
        self.fc1 = nn.Linear(att_size, 1600)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(1600, feature_size)

    def forward(self, feature, att):  # feature:, att:
        att = torch.relu(self.fc1(att))
        att = torch.relu(self.fc2(att))
        # score = F.cosine_similarity(att, feature, dim=-1)
        score = self.sigma * F.cosine_similarity(att, feature, dim=-1)
        return score
    
class Classifier(nn.Module):
    def __init__(self, feature_size=2048, nclass=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(feature_size, 1024)
        self.fc2 = nn.Linear(1024, nclass)

    def forward(self, feature):  # feature:, att:
        feature = torch.relu(self.fc1(feature))
        feature = torch.sigmoid(self.fc2(feature))
        pred = F.softmax(feature, dim=1)
        return pred
    
# class Classifier_GZSL(nn.Module):
#     def __init__(self, feature_size=2048, nclass=50):
#         super(Classifier_GZSL, self).__init__()
#         self.fc1 = nn.Linear(feature_size, 1024)
#         self.fc2 = nn.Linear(1024, nclass)

#     def forward(self, feature):  # feature:, att:
#         feature = torch.relu(self.fc1(feature))
#         feature = torch.sigmoid(self.fc2(feature))
#         pred = F.softmax(feature, dim=1)
#         return pred