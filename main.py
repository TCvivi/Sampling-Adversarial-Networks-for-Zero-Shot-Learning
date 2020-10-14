import data3, model3, train3
import torch
import argparse


def main(opt):
    dataset = data3.Dataset(dataset=opt.dataset, pool_size=opt.pool_size, sample_size=opt.sample_size)
    #dataset.show_inf()
    feature_size, att_size, nclass = dataset.feature_size, dataset.att_size, dataset.unseen_class
    discriminator = model3.Discriminator(feature_size, att_size, opt.t1).cuda()
    generator = model3.Generator(feature_size, att_size, opt.t2).cuda()
    classifier = model3.Classifier(feature_size, nclass).cuda()
    train3.train(discriminator, generator,classifier, dataset, d_lr=opt.d_lr, g_lr=opt.g_lr,\
                 batch_size=opt.batch_size, alpha=opt.alpha, epochs=opt.epochs)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("please use gpu")
        exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AWA1')
    parser.add_argument('--data_dir', default='../ZSL_MINE/data/GBU/')
    parser.add_argument('--d_lr', type=float, default=0.0005)
    parser.add_argument('--g_lr', type=float, default=0.0005)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--t1', type=float, default=10.0)
    parser.add_argument('--t2', type=float, default=10.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pool_size', type=int, default=100)
    parser.add_argument('--sample_size', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)

    opt = parser.parse_args()
    main(opt)


