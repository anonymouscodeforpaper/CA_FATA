#!/usr/bin/env python
# coding: utf-8






from train import train_pre
import argparse





parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='Frappe', help='selection of dataset')
parser.add_argument('--n_epochs', type=int, default=100, help='the number of epochs')
parser.add_argument('--dim', type=int, default=256, help='dimension of user and entity embeddings')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=5e-5, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--context_or', type=str, default=True, help = 'Contextualized or not')
parser.add_argument('--average_or', type=str, default=True, help = 'Importance of feature type or not')


# parser = argparse.ArgumentParser()
# parser.add_argument('--name', type=str, default='Yelp', help='selection of dataset')
# parser.add_argument('--n_epochs', type=int, default=100, help='the number of epochs')
# parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
# parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-3, help='weight of l2 regularization')
# parser.add_argument('--lr', type=float, default=5e-2, help='learning rate')
# parser.add_argument('--context_or', type=str, default=True, help = 'Contextualized or not')
# parser.add_argument('--average_or', type=str, default=True, help = 'Importance of feature type or not')






args = parser.parse_args()






train_pre(args,verbos = True)






