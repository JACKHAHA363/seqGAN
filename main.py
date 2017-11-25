from __future__ import print_function
from math import ceil
import numpy as np
import sys
import ipdb

import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers

import args

def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, args):
    """
    Max Likelihood Pretraining for the generator
    """
    num_data = len(real_data_samples)
    total_loss = 0
    for i in range(0, num_data, args.g_bsz):
        inp, target = helpers.prepare_generator_batch(
            real_data_samples[i:i + args.g_bsz],
            start_letter=args.start_letter,
            gpu=args.cuda
        )
        gen_opt.zero_grad()
        loss = gen.batchNLLLoss(inp, target)
        loss.backward()
        gen_opt.step()

        total_loss += loss.data[0]

        if (i / args.g_bsz) % ceil(
                        ceil(num_data / float(args.g_bsz)) / 10.) == 0:  # roughly every 10% of an epoch
            print('.', end='')
            sys.stdout.flush()

    # each loss in a batch is loss per sample
    total_loss = total_loss / ceil(num_data / float(args.g_bsz)) / args.max_seq_len

    # sample from generator and compute oracle NLL
    oracle_loss = helpers.batchwise_oracle_nll(
        gen, oracle, args.num_eval, args
    )
    return oracle_loss, total_loss


def train_generator_PG(gen, gen_opt, dis, args):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for batch in range(args.g_steps):
        s = gen.sample(args.g_bsz)
        inp, target = helpers.prepare_generator_batch(
            s, start_letter=args.start_letter, gpu=args.cuda
        )
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

def train_discriminator(discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs, args):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    pos_val = oracle.sample(100)
    neg_val = generator.sample(100)
    val_inp, val_target = helpers.prepare_discriminator_data(
        pos_val, neg_val, gpu=args.cuda
    )
    num_data = len(real_data_samples)
    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, args.num_data)
        dis_inp, dis_target = helpers.prepare_discriminator_data(
            real_data_samples, s, gpu=args.cuda
        )
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * num_data, args.d_bsz):
                inp, target = dis_inp[i:i + args.d_bsz], dis_target[i:i + args.d_bsz]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data[0]
                total_acc += torch.sum((out>0.5)==(target>0.5)).data[0]

                if (i / args.d_bsz) % ceil(ceil(2 * num_data / float(
                        args.d_bsz)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * num_data / float(args.d_bsz))
            total_acc /= float(2 * num_data)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data[0]/200.))

# MAIN
if args.oracle_load is not None:
    oracle = generator.Generator(
        args.g_emb_dim, args.g_hid_dim,
        args.vocab_size, args.max_seq_len, gpu=args.cuda
    )
    oracle.load_state_dict(torch.load(args.oracle_load))
else:
     oracle = generator.Generator(
        args.g_emb_dim, args.g_hid_dim,
        args.vocab_size, args.max_seq_len, gpu=args.cuda,
        oracle_init=True
     )

gen = generator.Generator(
    args.g_emb_dim, args.g_hid_dim,
    args.vocab_size, args.max_seq_len, gpu=args.cuda,
)

dis = discriminator.Discriminator(
    args.d_emb_dim, args.d_hid_dim, args.vocab_size,
    args.max_seq_len, gpu=args.cuda
)

if args.cuda:
    oracle = oracle.cuda()
    gen = gen.cuda()
    dis = dis.cuda()

oracle_samples = helpers.batchwise_sample(oracle, args.num_data)
if args.oracle_save is not None:
    torch.save(oracle.state_dict(), args.oracle_save)

# GENERATOR MLE TRAINING
gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
if args.pre_g_load is not None:
    print("Load pretrained MLE gen")
    gen.load_state_dict(torch.load(args.pre_g_load))
else:
    print('Starting Generator MLE Training...')
    for epoch in range(args.mle_epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()

        oracle_loss, total_loss = train_generator_MLE(
            gen, gen_optimizer, oracle, oracle_samples, args
        )
        print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))

if args.pre_g_save is not None:
    torch.save(gen.state_dict(), args.pre_g_save)


# PRETRAIN DISCRIMINATOR
dis_optimizer = optim.Adagrad(dis.parameters())
if args.pre_d_load is not None:
    print("Load pretrained D")
    dis.load_state_dict(torch.load(args.pre_d_load))
else:
    print('\nStarting Discriminator Training...')
    train_discriminator(
        dis, dis_optimizer, oracle_samples,
        gen, oracle, args.d_pre_steps, args.d_pre_epochs, args
    )
if args.pre_d_save is not None:
    torch.save(dis.state_dict(), args.pre_d_save)


# ADVERSARIAL TRAINING
print('\nStarting Adversarial Training...')
oracle_loss = helpers.batchwise_oracle_nll(
    gen, oracle, args.num_eval, args
)
print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)

for epoch in range(args.adv_epochs):
    print('\n--------\nEPOCH %d\n--------' % (epoch+1))
    # TRAIN GENERATOR
    print('\nAdversarial Training Generator : ', end='')
    sys.stdout.flush()
    train_generator_PG(gen, gen_optimizer, dis, args)
    # sample from generator and compute oracle NLL
    oracle_loss = helpers.batchwise_oracle_nll(
        gen, oracle, args.num_eval, args)
    print(' oracle_sample_NLL = %.4f' % oracle_loss)

    # TRAIN DISCRIMINATOR
    print('\nAdversarial Training Discriminator : ')
    train_discriminator(
        dis, dis_optimizer, oracle_samples,
        gen, oracle, args.d_steps, args.d_epochs, args
    )