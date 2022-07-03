import argparse
from inspect import trace
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion,  finn_eval_seq, pred, rec,plot_seq

from math import fsum

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=10, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0.01, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  

    args = parser.parse_args()
    return args

def train(x, cond, modules, optimizer, kl_anneal, args):
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0
    use_teacher_forcing = True if random.random() < args.tfr else False
    n_frame=args.n_past + args.n_future

    mse_fn=nn.MSELoss()
    h_seq = [modules['encoder'](x[:,i]) for i in range(n_frame)]

    if use_teacher_forcing:
        for i in range(1, n_frame):
            h_target=h_seq[i][0] # h1,_=modules['encoder'](x[:,i])
            z, mu, logvar=modules['posterior'](h_target) # zt
            
            if i<=args.n_past:
                h2,skip=h_seq[i-1]# h2,skip=modules['encoder'](x[:,i-1])
            else:
                h2=h_seq[i-1][0]
            
            code_in=torch.cat((h2,z,cond[:,i-1]),1)
            code_in=code_in.type(torch.FloatTensor)
            g=modules['frame_predictor'](code_in)
            y=modules['decoder']((g,skip)) #xt_head

            kld+=kl_criterion(mu,logvar,args)
            mse+=mse_fn(x[:,i],y)

    else:      # not teacher forcing  
        post=h_seq[0]
        for i in range(1, args.n_past + args.n_future):
            h_target=h_seq[i][0] # h1,_=modules['encoder'](x[:,i])
            z, mu, logvar=modules['posterior'](h_target) # zt
            
            if i<=args.n_past:
                h2=h_seq[i-1][0] # feed 2 ground truth
                skip=h_seq[i-1][1]
            else:
                h2,_=modules['encoder'](post)                 

            code_in=torch.cat((h2,z,cond[:,i-1]),1)
            code_in=code_in.type(torch.FloatTensor)
            g=modules['frame_predictor'](code_in)
            y=modules['decoder']((g,skip))
            post=y

            kld+=kl_criterion(mu,logvar,args)
            mse+=mse_fn(x[:,i],y)

    beta = kl_anneal.get_beta()
    loss = mse + kld * beta
    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy() / n_frame, mse.detach().cpu().numpy() / n_frame, kld.detach().cpu().numpy() / n_frame,beta

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        self.beta=0
        self.min=0
        self.max=1

        self.useCyclical=args.kl_anneal_cyclical
        self.CyclicalCycle=args.kl_anneal_cycle # split total epoch into n cycle
        if self.useCyclical:
            #self.period=args.niter//self.CyclicalCycle # each cycle lengthr # each cycle length
            self.period=args.epoch_size//self.CyclicalCycle # for update with iter
        else:
            #self.period=args.niter
            self.period=args.epoch_size # for update with iter
        self.ratio=args.kl_anneal_ratio # use n-times period to reach max
        self.step=(self.max-self.min)/(self.period*self.ratio) # increace step

    def update(self,current_epoch):
        if self.useCyclical:
            if current_epoch%self.period==0:
                self.beta=0
                return
        if self.beta<self.max:
            self.beta+=self.step       
        if self.beta>self.max:
            self.beta=self.max
        return
    
    def get_beta(self):
        return self.beta


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'using {device}')


    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load(f'{args.model_dir}/model.pth')
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = f'{args.log_dir}/continued'
        start_epoch = saved_model['last_epoch']
    else:
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(f'{args.log_dir}/gen/', exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(f'./{args.log_dir}/train_record.txt'):
        os.remove(f'./{args.log_dir}/train_record.txt')
    #args.tfr=1
    print(args)

    with open(f'./{args.log_dir}/train_record.txt', 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim+7, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder'].to(device)
        encoder = saved_model['encoder'].to(device)
    else:
        encoder = vgg_encoder(args.g_dim).to(device)
        decoder = vgg_decoder(args.g_dim).to(device)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)  

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)


    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)   
    test_iterator =iter(test_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- training loop ------------------------------------

    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    best_test_psnr=0

    for epoch in range(start_epoch, start_epoch + niter):
        frame_predictor.train()
        posterior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0

        for temp in range(args.epoch_size):
            try:
                seq, cond = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)
            seq=seq.to(device)
            cond=cond.to(device)

            loss, mse, kld, KL_weight = train(seq, cond, modules, optimizer, kl_anneal, args)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld
            
            # update teacher forcing ratio and KL weight
            kl_anneal.update(temp)
        
        with open(f'./{args.log_dir}/train_record.txt', 'a') as train_record:
            train_record.write(f'[epoch: {epoch:02d}] | loss: {epoch_loss/args.epoch_size:.5f} | mse loss: { epoch_mse / args.epoch_size:.5f} | kld loss: {epoch_kld / args.epoch_size:.5f} | KL weight: {KL_weight:.4f} | tfr: {args.tfr:.4f} \n')
        
        # update teacher forcing ratio and KL weight
        #kl_anneal.update(epoch)
        
        if (epoch+1)%2==0 and epoch >= args.tfr_start_decay_epoch:

            ### Update teacher forcing ratio ###
            ## todo 
            args.tfr-=args.tfr_decay_step
            if args.tfr < args.tfr_lower_bound:
                args.tfr =args.tfr_lower_bound     

        progress.update(1)       

        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

        if epoch % 5 == 0:
            psnr_list = []
            for _ in range(len(validate_data) // args.batch_size):
                try:
                    validate_seq, validate_cond = next(validate_iterator)
                except StopIteration:
                    validate_iterator = iter(validate_loader)
                    validate_seq, validate_cond = next(validate_iterator)
                validate_seq=validate_seq.to(device)
                validate_cond=validate_cond.to(device)
                pred_seq = pred(validate_seq, validate_cond, modules, args, device)
                #print(validate_seq[:,args.n_past:].shape,pred_seq[:,args.n_past:].shape)
                #_, _, psnr = finn_eval_seq(validate_seq[args.n_past:], pred_seq[args.n_past:])
                _, _, psnr = finn_eval_seq(validate_seq[:,args.n_past:], pred_seq[:,1:]) # [batch_size,10,3,64,64]
                psnr_list.append(psnr)
                
            #ave_psnr = np.mean(np.concatenate(psnr))
            ave_psnr = np.mean(psnr_list)
            
            with open(f'./{args.log_dir}/train_record.txt', 'a') as train_record:
                train_record.write(f'====================== validate psnr = {ave_psnr:.5f} ========================\n')

            '''if ave_psnr > best_val_psnr:
                best_val_psnr = ave_psnr
                # save the model
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'args': args,
                    'last_epoch': epoch},
                    f'{args.log_dir}/model.pth')'''
            
            psnr_list = []
            for _ in range(len(test_data) // args.batch_size):
                try:
                    test_seq, test_cond = next(test_iterator)
                except StopIteration:
                    test_iterator = iter(test_loader)
                    test_seq,test_cond = next(test_iterator)
                test_seq=test_seq.to(device)
                test_cond=test_cond.to(device)
                pred_seq = pred(test_seq, test_cond, modules, args, device)
                _, _, psnr = finn_eval_seq(test_seq[:,args.n_past:], pred_seq[:,1:])
                psnr_list.append(psnr)
            test_psnr = np.mean(psnr_list)

            if test_psnr > best_test_psnr:
                best_test_psnr = test_psnr
                # save the model
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'args': args,
                    'last_epoch': epoch},
                    f'{args.log_dir}/model.pth')
            


            with open(f'./{args.log_dir}/train_record.txt', 'a') as train_record:
                train_record.write(f'====================== test psnr = {test_psnr:.5f} ========================\n')



        '''if epoch % 20 == 0:
            try:
                validate_seq, validate_cond = next(validate_iterator)
            except StopIteration:
                validate_iterator = iter(validate_loader)
                validate_seq, validate_cond = next(validate_iterator)

            rec_seq = rec(validate_seq, validate_cond, modules, args, device)
            plot_seq(rec_seq,f'rec_{epoch}')
            _, _, psnr = finn_eval_seq(validate_seq[:,args.n_past:], rec_seq[:,1:]) # [batch_size,10,3,64,64]
            
            with open(f'./{args.log_dir}/train_record.txt', 'a') as train_record:
                train_record.write(f'====================== rec valid psnr = {np.mean(psnr):.5f} ========================\n')
            
            pred_seq = pred(validate_seq, validate_cond, modules, args, device)
            plot_seq(pred_seq,f'pred_{epoch}')
            #plot_pred(validate_seq, validate_cond, modules, epoch, args)
            #plot_rec(validate_seq, validate_cond, modules, epoch, args)'''

if __name__ == '__main__':
    main()
        
