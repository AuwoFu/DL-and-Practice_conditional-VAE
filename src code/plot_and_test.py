
import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils import pred,pred_with_LP,finn_eval_seq,plot_seq
from dataset import bair_robot_pushing_dataset
import matplotlib.pyplot as plt

import imageio

def create_gif(savePath,seq):
    # ground truth
    images = []
    filenames=[f'./data/processed_data/test/traj_0_to_255.tfrecords/{seq}/{i}.png' for i in range(12)]
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{savePath}/test/{seq}/ground_truth.gif', images)
    
    pred_img=images[:2]
    filenames=[f'{savePath}/test/{seq}/{i+2}.png' for i in range(10)]
    for filename in filenames:
        pred_img.append(imageio.imread(filename))
    imageio.mimsave(f'{savePath}/test/{seq}//predicted_seq.gif', pred_img)



def plot_loss(record_path):
    f=open(f'{record_path}/all_train_record.txt','r')
    
    psnr=[]
    tfr=[]
    KL_weight=[]
    loss=[]
    KLD_loss=[]
    mse_loss=[]
    for l in f.readlines():
        word=l.split(' ')
        if word[1]=='validate':
            psnr.append(float(word[4]))

        elif word[2]=='|':
            loss.append(float(word[4]))
            mse_loss.append(float(word[8]))
            KLD_loss.append(float(word[12]))
            KL_weight.append(float(word[16]))
            tfr.append(float(word[19]))
        else:
            continue
        
    X1=[x for x in range(len(loss))]
    X2=[x*5 for x in range(len(psnr))]

    plt.clf()
    plt.title('training_loss')
    plt.xlabel('Epoch')
    plt.plot(X1,loss,'-',label='epoch loss')
    plt.plot(X1,mse_loss,'-',label='MSE_loss')
    plt.plot(X1,KLD_loss,'-',label='KLD_loss')
    plt.legend()
    plt.savefig(f'{record_path}/training_loss.png')

    fig,ax_1=plt.subplots() # [0,1]
    ax_2=ax_1.twinx() 
    ax_1.set_xlabel('Epoch')
    #ax_2.plot(X1,KL_weight,'--',label='KL weight')
    ax_2.set_ylabel('Teacher ratio')
    ax_2.plot(X1,tfr,'--',label='Teacher forcing ratio')
    
    ax_1.set_ylabel('Validate psnr')
    ax_1.plot(X2,psnr,'.-',label='validate psnr')
    ax_1.legend()
    ax_2.legend()
    plt.savefig(f'{record_path}/psnr_and_ratio.png')
    
    
    



@torch.no_grad()
def test_FP(model_dir):
    saved_model = torch.load(f'{model_dir}/model.pth')
    args = saved_model['args']

    plot_loss(model_dir)
    # --------- transfer to device ------------------------------------
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')

    frame_predictor = saved_model['frame_predictor'].to(device)
    posterior = saved_model['posterior'].to(device)
    decoder = saved_model['decoder'].to(device)
    encoder = saved_model['encoder'].to(device)
    modules = {
        'frame_predictor': frame_predictor.eval(),
        'posterior': posterior.eval(),
        'encoder': encoder.eval(),
        'decoder': decoder.eval(),
    }

    # --------- load a dataset ------------------------------------
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)   
    test_iterator =iter(test_loader)  

    # ---------------start test ---------
    os.makedirs(f'{model_dir}/test', exist_ok=True)

    best_psnr=0
    psnr_list = []
    save_seq=[]
    for batch_index in range(len(test_data) // args.batch_size):
        try:
            test_seq, test_cond = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_loader)
            test_seq,test_cond = next(test_iterator)
        test_seq=test_seq.to(device)
        test_cond=test_cond.to(device)
        pred_seq = pred(test_seq, test_cond, modules, args, device)

        #_, _, psnr = finn_eval_seq(validate_seq[args.n_past:], pred_seq[args.n_past:])
        _, _, psnr = finn_eval_seq(test_seq[:,args.n_past:], pred_seq[:,1:]) # [batch_size,10,3,64,64]
        seq_psnr=np.mean(psnr, axis=1) #[bs,10] -> [bs]
        psnr_list.append(seq_psnr)
        
        if np.max(seq_psnr)>best_psnr:
            best_psnr=np.max(seq_psnr)
            k=np.argmax(seq_psnr)          
            seq=pred_seq[k,1:]
            savePath=f'{model_dir}/test/{args.batch_size*batch_index+k}'
            plot_seq(seq,savePath,args)
            save_seq.append(args.batch_size*batch_index+k)
            
    
    
    #ave_psnr = np.mean(np.concatenate(psnr))
    ave_psnr = np.mean(psnr_list)
    print(f'ave psnr: {ave_psnr}')
    print(f'max seq psnr: {np.max(psnr_list)} at seq {np.argmax(psnr_list)}')

    f=open(f'{model_dir}/test_record.txt','w')
    f.write(f'ave psnr: {ave_psnr}\n')
    f.write(f'max seq psnr: {np.max(psnr_list)} at seq {np.argmax(psnr_list)}\n')
    f.close()
    return save_seq

@torch.no_grad()
def test_LP(model_dir):
    saved_model = torch.load(f'{model_dir}/model.pth')
    args = saved_model['args']

    # --------- transfer to device ------------------------------------
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')

    frame_predictor = saved_model['frame_predictor'].to(device)
    posterior = saved_model['posterior'].to(device)
    decoder = saved_model['decoder'].to(device)
    encoder = saved_model['encoder'].to(device)
    prior=saved_model['prior'].to(device)

    modules = {
        'frame_predictor': frame_predictor.eval(),
        'posterior': posterior.eval(),
        'encoder': encoder.eval(),
        'decoder': decoder.eval(),
        'prior':prior.eval()
    }

    # --------- load a dataset ------------------------------------
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)   
    test_iterator =iter(test_loader)  
    
    # ---------------start test ---------
    os.makedirs(f'{model_dir}/test', exist_ok=True)

    best_psnr=0
    psnr_list = []
    save_seq=[]
    f=open(f'{model_dir}/test_record.txt','w')
    for batch_index in range(len(test_iterator)):
        try:
            test_seq, test_cond = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_loader)
            test_seq,test_cond = next(test_iterator)
        test_seq=test_seq.to(device)
        test_cond=test_cond.to(device)
        pred_seq = pred_with_LP(test_seq, test_cond, modules, args, device)

        #_, _, psnr = finn_eval_seq(validate_seq[args.n_past:], pred_seq[args.n_past:])
        _, _, psnr = finn_eval_seq(test_seq[:,args.n_past:], pred_seq[:,1:]) # [batch_size,10,3,64,64]
        seq_psnr=np.mean(psnr, axis=1) #[bs,10] -> [bs]
        psnr_list.append(seq_psnr)
           
        
        if np.max(seq_psnr)>best_psnr:
            best_psnr=np.max(seq_psnr)
            k=np.argmax(seq_psnr)          
            seq=pred_seq[k,1:]
            savePath=f'{model_dir}/test/{args.batch_size*batch_index+k}'
            plot_seq(seq,savePath,args)
            save_seq.append(args.batch_size*batch_index+k)
    

            

    
    
    #ave_psnr = np.mean(np.concatenate(psnr))
    ave_psnr = np.mean(psnr_list)
    print(f'ave psnr: {ave_psnr}')
    print(f'max seq psnr: {np.max(psnr_list)} at seq {np.argmax(psnr_list)}')

    f=open(f'{model_dir}/test_record.txt','w')
    f.write(f'ave psnr: {ave_psnr}\n')
    f.write(f'max seq psnr: {np.max(psnr_list)} at seq {np.argmax(psnr_list)}\n')
    f.close()
    return save_seq

        
  

if __name__=='__main__':
    # LP
    plot_loss('./logs/lp/best_cycle')
    #save_seq=test_LP('./logs/lp/best_cycle/continued')

    #for i in save_seq:
    #    create_gif('./logs/lp/best_cycle/continued',i)
    
    
    # FP
    #test_FP('./logs/fp/best/continued')
