#!/usr/bin/env python

# Add an encoder/decoder
from tensorboardX import SummaryWriter

import sys
from TrackLoss import TrackLoss

reload(sys)
sys.setdefaultencoding('utf-8')

import os
import os.path as ops
import numpy as np
from sklearn.metrics import f1_score

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

#import tensorflow as tf
from models_multiview import FrontaliseModelMasks_wider

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import pdb
import argparse

os.environ['CUDA_VISIBLE_DEVICES']='0'

BASE_LOCATION = os.getcwd()


arguments = argparse.ArgumentParser()
arguments.add_argument('--lr', type=float, default=0.001)
arguments.add_argument('--momentum', type=float, default=0.9)
arguments.add_argument('--load_old_model', action='store_true', default=False)
arguments.add_argument('--num_views', type=int, default=2, 
        help='Number of source views + 1 (e.g. the target view) so set = 2 for 1 source view')
arguments.add_argument('--continue_epoch', type=int, default=0)
arguments.add_argument('--crop_size', type=int, default=180)
arguments.add_argument('--num_additional_ids', type=int, default=32)
arguments.add_argument('--minpercentile', type=float, default=0.0)
arguments.add_argument('--maxpercentile', type=float, default=0.5)
arguments.add_argument('--use_landmark_supervision', action='store_true', default=False)
arguments.add_argument('--use_landmark_mask_supervision', action='store_true', default=False)
arguments.add_argument('--num_workers', type=int, default=4)
arguments.add_argument('--max_percentile', type=float, default=0.85)
arguments.add_argument('--diff_percentile', type=float, default=0.1)
arguments.add_argument('--batch_size', type=int, default=64)
arguments.add_argument('--log_dir', type=str, default=BASE_LOCATION+'/code_faces/runs/')
arguments.add_argument('--embedding_size', type=int, default=256)
arguments.add_argument('--run_dir', type=str, 
        default='curriculumwidervox2%d_fabnet%s/lr_%.4f_lambda%.4f_nv%d_addids%d_cropsize%d_exp_flow_l1_curriculum_lr_1e-3')
arguments.add_argument('--old_model', type=str, default=BASE_LOCATION + '')
arguments.add_argument('--model_epoch_path', type=str, 
        default=BASE_LOCATION + '/validate_tv_model/validate_l1_curriculum_lr_1e-3/')
arguments.add_argument('--learn_mask', action='store_true')
opt = arguments.parse_args()


opt.run_dir = opt.run_dir % (opt.embedding_size, str(opt.use_landmark_supervision), 
        opt.lr, 0, opt.num_views, opt.num_additional_ids, opt.crop_size)
opt.model_epoch_path = opt.model_epoch_path + 'model_' + str(opt.embedding_size) +\
        '_'+ str(opt.num_views) + '_' + str(opt.crop_size) + "_"

opt.model_epoch_path = opt.model_epoch_path + '_epoch'

#criterion_reconstruction = tf.losses.absolute_difference(reduction=None)
criterion_reconstruction = nn.L1Loss(reduce=False).cuda()

criterion_reconstruction_val = nn.L1Loss().cuda()

criterion_feature_recon = nn.MSELoss(reduce=False).cuda()

criterion_feature_recon_val = nn.MSELoss().cuda()

def criterion_l2(input_f, target_f):
    # return a per batch l2 loss
    res = (input_f - target_f)
    res = res * res
    return res

def get_sort_loss(pre_sort_loss, sort_idx_list, start, end):
    res = pre_sort_loss[sort_idx_list[start]]
    for idx in range(start+1, end):
        res += pre_sort_loss[sort_idx_list[idx]] 
    res = res / (end - start) 
    return res

def criterion_tv_curri(mat):
    res = torch.abs(mat[:,:,:,:-1] - mat[:,:,:,1:]).view(mat.size(0), -1) \
            + torch.abs(mat[:,:,:-1,:] - mat[:,:,1:,:]).view(mat.size(0), -1)
    res = res
    return res

def criterion_tv(mat):
    res = torch.abs(mat).view(mat.size(0), -1).mean(dim=1)
    return res

def criterion_tv_val(mat):
    res = torch.abs(mat).view(mat.size(0), -1).mean()
    return res

def w_hook(grad):
    print "shape of grad:{0}".format(str(grad.shape))
    pdb.set_trace()


model = FrontaliseModelMasks_wider(inner_nc=opt.embedding_size, \
        num_output_channels=3, num_masks=0, num_additional_ids=opt.num_additional_ids)
# model.headpose_encoder.eval()
# Note that head_pose_encoder in eval model for batch norm, we only need grad for input
"""
for param in model.headpose_encoder.parameters():
    param.requires_grad = False
"""
model = model.cuda()


model.lr = opt.lr
model.momentum = opt.momentum
writer = SummaryWriter('/%s/%s' % (opt.log_dir, opt.run_dir))


from TCAE_data import VoxCeleb2 as VoxCeleb2

if opt.num_views > 2:
    optimizer = optim.SGD(
            [
                {'params' : model.encoder.parameters(), 'lr' : opt.lr},
                {'params' : model.pose_decoder.parameters(), 'lr' : opt.lr},
                {'params' : model.expression_decoder.parameters(), 'lr' : opt.lr},
                ], 
            lr=opt.lr, momentum=opt.momentum
            )
else:
    # Here, we should select a model without mask [2018/09/06]
    #optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    optimizer = optim.SGD(
            [
                {'params' : model.encoder.parameters(), 'lr' : opt.lr},
                {'params' : model.pose_decoder.parameters(), 'lr' : opt.lr},
                {'params' : model.expression_decoder.parameters(), 'lr' : opt.lr},
                ], 
            lr=opt.lr, momentum=opt.momentum
            )


def train(epoch, model, criterion, preceptual_criterion, optimizer, w1=1.0, w2=1.0, w3=1.0, w4=1.0, \
        w5=1.0, w6=1.0, w7=1.0, w8=1.0, w9=1.0, w10=1.0, minpercentile=0, maxpercentile=0.50):

    train_set = VoxCeleb2(opt.num_views, epoch, 1, jittering=True)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, 
            batch_size=opt.batch_size, shuffle=True)

    """
    for x in model.pose_decoder.named_parameters():
        if x[0] == '30.weight':
            x[1].register_hook(w_hook)
            print "register hook ~"
            print "register hook ~"
    """

    t_loss = 0.0

    t_reconsturction_loss = 0.0

    t_pose_img_pose_loss  = 0.0
    t_pose_img_exp_loss  = 0.0
    t_exp_flow_tv_loss = 0.0 # regularize orientation in pose flow be similar in counter line

    t_exp_img_pose_loss  = 0.0
    t_exp_img_exp_loss  = 0.0

    t_out_img_pose_loss  = 0.0
    t_out_img_exp_loss  = 0.0

    t_exp_recon_loss = 0.0
    t_pose_recon_loss = 0.0

    epoch_t = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        optimizer.zero_grad()
        # suppose 
        target_image = Variable(batch[0]).cuda()
        offset = 1
        input_images = batch[offset:]
        input_images = Variable(input_images[0]).cuda()

        target_pose_f, target_exp_f = model.encoder(target_image)
        input_pose_f, input_exp_f = model.encoder(input_images)

        pose_vector = torch.cat((input_pose_f, target_pose_f), 1)
        exp_vector = torch.cat((input_exp_f, target_exp_f), 1)

        all_pose_flow_samplers = model.pose_decoder(pose_vector) #n,c,h,w
        pose_flow_samplers = all_pose_flow_samplers[:,0:2,:,:]
        pose_mask = all_pose_flow_samplers[:,-1,:,:].unsqueeze(1)
        if iteration % 100 == 0:
            print "pose_flow example:"
            print pose_flow_samplers[0][0][0]
        pose_grid = np.linspace(-1,1, pose_flow_samplers.size(2))
        pose_grid = np.meshgrid(pose_grid, pose_grid)
        pose_grid = np.stack(pose_grid, 2)  # w x h x 2
        pose_grid = torch.Tensor(pose_grid).unsqueeze(0).repeat(target_pose_f.size(0), 1,1,1).cuda()
        pose_grid = Variable(pose_grid, requires_grad=False)
        pose_samplers = (pose_flow_samplers.permute(0,2,3,1) + pose_grid).clamp(min=-1,max=1)
        pose_image = F.grid_sample(input_images.detach(), pose_samplers)
        # pose_flow_samplers.register_hook(w_hook)
        if epoch >= epoch_t:
            pose_img_pose_f, pose_img_exp_f = model.encoder(pose_image)
            pose_img_pose_loss_pre_sort = criterion_l2(pose_img_pose_f, target_pose_f).view(pose_img_pose_f.size(0), -1).mean(dim=1)

            pose_img_exp_loss_pre_sort = criterion_l2(pose_img_exp_f, input_exp_f).view(pose_img_exp_f.size(0), -1).mean(dim=1)

            pose_2vector = torch.cat((pose_img_pose_f, input_pose_f), 1)
            all_pose_2flow_samplers = model.pose_decoder(pose_2vector) 
            pose_flow_2samplers = all_pose_2flow_samplers[:,0:2,:,:]
            pose_2grid = np.linspace(-1,1, pose_flow_2samplers.size(2))
            pose_2grid = np.meshgrid(pose_2grid, pose_2grid)
            pose_2grid = np.stack(pose_2grid, 2)  # w x h x 2
            pose_2grid = torch.Tensor(pose_2grid).unsqueeze(0).repeat(pose_img_pose_f.size(0), 1,1,1).cuda()
            pose_2grid = Variable(pose_2grid, requires_grad=False)
            pose_2samplers = (pose_flow_2samplers.permute(0,2,3,1) + pose_2grid).clamp(min=-1,max=1)
            pose_2image = F.grid_sample(pose_image.detach(), pose_2samplers)
            pose_recon_loss_pre_sort = criterion_reconstruction(pose_2image, input_images).view(input_images.size(0), -1).mean(dim=1)

        all_exp_flow_samplers = model.expression_decoder(exp_vector) 
        exp_flow_samplers = all_exp_flow_samplers[:,0:2,:,:]
        exp_mask = all_exp_flow_samplers[:,-1,:,:].unsqueeze(1)
        if iteration % 100 == 0:
            print "exp_flow example:"
            print exp_flow_samplers[0][0][0]
        exp_grid = np.linspace(-1,1, exp_flow_samplers.size(2))
        exp_grid = np.meshgrid(exp_grid, exp_grid)
        exp_grid = np.stack(exp_grid, 2)  # w x h x 2
        exp_grid = torch.Tensor(exp_grid).unsqueeze(0).repeat(target_exp_f.size(0), 1,1,1).cuda()
        exp_grid = Variable(exp_grid, requires_grad=False)
        exp_samplers = (exp_flow_samplers.permute(0,2,3,1) + exp_grid).clamp(min=-1,max=1)
        exp_image = F.grid_sample(input_images.detach(), exp_samplers)
        # we constrain generated pose image has similar pose to target
        if epoch >= epoch_t:
            exp_img_pose_f, exp_img_exp_f = model.encoder(exp_image)
            exp_img_pose_loss_pre_sort = criterion_l2(exp_img_pose_f, input_pose_f).view(exp_img_pose_f.size(0), -1).mean(dim=1)

            exp_img_exp_loss_pre_sort = criterion_l2(exp_img_exp_f, target_exp_f).view(exp_img_exp_f.size(0), -1).mean(dim=1)
            
            exp_2vector = torch.cat((exp_img_exp_f, input_exp_f), 1)
            all_exp_2flow_samplers = model.expression_decoder(exp_2vector) 
            exp_flow_2samplers = all_exp_2flow_samplers[:,0:2,:,:]
            exp_2grid = np.linspace(-1,1, exp_flow_2samplers.size(2))
            exp_2grid = np.meshgrid(exp_2grid, exp_2grid)
            exp_2grid = np.stack(exp_2grid, 2)  # w x h x 2
            exp_2grid = torch.Tensor(exp_2grid).unsqueeze(0).repeat(exp_img_exp_f.size(0), 1,1,1).cuda()
            exp_2grid = Variable(exp_2grid, requires_grad=False)
            exp_2samplers = (exp_flow_2samplers.permute(0,2,3,1) + exp_2grid).clamp(min=-1,max=1)
            exp_2image = F.grid_sample(exp_image.detach(), exp_2samplers)
            exp_recon_loss_pre_sort = criterion_reconstruction(exp_2image, input_images).view(input_images.size(0), -1).mean(dim=1)

            # we add tv loss for exp flow
            exp_sub = exp_samplers - exp_grid 
            exp_flow_tv_loss_pre_sort = criterion_tv(exp_sub)

        merge_flow = torch.cat([pose_flow_samplers.unsqueeze(4), exp_flow_samplers.unsqueeze(4)], 4)
        attention_maps = torch.cat([pose_mask.unsqueeze(4), exp_mask.unsqueeze(4)], 4)
        output_flow = (merge_flow * attention_maps.exp()).sum(dim=4) / attention_maps.exp().sum(dim=4)
        merge_grid = np.linspace(-1,1, merge_flow.size(2))
        merge_grid = np.meshgrid(merge_grid, merge_grid)
        merge_grid = np.stack(merge_grid, 2)  # w x h x 2
        merge_grid = torch.Tensor(merge_grid).unsqueeze(0).repeat(exp_vector.size(0), 1,1,1).cuda()
        merge_grid = Variable(merge_grid, requires_grad=False)
        samplers = (output_flow.permute(0,2,3,1) + merge_grid).clamp(min=-1,max=1)
        output_image = F.grid_sample(input_images.detach(), samplers)

        out_img_pose_f, out_img_exp_f = model.encoder(output_image)
        out_img_pose_loss_pre_sort = criterion_l2(out_img_pose_f, target_pose_f).view(out_img_pose_f.size(0), -1).mean(dim=1)

        out_img_exp_loss_pre_sort = criterion_l2(out_img_exp_f, target_exp_f).view(out_img_exp_f.size(0), -1).mean(dim=1)

        loss_pre_sort = criterion_reconstruction(output_image, target_image).view(output_image.size(0), -1).mean(dim=1)
        loss_sort_all = loss_pre_sort.sort()
        sort_loss = loss_sort_all[0]

        end_idx = int(maxpercentile * input_images.size(0)) 
        start_idx = int(minpercentile * input_images.size(0))

        loss = sort_loss[start_idx:end_idx].mean()
        pose_img_pose_loss_sort = pose_img_pose_loss_pre_sort.sort()[0][start_idx:end_idx].mean()
        pose_img_exp_loss_sort = pose_img_exp_loss_pre_sort.sort()[0][start_idx:end_idx].mean() 
        exp_img_pose_loss_sort = exp_img_pose_loss_pre_sort.sort()[0][start_idx:end_idx].mean()
        exp_img_exp_loss_sort = exp_img_exp_loss_pre_sort.sort()[0][start_idx:end_idx].mean()
        out_img_pose_loss_sort = out_img_pose_loss_pre_sort.sort()[0][start_idx:end_idx].mean()
        out_img_exp_loss_sort = out_img_exp_loss_pre_sort.sort()[0][start_idx:end_idx].mean()
        exp_recon_loss_sort = exp_recon_loss_pre_sort.sort()[0][start_idx:end_idx].mean()
        pose_recon_loss_sort = pose_recon_loss_pre_sort.sort()[0][start_idx:end_idx].mean()

        exp_flow_tv_loss_sort = exp_flow_tv_loss_pre_sort.sort()[0][start_idx:end_idx].mean()


        total_loss = w1 * loss \
                + w2 * pose_img_pose_loss_sort  + w3 * pose_img_exp_loss_sort \
                + w5 * exp_img_pose_loss_sort + w6 * exp_img_exp_loss_sort \
                + w7 * out_img_pose_loss_sort + w8 * out_img_exp_loss_sort \
                + w9 * exp_recon_loss_sort + w10 * pose_recon_loss_sort \
                + w4 * exp_flow_tv_loss_sort 


        total_loss.backward()


        t_loss += total_loss.cpu().item()
        t_reconsturction_loss += w1 * loss.cpu().item()
        t_pose_img_pose_loss += w2 * pose_img_pose_loss_sort.cpu().item() # source -> expression flow,  target -> pose flow
        t_pose_img_exp_loss += w3 * pose_img_exp_loss_sort.cpu().item() # source -> expression flow,  target -> pose flow
        t_exp_img_pose_loss += w5 * exp_img_pose_loss_sort.cpu().item() # source -> expression flow,  target -> pose flow
        t_exp_img_exp_loss += w6 * exp_img_exp_loss_sort.cpu().item() # source -> expression flow,  target -> pose flow
        t_out_img_pose_loss += w7 * out_img_pose_loss_sort.cpu().item() # source -> expression flow,  target -> pose flow
        t_out_img_exp_loss += w8 * out_img_exp_loss_sort.cpu().item() # source -> expression flow,  target -> pose flow
        t_exp_recon_loss += w9 * exp_recon_loss_sort.cpu().item() # source -> expression flow,  target -> pose flow
        t_pose_recon_loss += w10 * pose_recon_loss_sort.cpu().item() # source -> expression flow,  target -> pose flow
        t_exp_flow_tv_loss += w4 * exp_flow_tv_loss_sort.cpu().item()

        if iteration == 1 or iteration == 10:
            # save input-output image and loss
            writer.add_image('Image_train/%d_input_%d' % (iteration, 0), 
                    torchvision.utils.make_grid(input_images[0:8,:,:,:].data), epoch)
            writer.add_image('Image_train/%d_output' % iteration, 
                    torchvision.utils.make_grid(output_image[0:8,:,:,:].data), epoch)
            writer.add_image('Image_train/%d_target' % iteration, 
                    torchvision.utils.make_grid(target_image[0:8,:,:,:].data), epoch)
            writer.add_image('Image_train/%d_exp' % iteration, 
                    torchvision.utils.make_grid(exp_image[0:8,:,:,:].data), epoch)
            writer.add_image('Image_train/%d_pose' % iteration, 
                    torchvision.utils.make_grid(pose_image[0:8,:,:,:].data), epoch)


        optimizer.step()

        if iteration % 20 == 0:
            print("train: Epoch {}: {}/{} with error {:.4f}, reconstruction loss {:.4f},\n \
                    train pose img pose loss(target<->pose img) {:.4f}, pose img exp loss(input <->pose img) {:.4f}\n \
                    train exp img pose loss(input <->exp img) {:.4f}, exp img exp loss(target<->exp img) {:.4f}\n \
                    train out img pose loss(input <->exp img) {:.4f}, out img exp loss(target<->exp img) {:.4f}\n \
                    train exp img recon loss(input <->exp img) {:.4f}, pose img recon loss(input <->pose img) {:.4f}\n \
                    train exp flow tv loss {:.4f}".
                    format(epoch, iteration, 
                        len(training_data_loader), 
                        t_loss / float(iteration), 
                        t_reconsturction_loss/float(iteration),
                        t_pose_img_pose_loss/float(iteration),
                        t_pose_img_exp_loss/float(iteration),
                        t_exp_img_pose_loss/float(iteration),
                        t_exp_img_exp_loss/float(iteration),
                        t_out_img_pose_loss/float(iteration),
                        t_out_img_exp_loss/float(iteration),
                        t_exp_recon_loss/float(iteration),
                        t_pose_recon_loss/float(iteration),
                        t_exp_flow_tv_loss/float(iteration)
                        )
                    )

    return {
            'total_error' : t_loss / float(iteration),
            'reconsturction_loss': t_reconsturction_loss/float(iteration),
            'pose_img_pose_loss': t_pose_img_pose_loss/float(iteration),
            'pose_img_exp_loss': t_pose_img_exp_loss/float(iteration),
            'exp_img_pose_loss': t_exp_img_pose_loss/float(iteration),
            'exp_img_exp_loss': t_exp_img_exp_loss/float(iteration),
            'out_img_pose_loss': t_out_img_pose_loss/float(iteration),
            'out_img_exp_loss': t_out_img_exp_loss/float(iteration),
            'exp_img_recon_loss': t_exp_recon_loss/float(iteration),
            'pose_img_recon_loss': t_pose_recon_loss/float(iteration),
            'exp_flow_tv_loss': t_exp_flow_tv_loss/float(iteration)
            }

def val(epoch, model, criterion, criterion_feature_recon, optimizer,  w1=1.0, w2=1.0, w3=0.1, \
        w4=1.0, w5=1.0, w6=1.0, w7=1.0, w8=1.0, w9=1.0, w10=1.0,  minpercentile=0, maxpercentile=50):

    val_set = VoxCeleb2(opt.num_views, 0, 2, jittering=True) 

    val_data_loader = DataLoader(dataset=val_set, 
            num_workers=opt.num_workers, batch_size=opt.batch_size/2, shuffle=False)

    t_loss = 0.0
    t_reconsturction_loss = 0.0

    t_pose_img_pose_loss  = 0.0
    t_pose_img_exp_loss  = 0.0
    t_exp_flow_tv_loss = 0.0 # regularize orientation in pose flow be similar in counter line

    t_exp_img_pose_loss  = 0.0
    t_exp_img_exp_loss  = 0.0

    t_out_img_pose_loss  = 0.0
    t_out_img_exp_loss  = 0.0

    t_exp_recon_loss = 0.0
    t_pose_recon_loss = 0.0

    for iteration, batch in enumerate(val_data_loader, 1):
        target_image = Variable(batch[0]).cuda()
        offset = 1
        input_images = batch[offset:]
        input_images = Variable(input_images[0]).cuda()

        target_pose_f, target_exp_f = model.encoder(target_image)
        input_pose_f, input_exp_f = model.encoder(input_images)

        pose_vector = torch.cat((input_pose_f, target_pose_f), 1)
        exp_vector = torch.cat((input_exp_f, target_exp_f), 1)

        all_pose_flow_samplers = model.pose_decoder(pose_vector) 
        pose_flow_samplers = all_pose_flow_samplers[:,0:2,:,:]
        pose_mask = all_pose_flow_samplers[:,-1,:,:].unsqueeze(1)
        pose_grid = np.linspace(-1,1, pose_flow_samplers.size(2))
        pose_grid = np.meshgrid(pose_grid, pose_grid)
        pose_grid = np.stack(pose_grid, 2)  # w x h x 2
        pose_grid = torch.Tensor(pose_grid).unsqueeze(0).repeat(target_pose_f.size(0), 1,1,1).cuda()
        pose_grid = Variable(pose_grid, requires_grad=False)
        pose_samplers = (pose_flow_samplers.permute(0,2,3,1) + pose_grid).clamp(min=-1,max=1)
        pose_image = F.grid_sample(input_images.detach(), pose_samplers)
        # we constrain generated pose image has similar pose to target
        pose_img_pose_f, pose_img_exp_f = model.encoder(pose_image)
        # pose_img_pose_f near target_img_pose_f, pose_img_exp_f near input_img_exp_f
        pose_img_pose_loss = criterion_feature_recon_val(pose_img_pose_f, target_pose_f)
        pose_img_pose_loss = pose_img_pose_loss / target_pose_f.size(0) / target_pose_f.size(1)  #.mean()

        pose_img_exp_loss = criterion_feature_recon_val(pose_img_exp_f, input_exp_f)
        pose_img_exp_loss = pose_img_exp_loss / target_pose_f.size(0) / target_pose_f.size(1)  #.mean()

        pose_2vector = torch.cat((pose_img_pose_f, input_pose_f), 1)
        all_pose_2flow_samplers = model.pose_decoder(pose_2vector) 
        pose_flow_2samplers = all_pose_2flow_samplers[:,0:2,:,:]
        pose_2grid = np.linspace(-1,1, pose_flow_2samplers.size(2))
        pose_2grid = np.meshgrid(pose_2grid, pose_2grid)
        pose_2grid = np.stack(pose_2grid, 2)  # w x h x 2
        pose_2grid = torch.Tensor(pose_2grid).unsqueeze(0).repeat(pose_img_pose_f.size(0), 1,1,1).cuda()
        pose_2grid = Variable(pose_2grid, requires_grad=False)
        pose_2samplers = (pose_flow_2samplers.permute(0,2,3,1) + pose_2grid).clamp(min=-1,max=1)
        pose_2image = F.grid_sample(pose_image.detach(), pose_2samplers)
        pose_recon_loss = criterion_reconstruction_val(pose_2image, input_images)


        all_exp_flow_samplers = model.expression_decoder(exp_vector) 
        exp_flow_samplers = all_exp_flow_samplers[:,0:2,:,:]
        exp_mask = all_exp_flow_samplers[:,-1,:,:].unsqueeze(1)
        exp_grid = np.linspace(-1,1, exp_flow_samplers.size(2))
        exp_grid = np.meshgrid(exp_grid, exp_grid)
        exp_grid = np.stack(exp_grid, 2)  # w x h x 2
        exp_grid = torch.Tensor(exp_grid).unsqueeze(0).repeat(target_exp_f.size(0), 1,1,1).cuda()
        exp_grid = Variable(exp_grid, requires_grad=False)
        exp_samplers = (exp_flow_samplers.permute(0,2,3,1) + exp_grid).clamp(min=-1,max=1)
        exp_image = F.grid_sample(input_images.detach(), exp_samplers)

        exp_img_pose_f, exp_img_exp_f = model.encoder(exp_image)
        exp_img_pose_loss = criterion_feature_recon_val(exp_img_pose_f, input_pose_f)
        exp_img_pose_loss = exp_img_pose_loss / input_exp_f.size(0) / input_exp_f.size(1)  #.mean()

        exp_img_exp_loss = criterion_feature_recon_val(exp_img_exp_f, target_exp_f)
        exp_img_exp_loss = exp_img_exp_loss / input_exp_f.size(0) / input_exp_f.size(1)  #.mean()

        exp_2vector = torch.cat((exp_img_exp_f, input_exp_f), 1)
        all_exp_2flow_samplers = model.expression_decoder(exp_2vector) 
        exp_flow_2samplers = all_exp_2flow_samplers[:,0:2,:,:]
        exp_2grid = np.linspace(-1,1, exp_flow_2samplers.size(2))
        exp_2grid = np.meshgrid(exp_2grid, exp_2grid)
        exp_2grid = np.stack(exp_2grid, 2)  # w x h x 2
        exp_2grid = torch.Tensor(exp_2grid).unsqueeze(0).repeat(exp_img_exp_f.size(0), 1,1,1).cuda()
        exp_2grid = Variable(exp_2grid, requires_grad=False)
        exp_2samplers = (exp_flow_2samplers.permute(0,2,3,1) + exp_2grid).clamp(min=-1,max=1)
        exp_2image = F.grid_sample(exp_image.detach(), exp_2samplers)
        exp_recon_loss = criterion_reconstruction_val(exp_2image, input_images)

        exp_sub = exp_samplers - exp_grid 
        exp_flow_tv_loss = criterion_tv_val(exp_sub)

        merge_flow = torch.cat([pose_flow_samplers.unsqueeze(4), exp_flow_samplers.unsqueeze(4)], 4)
        attention_maps = torch.cat([pose_mask.unsqueeze(4), exp_mask.unsqueeze(4)], 4)
        output_flow = (merge_flow * attention_maps.exp()).sum(dim=4) / attention_maps.exp().sum(dim=4)
        merge_grid = np.linspace(-1,1, merge_flow.size(2))
        merge_grid = np.meshgrid(merge_grid, merge_grid)
        merge_grid = np.stack(merge_grid, 2)  # w x h x 2
        merge_grid = torch.Tensor(merge_grid).unsqueeze(0).repeat(exp_vector.size(0), 1,1,1).cuda()
        merge_grid = Variable(merge_grid, requires_grad=False)
        samplers = (output_flow.permute(0,2,3,1) + merge_grid).clamp(min=-1,max=1)
        output_image = F.grid_sample(input_images.detach(), samplers)

        out_img_pose_f, out_img_exp_f = model.encoder(output_image)
        out_img_pose_loss = criterion_feature_recon_val(out_img_pose_f, target_pose_f)
        out_img_pose_loss = out_img_pose_loss / out_img_exp_f.size(0) / out_img_exp_f.size(1)  #.mean()

        out_img_exp_loss = criterion_feature_recon_val(out_img_exp_f, target_exp_f)
        out_img_exp_loss = out_img_exp_loss / out_img_exp_f.size(0) / out_img_exp_f.size(1)  #.mean()

        loss_pre_sort = criterion_reconstruction_val(output_image, target_image)
        loss = loss_pre_sort

        total_loss = w1 * loss \
                + w2 * pose_img_pose_loss  + w3 * pose_img_exp_loss \
                + w5 * exp_img_pose_loss + w6 * exp_img_exp_loss \
                + w7 * out_img_pose_loss + w8 * out_img_exp_loss \
                + w9 * exp_recon_loss + w10 * pose_recon_loss \
                +  w4 * exp_flow_tv_loss 

        t_loss += total_loss.cpu().item()
        t_reconsturction_loss += w1 * loss.cpu().item()
        t_pose_img_pose_loss += w2 * pose_img_pose_loss.cpu().item() # source -> expression flow,  target -> pose flow
        t_pose_img_exp_loss += w3 * pose_img_exp_loss.cpu().item() # source -> expression flow,  target -> pose flow
        t_exp_img_pose_loss += w5 * exp_img_pose_loss.cpu().item() # source -> expression flow,  target -> pose flow
        t_exp_img_exp_loss += w6 * exp_img_exp_loss.cpu().item() # source -> expression flow,  target -> pose flow
        t_out_img_pose_loss += w5 * out_img_pose_loss.cpu().item() # source -> expression flow,  target -> pose flow
        t_out_img_exp_loss += w6 * out_img_exp_loss.cpu().item() # source -> expression flow,  target -> pose flow
        t_exp_recon_loss += w9 * exp_recon_loss.cpu().item() # source -> expression flow,  target -> pose flow
        t_pose_recon_loss += w10 * pose_recon_loss.cpu().item() # source -> expression flow,  target -> pose flow
        t_exp_flow_tv_loss +=  w4 * exp_flow_tv_loss.cpu().item()

        if iteration == 1 or iteration == 10:
            # save input-output image and loss
            writer.add_image('Image_val/input', 
                    torchvision.utils.make_grid(input_images[0:8,:,:,:].data, nrow=4), epoch+iteration)
            writer.add_image('Image_val/output', 
                    torchvision.utils.make_grid(output_image[0:8,:,:,:].data, nrow=4), epoch+iteration)
            writer.add_image('Image_val/target', 
                    torchvision.utils.make_grid(target_image[0:8,:,:,:].data, nrow=4), epoch+iteration)
            writer.add_image('Image_val/exp', 
                    torchvision.utils.make_grid(exp_image[0:8,:,:,:].data, nrow=4), epoch+iteration)
            writer.add_image('Image_val/pose', 
                    torchvision.utils.make_grid(pose_image[0:8,:,:,:].data, nrow=4), epoch+iteration)


        if iteration % 10 == 0:
            print("val: Epoch {}: {}/{} with error {:.4f}, reconstruction loss {:.4f},\n \
                    val pose img pose loss(target<->pose img) {:.4f}, pose img exp loss(input <->pose img) {:.4f}\n \
                    val exp img pose loss(input <->exp img) {:.4f}, exp img exp loss(target<->exp img) {:.4f}\n \
                    val out img pose loss(input <->exp img) {:.4f}, out img exp loss(target<->exp img) {:.4f}\n \
                    val exp img recon loss(input <->exp img) {:.4f}, pose img recon loss(input <->pose img) {:.4f}\n \
                    val exp flow tv loss {:.4f}".
                    format(epoch, iteration, 
                        len(val_data_loader), 
                        t_loss / float(iteration), 
                        t_reconsturction_loss/float(iteration),
                        t_pose_img_pose_loss/float(iteration),
                        t_pose_img_exp_loss/float(iteration),
                        t_exp_img_pose_loss/float(iteration),
                        t_exp_img_exp_loss/float(iteration),
                        t_out_img_pose_loss/float(iteration),
                        t_out_img_exp_loss/float(iteration),
                        t_exp_recon_loss/float(iteration),
                        t_pose_recon_loss/float(iteration),
                        t_exp_flow_tv_loss/float(iteration)
                        )
                    )

    return {
            'total_error' : t_loss / float(iteration),
            'reconsturction_loss': t_reconsturction_loss/float(iteration),
            'pose_img_pose_loss': t_pose_img_pose_loss/float(iteration),
            'pose_img_exp_loss': t_pose_img_exp_loss/float(iteration),
            'exp_img_pose_loss': t_exp_img_pose_loss/float(iteration),
            'exp_img_exp_loss': t_exp_img_exp_loss/float(iteration),
            'out_img_pose_loss': t_out_img_pose_loss/float(iteration),
            'out_img_exp_loss': t_out_img_exp_loss/float(iteration),
            'exp_img_recon_loss': t_exp_recon_loss/float(iteration),
            'pose_img_recon_loss': t_pose_recon_loss/float(iteration),
            'exp_flow_tv_loss': t_exp_flow_tv_loss/float(iteration)
            }


def checkpoint(model, save_path, epoch):
    print "save model, current epoch:{0}".format(epoch)
    save_epoch = epoch + 0
    final_save_path = save_path + '_' + str(save_epoch) + '.pth'
    checkpoint_state = {
            'state_dict' : model.state_dict(), 
            'optimizer' : optimizer.state_dict(), 
            'epoch' : model.epoch, 
            'lr' : model.lr, 
            'momentum' : model.momentum, 
            'opts' : opt
            }

    torch.save(checkpoint_state, final_save_path)

def run(minpercentile=0, maxpercentile=0.5):
    scheduler = TrackLoss()
    old_model_name = ''
    print "pretraned model:{0}".format(old_model_name)
    pretrain = False
    if len(old_model_name) > 0:
        pretrain = True
        print "old model name: {0}".format(old_model_name)
    else:
        print "no pretrain."

    plateauscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    if pretrain:
        model_state = model.state_dict()
        past_state = torch.load(old_model_name)['state_dict']
        pretrained_dict = {k: v for k, v in past_state.items() if k in model_state}
        model_state.update(pretrained_dict)
        model.load_state_dict(model_state)
        print "successful load model weight"
        #optimizer.load_state_dict(torch.load(old_model_name)['optimizer'])

        minpercentile = 0.4 #percentiles.minpercentile
        maxpercentile = 0.9 #percentiles.maxpercentile

    w1 = 10
    w2 = 1 
    w3 = 1
    w4 = 0.1
    w5 = 1
    w6 = 1
    w7 = 1
    w8 = 1
    w9 = 1
    w10 = 1


    model.eval()
    with torch.no_grad():
        loss = val(-1, model, criterion_reconstruction, criterion_feature_recon, optimizer, \
                w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, w6=w6, w7=w7, w8=w8, w9=w9, w10=w10, minpercentile=minpercentile, maxpercentile=maxpercentile)
        print "epoch:{0}, current val total_error: {1}".format(-1, loss['total_error'])

    #for epoch in range(opt.continue_epoch, 2000):
    for epoch in range(1, 2000):
        model.epoch = epoch
        model.optimizer_state = optimizer.state_dict()
        model.train()
        train_loss = train(epoch, model, criterion_reconstruction, criterion_feature_recon,
                optimizer, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, w6=w6,  w7=w7, w8=w8, w9=w9, w10=w10, minpercentile=minpercentile, maxpercentile=maxpercentile)
        print "epoch:{0}, current train total_error: {1}".format(epoch, train_loss['total_error'])

        if epoch % 100 == 0:
            checkpoint(model, opt.model_epoch_path, epoch)


        model.eval()
        with torch.no_grad():
            loss = val(epoch, model, criterion_reconstruction, criterion_feature_recon, optimizer, \
                    w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, w6=w6,  w7=w7, w8=w8, w9=w9, w10=w10, minpercentile=minpercentile, maxpercentile=maxpercentile)
            print "epoch:{0}, current val total_error: {1}".format(epoch, loss['total_error'])
        scheduler.update(loss['total_error'], epoch)

        if scheduler.drop_learning_rate(epoch):
            if maxpercentile < opt.max_percentile:
                maxpercentile += opt.diff_percentile
                minpercentile += opt.diff_percentile
                scheduler = TrackLoss()
            else:
                print "lr step check.."
                plateauscheduler.step(loss['total_error'])

        print "current lr(group 0):{0}, start_percentage:{1} end_percentage:{2}".\
                format(optimizer.param_groups[0]['lr'], minpercentile, maxpercentile)

        if epoch % 3 != 0:
            continue

        writer.add_scalars('loss/train_val_total', {'train' : train_loss['total_error'], 
            'val' : loss['total_error']}, epoch)

        writer.add_scalars('loss/train_val_reconstruction', {'train' : train_loss['reconsturction_loss'], 
            'val' : loss['reconsturction_loss']}, epoch)

        writer.add_scalars('loss/train_val_pose_img_pose_loss', {'train' : train_loss['pose_img_pose_loss'], 
            'val' : loss['pose_img_pose_loss']}, epoch)
        writer.add_scalars('loss/train_val_pose_img_exp_loss', {'train' : train_loss['pose_img_exp_loss'], 
            'val' : loss['pose_img_exp_loss']}, epoch)
        writer.add_scalars('loss/train_val_exp_img_pose_loss', {'train' : train_loss['exp_img_pose_loss'], 
            'val' : loss['exp_img_pose_loss']}, epoch)
        writer.add_scalars('loss/train_val_exp_img_exp_loss', {'train' : train_loss['exp_img_exp_loss'], 
            'val' : loss['exp_img_exp_loss']}, epoch)
        writer.add_scalars('loss/train_val_out_img_pose_loss', {'train' : train_loss['out_img_pose_loss'], 
            'val' : loss['out_img_pose_loss']}, epoch)
        writer.add_scalars('loss/train_val_out_img_exp_loss', {'train' : train_loss['out_img_exp_loss'], 
            'val' : loss['out_img_exp_loss']}, epoch)
        writer.add_scalars('loss/train_val_exp_img_recon_loss', {'train' : train_loss['exp_img_recon_loss'], 
            'val' : loss['exp_img_recon_loss']}, epoch)

        writer.add_scalars('loss/train_val_exp_flow_tv_loss', {'train' : train_loss['exp_flow_tv_loss'], 
            'val' : loss['exp_flow_tv_loss']}, epoch)


if __name__ == '__main__':
    if opt.load_old_model:
        model.load_state_dict(torch.load(opt.old_model)['state_dict'])
        percentiles = torch.load(opt.old_model)['opts']
        minpercentile = percentiles.minpercentile
        maxpercentile = percentiles.maxpercentile

        run(minpercentile, maxpercentile)

    else:
        run()
