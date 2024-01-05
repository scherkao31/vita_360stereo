from __future__ import print_function
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models import LCV_ours_sub3

parser = argparse.ArgumentParser(description='360SD-Net')
parser.add_argument('--maxdisp', type=int, default=68, help='maxium disparity')
parser.add_argument('--model', default='360SDNet', help='select model')
parser.add_argument('--datapath', default='data/MP3D/train/', help='datapath')
parser.add_argument('--datapath_val',
                    default='data/MP3D/val/',
                    help='datapath for validation')
parser.add_argument('--batch',
                    type=int,
                    default=16,
                    help='number of batch to train')
parser.add_argument('--checkpoint', default=None, help='load checkpoint path')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--real',
                    action='store_true',
                    default=False,
                    help='adapt to real world images')
parser.add_argument('--SF3D',
                    action='store_true',
                    default=False,
                    help='read stanford3D data')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# import dataloader ------------------------------
from dataloader import filename_loader as lt
if args.real:
    from dataloader import grayscale_Loader as DA
    print("Real World image loaded!!!")
else:
    from dataloader import RGB_Loader as DA
    print("Synthetic data image loaded!!!")
# -------------------------------------------------

# Random Seed -----------------------------
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# ------------------------------------------

image_size_x = 512#512#1024   #CHANGED 
image_size_y = 3840#1024#2048 #CHANGED 
# Create Angle info ------------------------------------------------
# Y angle
angle_y = np.array([(i - 0.5) / image_size_x * 180 for i in range(image_size_x/2, -image_size_x/2, -1)])
angle_ys = np.tile(angle_y[:, np.newaxis, np.newaxis], (1, image_size_y, 1))
equi_info = angle_ys
# -------------------------------------------------------------------


# Load Data ---------------------------------------------------------
train_up_img, train_down_img, train_up_disp, valid_up_img, valid_down_img, valid_up_disp = lt.dataloader(
    args.datapath, args.datapath_val)
'''
### If 2 seqs ###
filepath1 = '/scratch/izar/scherkao/stereo360/360epfl_seq1/'
filepath2 = '/scratch/izar/scherkao/stereo360/360epfl_seq2/'
filepath3 = '/scratch/izar/scherkao/stereo360/360epfl_seq3/'
valpath1 = '/scratch/izar/scherkao/stereo360/360epfl_seq1_val/'
valpath2 = '/scratch/izar/scherkao/stereo360/360epfl_seq2_val/'
valpath3 = '/scratch/izar/scherkao/stereo360/360epfl_seq3_val/'
#train_up_img, train_down_img, train_up_disp, valid_up_img, valid_down_img, valid_up_disp = lt.dataloader_2seqs(
 #   filepath1, filepath2, valpath1, valpath2)
train_up_img, train_down_img, train_up_disp, valid_up_img, valid_down_img, valid_up_disp = lt.dataloader_3seqs(
    filepath1, filepath2, filepath3, valpath1, valpath2, valpath3)
#'''
Equi_infos = equi_info
TrainImgLoader = torch.utils.data.DataLoader(DA.myImageFolder(
    Equi_infos, train_up_img, train_down_img, train_up_disp, True),
                                             batch_size=args.batch,
                                             shuffle=True,
                                             num_workers=8,
                                             drop_last=False)

ValidImgLoader = torch.utils.data.DataLoader(DA.myImageFolder(
    Equi_infos, valid_up_img, valid_down_img, valid_up_disp, False),
                                             batch_size=args.batch,
                                             shuffle=False,
                                             num_workers=4,
                                             drop_last=False)
# -----------------------------------------------------------------------------------------

# Load model ----------------------------------------------
if args.model == '360SDNet':
    model = LCV_ours_sub3(args.maxdisp)
else:
    raise NotImplementedError('Model Not Implemented!!!')
# ----------------------------------------------------------

# assign initial value of filter cost volume ---------------------------------
init_array = np.zeros((1, 1, 7, 1))  # 7 of filter
init_array[:, :, 3, :] = 28. / 540
init_array[:, :, 2, :] = 512. / 540
model.forF.forfilter1.weight = torch.nn.Parameter(torch.Tensor(init_array))
# -----------------------------------------------------------------------------

# Multi_GPU for model ----------------------------
if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()
# -------------------------------------------------

# Load checkpoint --------------------------------
print("Checkpoint : ", args.checkpoint)
if args.checkpoint is not None:
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict['state_dict'])
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
# -------------------------------------------------

def compute_D1_error_depth(output, depth_true, mask, threshold_percentage=0.05):
    """
    Compute the D1 error for depth maps in meters, using a percentage of the true depth as the threshold.

    Parameters:
    output (torch.Tensor): The estimated depth map.
    depth_true (torch.Tensor): The ground truth depth map.
    mask (torch.Tensor): The mask to specify valid pixels for comparison.
    threshold_percentage (float): Percentage of the true depth to set as the threshold.

    Returns:
    float: The D1 error percentage.
    """
    # Calculate absolute depth error
    abs_depth_error = torch.abs(output[mask] - depth_true[mask])

    # Calculate threshold as a percentage of the true depth
    threshold = threshold_percentage * depth_true[mask]

    # Count pixels where the depth error exceeds the threshold
    error_pixels = torch.sum(abs_depth_error > threshold)

    # Calculate the D1 error percentage
    d1_error = 100.0 * error_pixels / mask.sum()

    return d1_error.item()


def compute_threshold_accuracy(output, depth_true, mask, threshold = 1.25):
    """
    Compute the Threshold Accuracy for depth maps in meters.

    Parameters:
    output (torch.Tensor): The estimated depth map.
    depth_true (torch.Tensor): The ground truth depth map.
    mask (torch.Tensor): The mask to specify valid pixels for comparison.
    thresholds (list of float): Factors of the true depth for calculating threshold accuracy.

    Returns:
    list of float: The Threshold Accuracy percentages for each threshold.
    """
    # Ensure the output and depth_true have the same shape and mask is boolean
    #assert output.shape == depth_true.shape and mask.dtype == torch.bool

    # Calculate the ratio of predicted depth to true depth
    depth_ratio = output[mask] / depth_true[mask]


    # Calculate accuracy for each threshold
    # Count pixels where the predicted depth is within the threshold factor of the true depth
    correct_pixels = torch.sum((depth_ratio >= 1 / threshold) & (depth_ratio <= threshold))
    
    # Calculate the accuracy percentage
    accuracy = 100.0 * correct_pixels / mask.sum()

    return accuracy.item()

# Valid Function -----------------------
def val(imgU, imgD, disp_true):
    model.eval()
    imgU = Variable(torch.FloatTensor(imgU.float()))
    imgD = Variable(torch.FloatTensor(imgD.float()))
    if imgD.shape[2] == 512: #ADDED
        #imgU = F.pad(imgU, (0, 0, 6, 6), "constant", 0)  # pad left, right, top, bottom #ADDED
        #imgD = F.pad(imgD, (0, 0, 6, 6), "constant", 0) #ADDED
        #disp_true = F.pad(disp_true, (0, 0, 6, 6), "constant", 0)/1000 #ADDED
        imgU = F.pad(imgU, (128, 128, 0, 0), "constant", 0)  # pad left, right, top, bottom #ADDED
        imgD = F.pad(imgD, (128, 128, 0, 0), "constant", 0) #ADDED
        disp_true = F.pad(disp_true, (128, 128, 0, 0), "constant", 0)#/1000 #ADDED
    #disp_true = disp_true/1000 #ADDED
    #disp_true = np.ascontiguousarray(disp_true, dtype=np.float32)

    #print("Validation input shapes mm: ", imgU.shape, imgD.shape, disp_true.shape)
    # cuda?
    if args.cuda:
        imgU, imgD = imgU.cuda(), imgD.cuda()
    # mask value
    max_disp_ici = 1000#1000
    min_disp_ici = 0
    #mask = (disp_true < args.maxdisp) & (disp_true > 0)
    mask = (disp_true < max_disp_ici) & (disp_true > 0) #& (disp_true > min_disp_ici)

    with torch.no_grad():
        output3 = model(imgU, imgD)

    output = torch.squeeze(output3.data.cpu(), 1)
    if len(disp_true[mask]) == 0:
        loss = 0
        mae = 0
        mare = 0
        rmse = 0
        d1_1 = 0
        d1_2 = 0
        d1_3 = 0
    else:
        mae = torch.mean(torch.abs(output[mask] - disp_true[mask]))  # end-point-error
        mare = torch.mean(torch.abs((output[mask] - disp_true[mask]) / disp_true[mask]))
        rmse = torch.sqrt(torch.mean((output[mask] - disp_true[mask]) ** 2))

        d1_1 = compute_threshold_accuracy(output, disp_true, mask, threshold = 1.25**1)
        d1_2 = compute_threshold_accuracy(output, disp_true, mask, threshold = 1.25**2)
        d1_3 = compute_threshold_accuracy(output, disp_true, mask, threshold = 1.25**3)


    return mae, output, mare, rmse, d1_1, d1_2, d1_3



# Main Function ----------------------------------
def main():
    global_val = 0
        
    # Valid --------------------------------------------------
    total_val_loss = 0
    total_val_rmse = 0
    total_val_mare = 0
    total_d1_1 = 0
    total_d1_2 = 0
    total_d1_3 = 0

    # print("pret")
    for batch_idx, (imgU, imgD, disp) in tqdm(enumerate(ValidImgLoader),
                                                desc='Valid iter'):
        
        #print("yoy")
        val_loss, val_output, val_mare, val_rmse, d1_1, d1_2, d1_3 = val(imgU, imgD, disp)
        # Loss ---------------------------------
        total_val_loss += val_loss
        total_val_mare += val_mare
        total_val_rmse += val_rmse
        total_d1_1 += d1_1
        total_d1_2 += d1_2
        total_d1_3 += d1_3
        #total_val_crop_rmse += val_crop_rmse
        # ---------------------------------------
        # Step ------
        global_val += 1
        # ------------

    print('Total validation : mae, mare, rmse, d1, d2, d3 : ', total_val_loss / len(ValidImgLoader), total_val_mare / len(ValidImgLoader), total_val_rmse / len(ValidImgLoader),
           total_d1_1 / len(ValidImgLoader), total_d1_2 / len(ValidImgLoader), total_d1_3 / len(ValidImgLoader))
    #print('Total validation crop 26 depth rmse :', total_val_crop_rmse / len(ValidImgLoader))
# ----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
