import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Unet import UNet
from carvana_dataset import CarvanaDataset
import torch.nn.functional as F
import numpy as np
import cv2


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1.0 - (((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth)))

def show_example(img, gt_mask, pred_mask):
    img_np = img.cpu().data.numpy()
    img_np = np.transpose(img_np, [1,2,0])
    img_np += 128
    img_np = cv2.resize(img_np, (512,512))
    gt_mask_np = gt_mask.cpu().data.numpy()
    gt_mask_np = np.transpose(gt_mask_np, [1,2,0]) * 255.0
    gt_mask_np = np.repeat(gt_mask_np, 3, 2)
    gt_mask_np = cv2.resize(gt_mask_np, (512,512))
    pred_mask = pred_mask.cpu().data.numpy()
    pred_mask = np.transpose(pred_mask, [1,2,0]) * 255.0
    pred_mask = np.repeat(pred_mask, 3, 2)
    pred_mask = cv2.resize(pred_mask, (512,512))
    img = np.concatenate((img_np, gt_mask_np, pred_mask), axis=1)

    cv2.imshow('i', img.astype(np.uint8))
    cv2.waitKey(10)

def train():
    batch_size = 2
    grad_accu_times = 8
    init_lr = 0.01
    img_csv_file = 'train_masks.csv'
    train_img_dir = 'train'
    train_mask_dir = 'train_masks_png'
    dataset = CarvanaDataset(img_csv_file, train_img_dir, train_mask_dir)

    model = UNet().cuda()



    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
    opt = torch.optim.RMSprop(model.parameters(), lr=init_lr)
    opt.zero_grad()

    epoch = 0
    forward_times = 0
    for epoch in range(30):
        data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

        lr = init_lr * (0.1 ** (epoch // 10))
        for param_group in opt.param_groups:
            param_group['lr'] = lr

        for idx, batch_data in enumerate(data_loader):

            batch_input = Variable(batch_data['img']).cuda()
            batch_gt_mask = Variable(batch_data['mask']).cuda()


            pred_mask = model(batch_input)
            forward_times += 1

            if (idx+1) % 10 == 0:
                show_example(batch_input[0], batch_gt_mask[0], F.sigmoid(pred_mask[0]))


            loss = loss_fn(pred_mask, batch_gt_mask)
            loss += dice_loss(F.sigmoid(pred_mask), batch_gt_mask)
            loss.backward()
            print('Epoch {:>3} | Batch {:>5} | lr {:>1.5f} | Loss {:>1.5f} '.format(epoch+1, idx+1, lr, loss.cpu().data.numpy()[0]))


            if forward_times == grad_accu_times:
                opt.step()
                opt.zero_grad()
                forward_times = 0
                print('\nUpdate weights ... \n')

        if (epoch+1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict(),
            }
            torch.save(checkpoint, 'unet1024-{}'.format(epoch+1))
        del data_loader






if __name__ == '__main__':
    train()