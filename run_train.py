import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Unet import UNet
from carvana_dataset import CarvanaDataset



def train():
    batch_size = 2
    grad_accu_times = 16
    img_csv_file = 'train_masks.csv'
    train_img_dir = 'train'
    train_mask_dir = 'train_masks_png'
    dataset = CarvanaDataset(img_csv_file, train_img_dir, train_mask_dir)

    model = UNet().cuda()



    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
    opt = torch.optim.RMSprop(model.parameters())
    opt.zero_grad()

    epoch = 0
    forward_times = 0
    while True:
        data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

        for idx, batch_data in enumerate(data_loader):

            batch_input = Variable(batch_data['img']).cuda()
            batch_gt_mask = Variable(batch_data['mask']).cuda()



            pred_mask = model(batch_input)
            forward_times += 1

            loss = loss_fn(pred_mask, batch_gt_mask)
            loss.backward()
            print('Epoch {:>3} | Batch {:>5} | Loss {:>1.5f}'.format(epoch+1, idx+1, loss.cpu().data.numpy()[0]))


            if forward_times == grad_accu_times:
                opt.step()
                opt.zero_grad()
                forward_times = 0
                print('\nUpdate weights ... \n')


        del data_loader
        epoch += 1






if __name__ == '__main__':
    train()