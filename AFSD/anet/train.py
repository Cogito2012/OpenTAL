import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from AFSD.common.anet_dataset import ANET_Dataset, detection_collate
from torch.utils.data import DataLoader
from AFSD.anet.BDNet import BDNet
from AFSD.anet.multisegment_loss import MultiSegmentLoss
from AFSD.common.config import config
from tensorboardX import SummaryWriter

batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
max_epoch = config['training']['max_epoch']
num_classes = 2
checkpoint_path = config['training']['checkpoint_path']
focal_loss = config['training']['focal_loss']
edl_loss = config['training']['edl_loss'] if 'edl_loss' in config['training'] else False
edl_config = config['training']['edl_config'] if 'edl_config' in config['training'] else None
cls_loss_type = 'edl' if edl_loss else 'focal' # by default, we use focal loss
os_head = config['model']['os_head'] if 'os_head' in config['model'] else False
random_seed = config['training']['random_seed']

train_state_path = os.path.join(checkpoint_path, 'training')
if not os.path.exists(train_state_path):
    os.makedirs(train_state_path)
if config['testing']['split'] == 0:
    tensorboard_path = os.path.join(checkpoint_path, 'tensorboard')
    os.makedirs(tensorboard_path, exist_ok=True)

resume = config['training']['resume'] 
# config['training']['ssl'] = 0.1


def print_training_info():
    print('batch size: ', batch_size)
    print('learning rate: ', learning_rate)
    print('weight decay: ', weight_decay)
    print('max epoch: ', max_epoch)
    print('checkpoint path: ', checkpoint_path)
    print('loc weight: ', config['training']['lw'])
    print('cls weight: ', config['training']['cw'])
    print('ctr weight: ', config['training']['ctw'])
    print('iou weight: ', config['training']['piou'])
    print('ssl weight: ', config['training']['ssl'])
    print('piou:', config['training']['piou'])
    print('resume: ', resume)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


GLOBAL_SEED = 1


def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)


def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states


def set_rng_state(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(states[3])


def update_the_latest(src_file, dest_file):
    # source file must exist
    assert os.path.exists(src_file), "src file does not exist!"
    # destinate file should be removed first if exists
    if os.path.lexists(dest_file):
        os.remove(dest_file)
    os.symlink(src_file, dest_file)


def save_model(epoch, model, optimizer):
    # save the model weights
    model_file = os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(epoch))
    torch.save(model.module.state_dict(), model_file)
    update_the_latest(model_file,
                     os.path.join(checkpoint_path, 'checkpoint-latest.ckpt'))
    # save the training status
    state_file = os.path.join(train_state_path, 'checkpoint_{}.ckpt'.format(epoch))
    torch.save({'optimizer': optimizer.state_dict(),
                'state': get_rng_states()},
                state_file)
    update_the_latest(state_file,
                     os.path.join(train_state_path, 'checkpoint_latest.ckpt'))


def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(resume))
        model.module.load_state_dict(torch.load(model_path))
        train_path = os.path.join(train_state_path, 'checkpoint_{}.ckpt'.format(resume))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict['optimizer'])
        set_rng_state(state_dict['state'])
    return start_epoch


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    return total_norm
def calc_bce_loss(start, end, scores):
    start = torch.tanh(start).mean(-1)
    end = torch.tanh(end).mean(-1)
    loss_start = F.binary_cross_entropy(start.view(-1),
                                        scores[:, 1].contiguous().view(-1).cuda(),
                                        reduction='mean')
    loss_end = F.binary_cross_entropy(end.view(-1),
                                      scores[:, 2].contiguous().view(-1).cuda(),
                                      reduction='mean')
    return loss_start, loss_end


def forward_one_epoch(net, clips, targets, scores=None, training=True, ssl=True):
    clips = clips.cuda()
    targets = [t.cuda() for t in targets]

    if training:
        if ssl:
            output_dict = net(clips, proposals=targets, ssl=ssl)
        else:
            output_dict = net(clips, ssl=False)
    else:
        with torch.no_grad():
            output_dict = net(clips)

    if ssl:
        anchor, positive, negative = output_dict
        loss_ = []
        weights = [1, 0.1, 0.1]
        for i in range(3):
            loss_.append(nn.TripletMarginLoss()(anchor[i], positive[i], negative[i]) * weights[i])
        trip_loss = torch.stack(loss_).sum(0)
        return trip_loss
    else:
        loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct, loss_act, loss_prop_act = CPD_Loss(
            [output_dict['loc'], output_dict['conf'],
             output_dict['prop_loc'], output_dict['prop_conf'],
             output_dict['center'], output_dict['priors'], output_dict['act'], output_dict['prop_act']],
            targets)
        loss_start, loss_end = calc_bce_loss(output_dict['start'], output_dict['end'], scores)
        versions = torch.__version__.split('.')
        if int(versions[0]) == 1 and int(versions[1]) >= 6: # version later than torch 1.6.0
            scores_ = F.interpolate(scores, scale_factor=1.0 / 8, recompute_scale_factor=True)
        else:
            scores_ = F.interpolate(scores, scale_factor=1.0 / 8)
        loss_start_loc_prop, loss_end_loc_prop = calc_bce_loss(output_dict['start_loc_prop'],
                                                               output_dict['end_loc_prop'],
                                                               scores_)
        loss_start_conf_prop, loss_end_conf_prop = calc_bce_loss(output_dict['start_conf_prop'],
                                                                 output_dict['end_conf_prop'],
                                                                 scores_)
        loss_start = loss_start + 0.1 * (loss_start_loc_prop + loss_start_conf_prop)
        loss_end = loss_end + 0.1 * (loss_end_loc_prop + loss_end_conf_prop)
        return loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct, loss_start, loss_end, loss_act, loss_prop_act


def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num, training=True):
    if training:
        net.train()
    else:
        net.eval()

    loss_loc_val = 0
    loss_conf_val = 0
    loss_prop_l_val = 0
    loss_prop_c_val = 0
    loss_ct_val = 0
    loss_start_val = 0
    loss_end_val = 0
    loss_trip_val = 0
    loss_contras_val = 0
    cost_val = 0
    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(pbar):
            loss_l, loss_c, loss_prop_l, loss_prop_c, \
                loss_ct, loss_start, loss_end, loss_act, loss_act_prop = forward_one_epoch(
                net, clips, targets, scores, training=training, ssl=False)

            loss_l = loss_l * config['training']['lw']
            loss_c = loss_c * config['training']['cw']
            loss_prop_l = loss_prop_l * config['training']['lw']
            loss_prop_c = loss_prop_c * config['training']['cw']
            loss_ct = loss_ct * config['training']['ctw']
            cost = loss_l + loss_c + loss_prop_l + loss_prop_c + loss_ct + loss_start + loss_end
            if os_head:
                loss_act = loss_act * config['training']['actw']
                loss_act_prop = loss_act_prop * config['training']['actw']
                cost = cost + loss_act + loss_act_prop

            if flags[0]:
                loss_trip = forward_one_epoch(net, ssl_clips, ssl_targets, training=training,
                                              ssl=True)
                loss_trip *= config['training']['ssl']
                cost = cost + loss_trip
                loss_trip_val += loss_trip.cpu().detach().numpy()

            cur_iter = i * epoch_step_num + n_iter
            if training:
                optimizer.zero_grad()
                cost.backward()
                grad_norm = get_grad_norm(net)
                if config['testing']['split'] == 0:
                    tb_writer.add_scalars(f'stats/grad_norm', {'grad_norm': grad_norm.mean().item()}, cur_iter)
                optimizer.step()

            # record the loss in tensorboards
            if config['testing']['split'] == 0:
                tb_writer.add_scalars(f'train_loss/coarse/loss_loc', {'loss_loc': loss_l.mean().item()}, cur_iter)
                tb_writer.add_scalars(f'train_loss/coarse/loss_cls', {'loss_cls': loss_c.mean().item()}, cur_iter)
                tb_writer.add_scalars(f'train_loss/refined/loss_loc', {'loss_loc': loss_prop_l.mean().item()}, cur_iter)
                tb_writer.add_scalars(f'train_loss/refined/loss_cls', {'loss_cls': loss_prop_c.mean().item()}, cur_iter)
                tb_writer.add_scalars(f'train_loss/regularizer/loss_quality', {'loss_q': loss_ct.mean().item()}, cur_iter)
                tb_writer.add_scalars(f'train_loss/regularizer/loss_start', {'loss_start': loss_start.mean().item()}, cur_iter)
                tb_writer.add_scalars(f'train_loss/regularizer/loss_end', {'loss_end': loss_end.mean().item()}, cur_iter)
                if flags[0]:
                    tb_writer.add_scalars(f'train_loss/regularizer/loss_trip', {'loss_trip': loss_trip.mean().item()}, cur_iter)
                tb_writer.add_scalars(f'train_loss/loss_total', {'loss_total': cost.mean().item()}, cur_iter)
                if os_head:
                    tb_writer.add_scalars(f'train_loss/coarse/loss_act', {'loss_act': loss_act.mean().item()}, cur_iter)
                    tb_writer.add_scalars(f'train_loss/refined/loss_act_prop', {'loss_act': loss_act_prop.mean().item()}, cur_iter)

            loss_loc_val += loss_l.cpu().detach().numpy()
            loss_conf_val += loss_c.cpu().detach().numpy()
            loss_prop_l_val += loss_prop_l.cpu().detach().numpy()
            loss_prop_c_val += loss_prop_c.cpu().detach().numpy()
            loss_ct_val += loss_ct.cpu().detach().numpy()
            loss_start_val += loss_start.cpu().detach().numpy()
            loss_end_val += loss_end.cpu().detach().numpy()
            cost_val += cost.cpu().detach().numpy()
            pbar.set_postfix(loss='{:.5f}'.format(float(cost.cpu().detach().numpy())))

    loss_loc_val /= (n_iter + 1)
    loss_conf_val /= (n_iter + 1)
    loss_prop_l_val /= (n_iter + 1)
    loss_prop_c_val /= (n_iter + 1)
    loss_ct_val /= (n_iter + 1)
    loss_start_val /= (n_iter + 1)
    loss_end_val /= (n_iter + 1)
    loss_trip_val /= (n_iter + 1)
    cost_val /= (n_iter + 1)

    if training and epoch > 10:
        prefix = 'Train'
        save_model(epoch, net, optimizer)
    else:
        prefix = 'Val'

    plog = 'Epoch-{} {} Loss: Total - {:.5f}, loc - {:.5f}, conf - {:.5f}, prop_loc - {:.5f}, ' \
           'prop_conf - {:.5f}, IoU - {:.5f}, start - {:.5f}, end - {:.5f}'.format(
        i, prefix, cost_val, loss_loc_val, loss_conf_val, loss_prop_l_val, loss_prop_c_val,
        loss_ct_val, loss_start_val, loss_end_val
    )
    plog = plog + ', Triplet - {:.5f}'.format(loss_trip_val)
    print(plog)


if __name__ == '__main__':
    print_training_info()
    set_seed(random_seed)
    """
    Setup model
    """
    use_edl = config['model']['use_edl'] if 'use_edl' in config['model'] else False
    net = BDNet(in_channels=config['model']['in_channels'],
                backbone_model=config['model']['backbone_model'], use_edl=use_edl)
    net = nn.DataParallel(net, device_ids=[0]).cuda()

    """
    Setup optimizer
    """
    optimizer = torch.optim.Adam([
        {'params': net.module.backbone.parameters(),
         'lr': learning_rate * 0.1,
         'weight_decay': weight_decay},
        {'params': net.module.coarse_pyramid_detection.parameters(),
         'lr': learning_rate,
         'weight_decay': weight_decay}
    ])

    """
    Setup loss
    """
    piou = config['training']['piou']
    num_cls = num_classes - 1 if os_head else num_classes
    CPD_Loss = MultiSegmentLoss(num_cls, piou, 1.0, cls_loss_type=cls_loss_type, edl_config=edl_config, os_head=os_head)

    """
    Setup dataloader
    """
    train_dataset = ANET_Dataset(config['dataset']['training']['video_info_path'],
                                 config['dataset']['training']['video_mp4_path'],
                                 config['dataset']['training']['clip_length'],
                                 config['dataset']['training']['crop_size'],
                                 config['dataset']['training']['clip_stride'],
                                 channels=config['model']['in_channels'],
                                 binary_class=True)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=4, worker_init_fn=worker_init_fn,
                                   collate_fn=detection_collate, pin_memory=True, drop_last=True)
    epoch_step_num = len(train_dataset) // batch_size
    """
    Setup tensorboard writer (only for the split_0)
    """
    if config['testing']['split'] == 0:
        # tensorboard logging
        tb_writer = SummaryWriter(tensorboard_path)

    """
    Start training
    """
    start_epoch = resume_training(resume, net, optimizer)

    for i in range(start_epoch, max_epoch + 1):
        if cls_loss_type == 'edl':
            CPD_Loss.cls_loss.epoch = i
            CPD_Loss.cls_loss.total_epoch = max_epoch
        run_one_epoch(i, net, optimizer, train_data_loader, len(train_dataset) // batch_size)