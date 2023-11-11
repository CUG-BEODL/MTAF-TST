import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
from data.dataset.Remote import RemoteDataset
import numpy as np
import os
import random
import datetime
from utils.metrics import runningScore
from utils.my_logging import Logger
from cross_fusion import MTAF_TST
from cross_fusion_wo_Aux_TST import Cross_Wo_Aux_TST
from cross_fusion_wo_Aux_TST_MTAF import Cross_Wo_Aux_TST_MTAF
from optic_TST import Optic_TST
from sar_TST import Sar_TST
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())

class LayerActivations:
    features = None
    def __init__(self, model):
        self.hook = model.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(3047)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--root_dir", default=None, type=str)
    parser.add_argument("--order", default=0, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--patch_size", default=100, type=int)
    parser.add_argument("--lradj", default="type1", type=str)
    parser.add_argument("--T_P", default=5, type=int)
    parser.add_argument("--T_S", default=5, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--net_G", default="MTAF_TST", type=str)
    parser.add_argument("--last_epoch", default=-1, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--do", default="train", type=str)
    parser.add_argument("--dropout", default=0.01, type=float)
    parser.add_argument("--lamdam", default=0.5, type=float)
    parser.add_argument("--dim_weight", default=128, type=int)
    parser.add_argument("--in_channels_P", default=9, type=int)
    parser.add_argument("--in_channels_S", default=9, type=int)
    parser.add_argument("--classes", default=3, type=int)

    args = parser.parse_args()
    return args

def build_model(args):
    if(args.net_G=="MTAF_TST"):
        return MTAF_TST(T=[args.T_P,args.T_S], input_c=[args.in_channels_P,args.in_channels_S], image_size=args.patch_size, label_c=args.classes, leve_dims=[32,64], dropout=args.dropout, dim_weight=args.dim_weight)
    elif(args.net_G=="Wo_Aux_TST"):
        return Cross_Wo_Aux_TST(T=[args.T_P,args.T_S], input_c=[args.in_channels_P,args.in_channels_S], image_size=args.patch_size, label_c=args.classes, leve_dims=[32,64], dropout=args.dropout, dim_weight=args.dim_weight)
    elif(args.net_G=="Wo_Aux_TST_MTAF"):
        return Cross_Wo_Aux_TST_MTAF(T=[args.T_P,args.T_S], input_c=[args.in_channels_P,args.in_channels_S], image_size=args.patch_size, label_c=args.classes, leve_dims=[32,64], dropout=args.dropout, dim_weight=args.dim_weight)
    elif(args.net_G=="Optic_TST"):
        return Optic_TST(T=[args.T_P,args.T_S], input_c=[args.in_channels_P,args.in_channels_S], image_size=args.patch_size, label_c=args.classes, leve_dims=[32,64], dropout=args.dropout, dim_weight=args.dim_weight)
    elif(args.net_G=="Sar_TST"):
        return Sar_TST(T=[args.T_P,args.T_S], input_c=[args.in_channels_P,args.in_channels_S], image_size=args.patch_size, label_c=args.classes, leve_dims=[32,64], dropout=args.dropout, dim_weight=args.dim_weight)



def train(model, args):
    model.train()
    best_f1 = 0
    train_set = RemoteDataset(root_dir=args.root_dir, order=args.order, split="train")
    trainloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    val_set = RemoteDataset(root_dir=args.root_dir, order=args.order, split="val")
    valloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.lradj=='type1':
        update_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, verbose=True,  min_lr=0)
    elif args.lradj=='type2':
        update_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=args.last_epoch)
    logger = Logger(args.net_G,args.root_dir,order=args.order)
    logger.info(args)
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

    epoch = 0
    step = 0
    report_loss = 0.0
    loss_list = []
    miou_list = []
    f1_list = []
    epoch_list = []
    training_metrics_val = runningScore(args.classes)
    while epoch < args.epochs:
        epoch += 1
        model.train()
        logger.info(f"Epoch:{epoch}/{args.epochs}, lr={optimizer.state_dict()['param_groups'][0]['lr']}")
        for batch in tqdm(trainloader, desc=f"Train:{epoch}/{args.epochs}", ncols=100):
            optimizer.zero_grad()
            x1, x2, mask = (
                batch["P"].to(device),
                batch["S"].to(device),
                batch["L"].to(device),
            )
            if(args.net_G=="MTAF_TST"):
                x1 = x1.reshape(x1.shape[0],args.T_P,args.in_channels_P,args.patch_size,args.patch_size)
                x2 = x2.reshape(x2.shape[0],args.T_S,args.in_channels_S,args.patch_size,args.patch_size)
                pred,pred_optic,pred_sar = model(x1, x2)
                pred_img_1 = torch.argmax((pred.cpu()), 1).byte().numpy()
                gt = mask.data.cpu().numpy()
                training_metrics_val.update(gt, pred_img_1)
                loss_total = loss_func(pred, mask)
                loss_optic = loss_func(pred_optic, mask)
                loss_sar = loss_func(pred_sar, mask)
                loss = loss_total+loss_optic+loss_sar
                report_loss += loss.item()
                loss.backward()
                optimizer.step()
                step += 1

            elif(args.net_G=="Optic_TST" or args.net_G=="Sar_TST" or args.net_G=="Wo_Aux_TST" or args.net_G=="Wo_Aux_TST_MTAF"):
                x1 = x1.reshape(x1.shape[0],args.T_P,args.in_channels_P,args.patch_size,args.patch_size)
                x2 = x2.reshape(x2.shape[0],args.T_S,args.in_channels_S,args.patch_size,args.patch_size)
                pred = model(x1,x2)
                pred_img_1 = torch.argmax((pred.cpu()), 1).byte().numpy()
                gt = mask.data.cpu().numpy()
                training_metrics_val.update(gt, pred_img_1)
                loss = loss_func(pred, mask)
                report_loss += loss.item()
                loss.backward()
                optimizer.step()
                step += 1

        score_train, mean_f1_train, m_iou_train = training_metrics_val.get_scores()
        for k, v in score_train.items():
            print(f"Train {k}{v}")
        logger.info(f"Train loss = {report_loss / step}")
        loss_list.append(report_loss / step)
        epoch_list.append(epoch)
        step = 0
        report_loss = 0.0
        logger.plot_loss(epoch_list, loss_list)
        score, mean_f1, m_iou = eval(model, args=args,valloader=valloader)
        f1_list.append(mean_f1)
        miou_list.append(m_iou)
        logger.plot_acc(epoch_list,miou_list,f1_list)
        if args.lradj=='type1':
            update_lr.step(mean_f1)
        elif args.lradj=='type2':
            update_lr.step()
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            logger.save_model(model)
            logger.info("\n")
            for k, v in score_train.items():
                logger.info(f"Train {k}{v}")
            for k, v in score.items():
                logger.info(f"{k}{v}")
            logger.info("\n")

    model.load_state_dict(torch.load(logger.model_file))
    score, mean_f1=test(model, args,logger.model_file)
    logger.info("###############################TEST SCORE################################")
    for k, v in score.items():
        logger.info(f"{k}{v}")


def eval(model, do=train, args=None, valloader=None):
    model.eval()
    running_metrics_val = runningScore(args.classes)#当标签类别数目改变时需要进行修改
    with torch.no_grad():
        for batch in tqdm(valloader, desc=f"Valid", ncols=100):
            x1, x2, mask = (
                batch["P"].to(device),
                batch["S"].to(device),
                batch["L"].to(device),
            )
            if(args.net_G=="MTAF_TST"):
                x1 = x1.reshape(x1.shape[0],args.T_P,args.in_channels_P,args.patch_size,args.patch_size)
                x2 = x2.reshape(x2.shape[0],args.T_S,args.in_channels_S,args.patch_size,args.patch_size)
                pred_img_1 ,pred_optic,pred_sar= model(x1, x2)
                pred = torch.argmax((pred_img_1.cpu()), 1).byte().numpy()
                print("pred",np.unique(pred))
                gt = mask.data.cpu().numpy()
                running_metrics_val.update(gt, pred)

            elif(args.net_G=="Optic_TST" or args.net_G=="Sar_TST" or args.net_G=="Wo_Aux_TST" or args.net_G=="Wo_Aux_TST_MTAF"):
                x1 = x1.reshape(x1.shape[0],args.T_P,args.in_channels_P,args.patch_size,args.patch_size)
                x2 = x2.reshape(x2.shape[0],args.T_S,args.in_channels_S,args.patch_size,args.patch_size)
                pred_img_1 = model(x1,x2)
                pred = torch.argmax((pred_img_1.cpu()), 1).byte().numpy()
                gt = mask.data.cpu().numpy()
                running_metrics_val.update(gt, pred)
    score, mean_f1, m_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(f"{k}{v}")
    return score, mean_f1, m_iou


def test(model, args, model_name):
    print(args)
    val_set = RemoteDataset(root_dir=args.root_dir, order=args.order, split="test")
    valloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    model.eval()
    running_metrics_val = runningScore(args.classes)
    with torch.no_grad():
        for batch in tqdm(valloader, desc=f"Valid", ncols=100):
            x1, x2, mask = (
                batch["P"].to(device),
                batch["S"].to(device),
                batch["L"].to(device),
            )

            if(args.net_G=="MTAF_TST"):
                x1 = x1.reshape(x1.shape[0],args.T_P,args.in_channels_P,args.patch_size,args.patch_size)
                x2 = x2.reshape(x2.shape[0],args.T_S,args.in_channels_S,args.patch_size,args.patch_size)
                pred_img_1 ,pred_optic,pred_sar= model(x1, x2)
            elif(args.net_G=="Optic_TST" or args.net_G=="Sar_TST" or args.net_G=="Wo_Aux_TST" or args.net_G=="Wo_Aux_TST_MTAF"):
                x1 = x1.reshape(x1.shape[0],args.T_P,args.in_channels_P,args.patch_size,args.patch_size)
                x2 = x2.reshape(x2.shape[0],args.T_S,args.in_channels_S,args.patch_size,args.patch_size)
                pred_img_1 = model(x1,x2)
            pred = torch.argmax((pred_img_1.cpu()), 1).byte().numpy()
            path=os.path.join("./test_pred",args.net_G,args.root_dir[13:],str(args.order),model_name[-23:-4])
            if(os.path.exists(path)==False):
                os.makedirs(path)
            for filename, lp in zip(batch["name"], pred):
                np.save(os.path.join(path,filename[:-5]+".npy"),lp)
            gt = mask.data.cpu().numpy()
            running_metrics_val.update(gt, pred)
    score, mean_f1, _ = running_metrics_val.get_scores()
    for k, v in score.items():
        print(f"{k}{v}")
    return score, mean_f1

def test_all(model, args, source,flag):
    print(args)
    val_set = RemoteDataset(root_dir=args.root_dir, order=args.order, split=flag)#test train val
    valloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    model.eval()
    running_metrics_val = runningScore(args.classes)
    save_tse=[]
    save_tse_mid=[]
    with torch.no_grad():
        for batch in tqdm(valloader, desc=f"Valid", ncols=100):
            x1, x2, mask = (
                batch["P"].to(device),
                batch["S"].to(device),
                batch["L"].to(device),
            )
            if(args.net_G=="MTAF_TST"):
                x1 = x1.reshape(x1.shape[0],args.T_P,args.in_channels_P,args.patch_size,args.patch_size)
                x2 = x2.reshape(x2.shape[0],args.T_S,args.in_channels_S,args.patch_size,args.patch_size)
                conv_out = LayerActivations(model.ConRelu)
                pred_img_1 ,pred_optic,pred_sar= model(x1, x2)
                conv_out.remove()
                feature = conv_out.features
            elif(args.net_G=="Optic_TST" or args.net_G=="Sar_TST" or args.net_G=="Wo_Aux_TST" or args.net_G=="Wo_Aux_TST_MTAF"):
                x1 = x1.reshape(x1.shape[0],args.T_P,args.in_channels_P,args.patch_size,args.patch_size)
                x2 = x2.reshape(x2.shape[0],args.T_S,args.in_channels_S,args.patch_size,args.patch_size)
                pred_img_1 = model(x1,x2)
            pred = torch.argmax((pred_img_1.cpu()), 1).byte().numpy()
            gt = mask.data.cpu().numpy()
            running_metrics_val.update(gt, pred)
    score, mean_f1, _ = running_metrics_val.get_scores()
    for k, v in score.items():
        print(f"{k}{v}")
    return score, mean_f1

if __name__ == "__main__":
    args = parse_args()
    model = build_model(args).to(device)
    if args.do != "train":
        model.load_state_dict(torch.load(args.checkpoint))
        start=datetime.datetime.now()
        test_all(model, args,"double","test")
        end=datetime.datetime.now()
    else:
        train(model, args)
