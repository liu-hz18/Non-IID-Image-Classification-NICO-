import os
import time
import random
import argparse
from tqdm import tqdm
from itertools import chain
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score

from logger import Logger
from models import (
    MLPClassifier,
    AttentionMLP,
    SampleWeighting,
    GANN,
    DiscriminatorStack,
)
from dataset import make_datasets, NicoNaiveTestDataset
from metrics import Metrics, MetricsAverageMeter
from utils import onehot


def init_parser():
    parser = argparse.ArgumentParser("Non-IID Image Classification.")
    parser.add_argument("--ckptdir", type=str, default="../checkpoints",
                        help="saved checkpoint directory")
    parser.add_argument("--savedir", type=str, default="../",
                        help="saved test prediction result directory")
    parser.add_argument("--train_dataset", type=str, default="../nico/feature_train.npy",
                        help="train dataset directory path")
    parser.add_argument("--test_dataset", type=str, default="../nico/feature_test.npy",
                        help="test dataset directory path")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="split ratio of train dataset and eval dataset")
    parser.add_argument("--rank", type=int, default=0, help="GPU rank to train or eval")
    parser.add_argument("--seed", type=int, default=49705, help="manual seed")
    parser.add_argument("--bsz", type=int, default=256, help="train or eval batch size")
    parser.add_argument("--nepoch", type=int, default=80, help="number of epochs")
    parser.add_argument("--arch", type=str, default="mlp", choices=["mlp", "attn", "gann"], 
                        help="model architecture")
    parser.add_argument("--optimizer", type=str,
                        default="sgd", choices=["sgd", "adam"], help="oprimizer")
    parser.add_argument("--lr", type=float, default=0.1, help="learing rate for optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay for optimizer")
    parser.add_argument("--lr_decay", type=float, default=0.1,
                        help="decay constant when update step reach `lr_step`")
    parser.add_argument("--lr_step", type=int, default=300,
                        help="decay lr per `lr_step`")
    parser.add_argument("--eval_step", type=int, default=5,
                        help="eval step after train")
    parser.add_argument("--save_ckpt_step", type=int, default=5,
                        help="steps to save checkpoints.")
    parser.add_argument("--sample_weight", action="store_true", default=False, 
                        help="apply RFF sample weight to train.")
    parser.add_argument("--reweight_step", type=int, default=100,
                        help="step of sampling weight to train per epoch.")
    parser.add_argument("--sw_lr", type=float, default=0.001,
                        help="learning rate to train sample_weight.")
    parser.add_argument("--gann_d_lr", type=float, default=0.01,
                        help="learning rate to train GANN discriminator.")
    parser.add_argument("--gann_g_lr", type=float, default=0.01,
                        help="learning rate to train GANN generator.")
    parser.add_argument("--gann_gr_alpha", type=float, default=0.1,
                        help="lambda value of gradient reversal module in GANN.")
    parser.add_argument("--gann_loss_alpha", type=float, default=0.1,
                        help="weight on domain_loss in GANN.")
    parser.add_argument("--gann_bsz", type=int, default=64,
                        help="batch size to train GANN.")
    parser.add_argument("--gann_epoch", type=int, default=20,
                        help="epochs to train GANN.")
    return parser


def init_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'manual seed: {seed}')


def save_checkpoint(ckptfile, model, optimizer, scheduler, epoch):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, ckptfile)


def load_checkpoint(ckptfile, model, optimizer, scheduler, device) -> int:
    state = torch.load(ckptfile, map_location=device, strict=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["epoch"]


def train_step_sample_weighting(features):
    # logger.info(f"re-train sampling weight.")
    global global_sw_train_step
    train_sw_meter.reset()
    features = features.to(device)
    global_sw_train_step += 1
    sample_weighting.reset_parameters()
    sample_weighting.train()
    for _ in range(args.reweight_step):
        loss = sample_weighting(features)
        optimizerSW.zero_grad()
        loss.backward()
        optimizerSW.step()

        train_sw_meter.update({
            "loss": loss.item(),
        })
    # logger.info(f"SampleWeighting train step {global_sw_train_step} finish. \nTrain {train_sw_meter.mean()}")


def train_epoch():
    global global_train_step
    train_meter.reset()
    model.train()
    sample_weighting.eval()
    for step, (features, cls_labels, _) in tqdm(enumerate(image_dataloader["train"]), total=len(image_datasets["train"])//args.bsz):
        global_train_step += 1
        features = features.to(device)
        cls_labels = cls_labels.to(device)
        
        logits = model(features)
        loss = criterion_cls(logits, cls_labels)
        if args.sample_weight:
            train_step_sample_weighting(features)
            sample_weighting.eval()
            loss = sample_weighting.apply_weight(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        cls_preds = logits.argmax(dim=-1).cpu()
        acc = accuracy_score(cls_preds, cls_labels.cpu())
        f1 = f1_score(cls_preds, cls_labels.cpu(), average="macro")

        train_meter.update({
            "loss": loss.item(),
            "acc": acc,
            "f1": f1,
        })
        if global_train_step % args.lr_step == 0:
            logger.info(f"Learning rate decayed to {optimizer.param_groups[0]['lr']}")
    logger.info(f"train step {global_train_step} finish. \nTrain {train_meter.mean()}")
    if epoch % args.save_ckpt_step == 0:
        save_checkpoint(os.path.join(
            ckptdir, f"trainstep_{global_train_step}"), model, optimizer, scheduler, epoch
        )


def eval_epoch():
    global global_eval_step, best_epoch, best_acc
    eval_meter.reset()
    model.eval()
    with torch.no_grad():
        for step, (features, cls_labels, _) in tqdm(enumerate(image_dataloader["eval"]), total=len(image_datasets["eval"])//args.bsz):
            global_eval_step += 1
            features = features.to(device)
            cls_labels = cls_labels.to(device)

            logits = model(features)
            loss = criterion_cls(logits, cls_labels)

            cls_preds = logits.argmax(dim=-1).cpu()
            acc = accuracy_score(cls_preds, cls_labels.cpu())
            f1 = f1_score(cls_preds, cls_labels.cpu(), average="macro")

            eval_meter.update({
                "loss": loss.mean().item(),
                "acc": acc,
                "f1": f1,
            })
    avg_metrics = eval_meter.mean()
    if avg_metrics._metrics["acc"] > best_acc:
        best_acc = avg_metrics._metrics["acc"]
        best_epoch = epoch
    logger.info(f"eval step {global_eval_step} finish. \nEval {avg_metrics}, best_acc={best_acc}, best_epoch={best_epoch}")


def test_epoch():
    logger.info("run final test")
    cls_preds = []
    model.eval()
    with torch.no_grad():
        for step, features in tqdm(enumerate(image_dataloader["test"]), total=len(image_datasets["test"])//args.bsz):
            features = features.to(device)

            logits = model(features)
            cls_preds += logits.argmax(dim=-1).tolist()
    os.makedirs(args.savedir, exist_ok=True)
    with open(os.path.join(args.savedir, f"submit_{args.arch + '_' + method}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([str(x) for x in cls_preds]))


def train_gann_discriminator():
    train_gann_meterD.reset()
    model.eval()
    discriminator.train()
    cls_idxs = list(range(10))
    classwise_dataloader_iter = [
        iter(class_loader) for class_loader in classwise_dataloader
    ]
    while True:
        cls_idx = random.choices(cls_idxs)[0]
        try:
            features, cls_labels, ctx_labels = next(classwise_dataloader_iter[cls_idx])
        except StopIteration:
            cls_idxs.remove(cls_idx)
            if len(cls_idxs) == 0:
                break
            else:
                continue
        features = features.to(device)
        cls_labels = cls_labels.to(device)
        ctx_labels = ctx_labels.to(device)

        _, feat = model(features, return_feature=True)
        domain_logits = discriminator(feat, cls_labels[0])
        loss_ctx = criterion_ctx(domain_logits, ctx_labels)

        optimizerD.zero_grad()
        optimizerG.zero_grad()
        loss_ctx.backward()
        optimizerD.step()
        train_gann_meterD.update({"D_loss": loss_ctx.item()})
    logger.info(f"discriminator train step finish. \nTrain {train_gann_meterD.mean()}")


def train_gann_generator():
    global global_gann_train_step
    train_gann_meterG.reset()
    model.train()
    discriminator.eval()
    cls_idxs = list(range(10))
    classwise_dataloader_iter = [
        iter(class_loader) for class_loader in classwise_dataloader
    ]
    while True:
        cls_idx = random.choices(cls_idxs)[0]
        try:
            features, cls_labels, ctx_labels = next(classwise_dataloader_iter[cls_idx])
        except StopIteration:
            cls_idxs.remove(cls_idx)
            if len(cls_idxs) == 0:
                break
            else:
                continue
        global_gann_train_step += 1
        features = features.to(device)
        cls_labels = cls_labels.to(device)
        ctx_labels = ctx_labels.to(device)
        fake_labels = (1. - onehot(ctx_labels, 10)).long()

        logits, feat = model(features, return_feature=True)
        domain_logits = discriminator(feat, cls_labels[0])
        loss_cls = criterion_cls(logits, cls_labels).mean()
        loss_adv = criterion_adv(domain_logits, fake_labels)
        loss = loss_adv

        cls_preds = logits.argmax(dim=-1).cpu()
        acc = accuracy_score(cls_preds, cls_labels.cpu())
        f1 = f1_score(cls_preds, cls_labels.cpu(), average="macro")

        optimizerG.zero_grad()
        optimizerD.zero_grad()
        loss_adv.backward()
        optimizerG.step()
        train_gann_meterG.update({
            "loss": loss.item(),
            "G_loss": loss_adv.item(),
            "cls_loss": loss_cls.item(),
            "acc": acc,
            "f1": f1,
        })
    logger.info(f"generator train step finish. \nTrain {train_gann_meterG.mean()}")


def train_gann_epoch():
    global global_gann_train_step
    for _ in range(10):
        train_gann_discriminator()
    train_gann_generator()


if __name__ == "__main__":
    begin_time = time.strftime("-%H-%M-%S", time.localtime())
    parser = init_parser()
    args = parser.parse_args()

    # environment and utils config
    method = "naive" if not args.sample_weight else "reweight"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(args.rank)
    init_seed(args.seed)
    ckptdir = os.path.join(args.ckptdir, args.arch + "_" + method + begin_time)
    os.makedirs(ckptdir, exist_ok=True)
    logger = Logger(name=args.arch + "_" + method + "_logger", directory=ckptdir)
    logger.save_args(args)
    logger.info(f"args: {args}")


    # dataset config
    train_dataset, eval_dataset, classwise_dataset = make_datasets(args.train_dataset)
    test_dataset = NicoNaiveTestDataset(path=args.test_dataset)
    image_datasets = {
        "train": train_dataset,
        "eval": eval_dataset,
        "test": test_dataset,
    }
    image_dataloader = {
        "train": DataLoader(image_datasets["train"], batch_size=args.bsz, shuffle=True, num_workers=0),
        "eval": DataLoader(image_datasets["eval"], batch_size=args.bsz, shuffle=False, num_workers=0),
        "test": DataLoader(image_datasets["test"], batch_size=args.bsz, shuffle=False, num_workers=0),
    }
    classwise_dataloader = [
        DataLoader(class_set, batch_size=args.gann_bsz, shuffle=True, num_workers=0)
        for class_set in classwise_dataset
    ]
    logger.info(f"train dataset: {len(image_datasets['train'])} samples")
    logger.info(f"eval dataset: {len(image_datasets['eval'])} samples")
    logger.info(f"test dataset: {len(image_datasets['test'])} samples")
    for i, class_set in enumerate(classwise_dataset):
        logger.info(f"class-wise train dataset {i}: {len(class_set)} samples")


    # model config
    if args.arch == "mlp":
        model = MLPClassifier(512, 10).to(device)
    elif args.arch == "attn":
        model = AttentionMLP(512, 10).to(device)
    elif args.arch == "gann":
        model = GANN(512, 10, 10, alpha=0.1).to(device)
        discriminator = DiscriminatorStack(512, 10, 10).to(device)
    else:
        logger.warn(f"model architecture `{args.arch}` not supported. abort.")
        os.removedirs(ckptdir)
        exit(-1)
    logger.info(f"model arch: \n{model}")
    if args.arch == "gann":
        logger.info(f"discriminator arch: \n{discriminator}")
    
    # sample weighting policy
    if args.sample_weight:
        sample_weighting = SampleWeighting(args.bsz, alpha=0.2).to(device)
    else:
        sample_weighting = nn.Identity()

    # optim config
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    else:
        logger.warn(f"optimizer `{args.optimizer}` not supported. abort.")
        os.removedirs(ckptdir)
        exit(-1)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_decay
    )

    if args.sample_weight:
        optimizerSW = optim.SGD(
            filter(lambda p: p.requires_grad, sample_weighting.parameters()),
            lr=args.sw_lr, momentum=args.momentum,
        )
    if args.arch == "gann":
        optimizerG = optim.SGD(
            filter(lambda p: p.requires_grad, model.feature_extractor.parameters()),
            lr=args.gann_g_lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
        optimizerD = optim.SGD(
            filter(lambda p: p.requires_grad, discriminator.parameters()),
            lr=args.gann_d_lr, momentum=args.momentum, weight_decay=args.weight_decay
        )

    # metric config
    metrics = Metrics(["loss", "acc", "f1"])
    train_meter = MetricsAverageMeter(metrics)
    eval_meter = MetricsAverageMeter(metrics)
    metrics_sw = Metrics(["loss"])
    train_sw_meter = MetricsAverageMeter(metrics_sw)
    if args.arch == "gann":
        gann_metricsD = Metrics(["D_loss"])
        train_gann_meterD = MetricsAverageMeter(gann_metricsD)
        gann_metricsG = Metrics(["loss", "G_loss", "cls_loss", "acc", "f1"])
        train_gann_meterG = MetricsAverageMeter(gann_metricsG)

    # criterion config
    criterion_cls = nn.CrossEntropyLoss(reduction="none" if args.sample_weight else "mean")
    criterion_ctx = nn.CrossEntropyLoss()
    criterion_adv = nn.MultiLabelSoftMarginLoss()

    # training task
    best_acc = 0.0
    best_epoch = 0
    global_train_step, global_eval_step, global_sw_train_step, global_gann_train_step = 0, 0, 0, 0
    # stage 1
    for epoch in range(args.nepoch):
        logger.info(f"epoch {epoch}")
        train_epoch()
        if epoch % args.eval_step == 0:
            eval_epoch()
    
    if args.arch == "gann":
        logger.info(f"GANN init epoches finished.")
        # stage 2
        for epoch in range(args.gann_epoch):
            logger.info(f"epoch GANN {epoch}")
            train_gann_epoch()
            if epoch % args.eval_step == 0:
                eval_epoch()
        eval_epoch()
        # stage 3
        train_meter.reset()
        eval_meter.reset()
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.gann_g_lr
        for epoch in range(args.nepoch):
            logger.info(f"epoch {epoch}")
            train_epoch()
            if epoch % args.eval_step == 0:
                eval_epoch()

    eval_epoch()
    test_epoch()
