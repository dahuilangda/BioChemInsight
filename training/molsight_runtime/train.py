'''
torchrun --nproc_per_node 4 -m train --data_path data --coords_file aux_file --valid_file real/USPTO.csv --test_file real/CLEF.csv,real/UOB.csv,real/USPTO.csv,real/staker.csv,real/acs.csv,synthetic/indigo.csv,synthetic/chemdraw.csv --vocab_file vocab/vocab_chars.json --formats char,coords,edges --dynamic_indigo --augment --mol_augment --include_condensed --n_coord_bins 64 --sep_xy --input_size 1024 --encoder vitdet --decoder transformer --encoder_lr 4e-5 --decoder_lr 4e-4 --save_mode last --load_ckpt last --epochs 1 --batch_size 4 --accum_freq 16 --warmup 0.02 --print_freq 100 --do_train --do_valid --do_test --fp16 --backend nccl --num_workers 4 --continuous_coords --train_datasets molparser7m --smiles_only --use_qknorm --use_swiglu --use_rmsnorm --exp_name exp3
'''
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import sys
import time
import json
import random
import argparse
import datetime
import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import get_scheduler
from datasets import load_dataset

from molsight.checkpoint import model_configuration, load_parameters, model_state
from molsight.cpu import loader_options
from molsight.inference import predict_images
from molsight.dataset import HybridDataset, ValDataset, collate_fn
from molsight.model import MolsightModel
from molsight.loss import Criterion
from molsight.optimization import supervised_update
from molsight.utils import seed_torch, save_args, AverageMeter, ProgressMeter, asMinutes, timeSince, \
    format_df
from molsight.chemistry import convert_graph_to_smiles, postprocess_smiles, keep_main_molecule
from molsight.tokenizer import CharTokenizer, SOS_ID, EOS_ID, PAD_ID
from molsight.logging import ScalarWriter, get_logger
from molsight.distributed import init_distributed_device
from evaluate import SmilesEvaluator

import warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__)

data_path_map = {
    'pubchem': 'pubchem/train_1m.csv',
    'uspto': 'uspto_mol/train_680k.csv',
    'molparser7m': 'MolParser-7M/pretrain_synthetic_7M',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--backend', type=str, default='nccl', choices=['gloo', 'nccl'])
    # Model
    parser.add_argument('--encoder', type=str, default='efficientvit', choices=['efficientvit', 'vitdet'])
    parser.add_argument('--vitdet_weight', type=str, default='model.safetensors')
    parser.add_argument('--decoder', choices=['transformer'], default='transformer')
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--enc_pos_emb', action='store_true')
    parser.add_argument("--dec_n_layer", help="No. of layers in transformer decoder", type=int, default=6)
    parser.add_argument("--dec_n_head", help="Decoder no. of attention heads", type=int, default=8)
    parser.add_argument("--hidden_dropout", help="Hidden dropout", type=float, default=0.1)
    parser.add_argument("--attn_dropout", help="Attention dropout", type=float, default=0.1)
    parser.add_argument("--max_relative_positions", help="Max relative positions", type=int, default=0)
    parser.add_argument("--use_qknorm", action='store_true',)
    parser.add_argument("--use_swiglu", action='store_true',)
    parser.add_argument("--use_rmsnorm", action='store_true',)
    parser.add_argument('--lora', action='store_true', help='Use LoRA for the decoder.')
    # Data
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--train_datasets', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--aux_file', type=str, default=None)
    parser.add_argument('--coords_file', type=str, default=None)
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument('--source_vocab_file', type=str, default=None)
    parser.add_argument('--dynamic_indigo', action='store_true')
    parser.add_argument('--default_option', action='store_true')
    parser.add_argument('--pseudo_coords', action='store_true')
    parser.add_argument('--include_condensed', action='store_true')
    parser.add_argument('--formats', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None, help='Data workers per local rank; default shares available CPUs')
    parser.add_argument('--input_size', type=int, default=384)
    parser.add_argument('--multiscale', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--mol_augment', action='store_true')
    parser.add_argument('--n_coord_bins', type=int, default=100)
    parser.add_argument('--sep_xy', action='store_true')
    parser.add_argument('--mask_ratio', type=float, default=0)
    parser.add_argument('--continuous_coords', action='store_true')
    parser.add_argument('--max_len', type=int, default=320)
    # Training
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--decoder_lr', type=float, default=4e-4)
    parser.add_argument('--predictor_lr', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'constant'], default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--accum_freq', type=int, default=1)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_model_only', action='store_true')
    parser.add_argument('--train_steps_per_epoch', type=int, default=-1)
    parser.add_argument('--log_base_dir', type=str, default='runs/')
    parser.add_argument("--exp_name", default="debug", type=str)
    parser.add_argument('--save_mode', type=str, default='best', choices=['best', 'all', 'last'])
    parser.add_argument('--load_ckpt', type=str, default='best')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--all_data', action='store_true', help='Use both train and valid data for training.')
    parser.add_argument('--init_scheduler', action='store_true')
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--shuffle_nodes', action='store_true')
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--vis_path', type=str, default='vis/')
    parser.add_argument('--regression', type=bool, default=False)  # coords path reads model.py:500
    parser.add_argument('--smiles_only', action='store_true', help='Do not predict edges and coordinates, only SMILES.')
    parser.add_argument('--steps_per_save', type=int, default=50000, help='Steps per save checkpoint.')
    # Inference
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--save_attns', action='store_true')
    parser.add_argument('--molblock', action='store_true')
    parser.add_argument('--compute_confidence', action='store_true')
    parser.add_argument('--keep_main_molecule', action='store_true')
    args = parser.parse_args()
    return args


def train_one_epoch(dataloader, model, criterion, optimizer, epoch,
                    scheduler, scaler, device, writer, args):
    model.train()
    policy = getattr(model, "module", model)
    if not any(parameter.requires_grad for parameter in policy.image_encoder.parameters()):
        policy.image_encoder.eval()
    iterator = iter(dataloader)
    meters = {name: AverageMeter(name) for name in criterion.criterion}
    started = time.monotonic()
    successful_updates = 0
    for step in range(args.train_steps_per_epoch):
        batches = [next(iterator) for _ in range(args.accum_freq)]
        losses, grad_norm, stepped = supervised_update(
            model, criterion, batches, optimizer, scheduler, scaler,
            device, args.max_grad_norm, amp=args.amp,
        )
        successful_updates += int(stepped)
        for name, value in losses.items():
            meters[name].update(value)
        global_step = args.train_steps_per_epoch * epoch + step
        if (step + 1) % args.print_freq == 0:
            if args.rank == 0:
                for name, meter in meters.items():
                    writer.add_scalar(f"train/{name}_loss", meter.avg, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                writer.add_scalar("train/successful_updates", successful_updates, global_step)
                logger.info(
                    f"Epoch {epoch} update {step + 1}/{args.train_steps_per_epoch} "
                    f"losses={{{', '.join(f'{name}: {meter.avg:.5f}' for name, meter in meters.items())}}} "
                    f"lr={scheduler.get_last_lr()[0]:.6g} grad_norm={grad_norm:.4f} "
                    f"elapsed={time.monotonic() - started:.1f}s"
                )
            for meter in meters.values():
                meter.reset()
        if args.rank == 0 and (step + 1) % args.steps_per_save == 0:
            checkpoint = {
                "epoch": epoch,
                "epoch_complete": False,
                "step": step + 1,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f"epoch_{epoch}_step_{step + 1}.pth"))


def inference(dataloader, model, tokenizer, device, args, epoch=0):
    batch_time_m = AverageMeter("Time", ":6.3f")
    progress = ProgressMeter(
        len(dataloader),
        [
            batch_time_m,
        ],
        prefix="Validation: [{}]".format(epoch),
    )
    # switch to evaluation mode
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    
    predictions = {}
    start = end = time.time()
    for i, batch in enumerate(dataloader):
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        n_img = len(batch["idx"])
        #with torch.no_grad():
        #    results = model(**batch)
        #    pred = results['logits'].argmax(dim=-1)
        
        batch_preds = predict_images(model, batch["image"], formats=args.formats)

        for j in range(n_img):
            output = {}
            for key in batch_preds.keys():
                if batch_preds[key] is not None:
                    output[key] = batch_preds[key][j]
            predictions[int(batch["idx"][j])] = output
        # measure elapsed time
        batch_time_m.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 or (i + 1) == len(dataloader):
            if args.rank == 0:
                progress.display(i + 1)
                logger.info(f"Elapsed {timeSince(start, float(i + 1) / len(dataloader))} min")
            #break   # need to remove when not debugging !!!
    
    if args.world_size > 1:
        # gather predictions from different GPUs
        gathered_preds = [None for i in range(dist.get_world_size())]
        dist.all_gather_object(gathered_preds, predictions)
    else:
        gathered_preds = [predictions]
    predictions = [{}] * len(dataloader.dataset)
    for preds in gathered_preds:
        for idx, pred in preds.items():
            predictions[idx] = pred
    return predictions


def train(args, train_data, valid_df, model, optimizer, scheduler, tokenizer, device, writer, start_epoch):
    if args.rank == 0:
        logger.info("========== training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_dataset = HybridDataset(args, train_data, tokenizer)
    if args.debug and args.rank == 0:
        # vis some data
        import cv2
        for idx in range(20):
            sample = train_dataset[idx]
            if 'smiles' not in sample:
                continue
            smiles = sample['smiles']
            image = cv2.putText(sample['image'], smiles, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
            os.makedirs(args.vis_path, exist_ok=True)
            cv2.imwrite(os.path.join(args.vis_path, f'{idx}.png'), image)
    if args.world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              **loader_options(args.num_workers),
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_fn)

    scaler = torch.amp.GradScaler('cuda', enabled=False)
    if getattr(args, 'scaler_state', None):
        scaler.load_state_dict(args.scaler_state)

    # ====================================================
    # loop
    # ====================================================
    criterion = Criterion(args, tokenizer).to(device)

    best_score = -np.inf
    best_loss = np.inf

    for epoch in range(start_epoch, args.epochs):

        if args.world_size > 1:
            train_sampler.set_epoch(epoch)
            dist.barrier()

        start_time = time.time()

        # train
        train_one_epoch(
            train_loader, model, criterion, optimizer, epoch,
            scheduler, scaler, device, writer, args)

        # save
        if args.rank == 0:
            checkpoint_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch_complete': True,
            }
            torch.save(checkpoint_dict, os.path.join(args.save_dir, f'epoch_{epoch}.pth'))
        if args.world_size > 1:
            dist.barrier()
        
        # eval
        scores = valid(args, valid_df, tokenizer, model, device, epoch, split='valid')

        if args.rank != 0:
            continue

        elapsed = time.time() - start_time
        logger.info(f"Epoch {epoch} done, elapsed time: {asMinutes(elapsed)} min")

        if 'exact' not in scores:
            raise ValueError('Raw exact structure accuracy is required for model selection')
        score = scores['exact']

        logger.info(scores)
        logger.info(f'Epoch {epoch} - Valid Score: {score:.4f}')
        writer.add_scalar("valid/exact", score, epoch)
        best_score = max(best_score, score)

    if args.rank == 0:
        logger.info('Best valid score: {:.4f}'.format(best_score))
    if args.world_size > 1:
        dist.barrier()


def valid(args, data_df, tokenizer, model, device, epoch=0, split='test'):
    if args.rank == 0:
        logger.info("========== inference ==========")
        logger.info(data_df.attrs['file'])

    dataset = ValDataset(args, data_df, tokenizer)
    if args.world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=max(1, args.batch_size // 2),
                            sampler=sampler,
                            **loader_options(args.num_workers),
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=collate_fn)
    predictions = inference(dataloader, model, tokenizer, device, args, epoch)

    # The evaluation and saving prediction is only performed in the master process.
    if args.rank != 0:
        return
    logger.info('Start evaluation')

    # Deal with discrepancies between datasets
    if 'pubchem_cid' in data_df.columns:
        data_df['image_id'] = data_df['pubchem_cid']
    if 'image_id' not in data_df.columns:
        data_df['image_id'] = [path.split('/')[-1].split('.')[0] for path in data_df['file_path']]
    pred_df = data_df[['image_id']].copy()
    scores = {}

    # temporarily limit the number of predictions for debugging
    #data_df = data_df[:40]
    #pred_df = pred_df[:40]
    #predictions = predictions[:40]

    # SMILES
    pred_df['SMILES'] = [preds['smiles'] for preds in predictions]
    pred_df['node_symbols'] = [preds['atoms'] for preds in predictions]
    pred_df['indices'] = [preds['indices'] for preds in predictions]
    pred_df['node_coords'] = [preds['coords'] for preds in predictions] if 'coords' in args.formats else None
    pred_df['edges'] = [preds['edges'] for preds in predictions] if 'edges' in args.formats else None

    '''# Construct graph from predicted atoms and bonds (including verify chirality)
    smiles_list, molblock_list, r_success = convert_graph_to_smiles(
        pred_df['node_coords'], pred_df['node_symbols'], pred_df['edges'])

    print(f'Graph to SMILES success ratio: {r_success:.4f}')
    pred_df['graph_SMILES'] = smiles_list'''

    # Postprocess the predicted SMILES (verify chirality, expand functional groups)
    smiles_list, _, r_success = postprocess_smiles(
        pred_df['SMILES'], pred_df['node_coords'], pred_df['node_symbols'], pred_df['edges'])
    print(f'Postprocess SMILES success ratio: {r_success * 100:.2f}%')
    pred_df['post_SMILES'] = smiles_list

    # Keep the main molecule
    if args.keep_main_molecule:
        if 'graph_SMILES' in pred_df:
            pred_df['graph_SMILES'] = keep_main_molecule(pred_df['graph_SMILES'])
        if 'post_SMILES' in pred_df:
            pred_df['post_SMILES'] = keep_main_molecule(pred_df['post_SMILES'])

    # Compute scores
    if 'SMILES' in data_df.columns:
        evaluator = SmilesEvaluator(data_df['SMILES'], tanimoto=True)
        print('label:', data_df['SMILES'].values[:5])
        if 'SMILES' in pred_df.columns:
            print('pred:', pred_df['SMILES'].values[:5])
            scores.update(evaluator.evaluate(pred_df['SMILES']))
        if 'post_SMILES' in pred_df.columns:
            post_scores = evaluator.evaluate(pred_df['post_SMILES'])
            for key, value in post_scores.items():
                scores[f'post_{key}'] = value
        if 'graph_SMILES' in pred_df.columns:
            graph_scores = evaluator.evaluate(pred_df['graph_SMILES'])
            for key, value in graph_scores.items():
                scores[f'graph_{key}'] = value

    print('Save predictions...')
    file_name = data_df.attrs['file'].split('/')[-1]
    pred_df = format_df(pred_df)
    pred_df.to_csv(os.path.join(args.log_dir, f'prediction_{file_name}'), index=False)
    # Save scores
    if split == 'test':
        with open(os.path.join(args.log_dir, f'eval_scores_{os.path.splitext(file_name)[0]}_{args.load_ckpt}.json'), 'w') as f:
            json.dump(scores, f)

    return scores


def load_data(args):
    train_data, valid_df, test_df = dict(), None, None
    if args.do_train:
        train_dataset_names = args.train_datasets.split(',')
        for name in train_dataset_names:
            path = os.path.join(args.data_path, data_path_map[name])
            if path.endswith('.csv'):
                train_data[name] = pd.read_csv(path)
            elif os.path.isdir(path):
                files = os.listdir(path)
                assert files[0].endswith('.parquet')
                train_data[name] = load_dataset(path, num_proc=4)['train']
            else:
                raise ValueError(f'Invalid path: {path}')
        for name in train_data.keys():
            if args.rank == 0:
                logger.info(f'{name} train.shape: {train_data[name].shape}')
        #print_rank_0(f'train.shape: {train_df.shape}')
    if args.do_train or args.do_valid:
        valid_df = pd.read_csv(os.path.join(args.data_path, args.valid_file))
        valid_df.attrs['file'] = args.valid_file
        if args.rank == 0:
            logger.info(f'valid.shape: {valid_df.shape}')
    if args.do_test:
        test_files = args.test_file.split(',')
        test_df = [pd.read_csv(os.path.join(args.data_path, file)) for file in test_files]
        for file, df in zip(test_files, test_df):
            df.attrs['file'] = file
            if args.rank == 0:
                logger.info(file + f' test.shape: {df.shape}')
    tokenizer = CharTokenizer(args.vocab_file)
    return train_data, valid_df, test_df, tokenizer


def main():
    args = parse_args()
    seed_torch(seed=args.seed)

    device = init_distributed_device(args)
    logger.info(
        f'Running in distributed mode with multiple processes. Device: {args.device}.'
        f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')

    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    args.save_dir = os.path.join(args.log_dir, "ckpt_model")
    if args.rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
        save_args(args)
        writer = ScalarWriter(args.log_dir)
    else:
        writer = None
    
    args.formats = args.formats.split(',')
    if args.rank == 0:
        logger.info(f"Output formats: {args.formats}")

    train_data, valid_df, test_df, tokenizer = load_data(args)

    checkpoint = None
    state = None
    if args.load_path:
        checkpoint = torch.load(args.load_path, map_location='cpu', weights_only=False, mmap=True)
        state = model_state(checkpoint)
        for key, value in model_configuration(state).items():
            setattr(args, key, value)
    if args.rank == 0:
        save_args(args)
    model = MolsightModel(args, tokenizer)
    model.to(device)

    encoder_params = list(model.image_encoder.parameters())
    decoder_params = []
    predictor_params = []
    for name, param in model.decoder.named_parameters():
        if "predictor" not in name and param.requires_grad:
            decoder_params.append(param)
        elif "predictor" in name and param.requires_grad:
            predictor_params.append(param)

    # set different learning rates for encoder and decoder
    optimizer_grouped_parameters = [
        #{"params": encoder_params, "lr": args.encoder_lr},
        {"params": decoder_params, "lr": args.decoder_lr},
        {"params": predictor_params, "lr": args.predictor_lr},  # edge and coord predictor
    ]
    if args.encoder_lr > 0:
        optimizer_grouped_parameters.append({"params": encoder_params, "lr": args.encoder_lr})
    else:
        if args.rank == 0:
            logger.warning("Encoder learning rate is set to 0, only decoder will be trained.")
        model.image_encoder.requires_grad_(False)

    if args.train_steps_per_epoch == -1:
        args.train_steps_per_epoch = sum([len(df) for df in train_data.values()]) // (args.batch_size * args.world_size) // args.accum_freq
    num_training_steps = args.epochs * args.train_steps_per_epoch
    args.num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        weight_decay=args.weight_decay,
        amsgrad=False
    )

    scheduler = get_scheduler(
        args.scheduler,
        optimizer,
        args.num_warmup_steps,
        num_training_steps
    )
    
    if args.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        if args.rank == 0:
            logger.info("DDP setup finished")
    
    # load 
    if args.resume:
        assert args.load_path is not None

    num_completed_epoch = 0
    if args.load_path:
        # load checkpoint
        ckpt_path = args.load_path
        if args.rank == 0:
            logger.info("=> loading checkpoint from {}".format(ckpt_path))
        source_vocabulary = None
        if args.source_vocab_file:
            with open(args.source_vocab_file) as stream:
                source_vocabulary = json.load(stream)
        load_parameters(getattr(model, "module", model), state,
                        source_vocabulary=source_vocabulary, target_vocabulary=tokenizer.stoi,
                        initialize_adapters=args.lora and args.load_model_only)
        if not args.load_model_only:
            if checkpoint.get("epoch_complete") is False:
                raise ValueError("Mid-epoch checkpoints require explicit data-position restoration")
            args.scaler_state = checkpoint.get("scaler")
            num_completed_epoch = checkpoint.get("epoch", 0) + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint['scheduler'])
    
    if args.do_train:
        train(args, train_data, valid_df, model, optimizer, scheduler, tokenizer, device, writer, num_completed_epoch)

    if args.do_valid:
        scores = valid(args, valid_df, tokenizer, model, device, num_completed_epoch, split='valid')
        if args.rank == 0:
            logger.info(json.dumps(scores, indent=4))

    if args.do_test:
        assert type(test_df) is list
        for df in test_df:
            scores = valid(args, df, tokenizer, model, device, num_completed_epoch, split='test')
            if args.rank == 0:
                logger.info(json.dumps(scores, indent=4))


if __name__ == "__main__":
    main()
