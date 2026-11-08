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
from torch.utils.tensorboard import SummaryWriter

from molsight.dataset import HybridDataset, ValDataset, collate_fn
from molsight.model import MolsightModel, get_edge_prediction, enable_lora
from molsight.loss import Criterion, SCSTLoss, GRPOLoss
from molsight.reward import *
from molsight.utils import seed_torch, save_args, AverageMeter, ProgressMeter, asMinutes, timeSince, \
    format_df, adapt_state_dict
from molsight.chemistry import _postprocess_smiles, postprocess_smiles, keep_main_molecule
from molsight.tokenizer import CharTokenizer, SOS_ID, EOS_ID, PAD_ID
from molsight.logging import get_logger
from molsight.distributed import init_distributed_device
from evaluate import SmilesEvaluator
#torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__)

data_path_map = {
    'pubchem': 'pubchem/train_1m.csv',
    'uspto': 'uspto_mol/train_680k.csv',
    'molparser7m': 'MolParser-7M/pretrain_synthetic_7M',
    'stereo': 'stereo/train.csv',
    'staker': 'real/staker.csv',
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
    parser.add_argument('--decoder', type=str, default='lstm')
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
    parser.add_argument('--dynamic_indigo', action='store_true')
    parser.add_argument('--default_option', action='store_true')
    parser.add_argument('--pseudo_coords', action='store_true')
    parser.add_argument('--include_condensed', action='store_true')
    parser.add_argument('--formats', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
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
    parser.add_argument('--smiles_only', action='store_true', help='Do not predict edges and coordinates, only SMILES.')
    parser.add_argument('--steps_per_save', type=int, default=50000, help='Steps per save checkpoint.')
    # Inference
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--save_attns', action='store_true')
    parser.add_argument('--molblock', action='store_true')
    parser.add_argument('--compute_confidence', action='store_true')
    parser.add_argument('--keep_main_molecule', action='store_true')
    args = parser.parse_args()
    return args


def train_one_epoch(dataloader, model, criterion, reward_funcs, optimizer, epoch,
             scheduler, scaler, device, writer, args):
    batch_time_m = AverageMeter("Time", ":6.3f")
    data_time_m = AverageMeter("Data", ":6.3f")
    rl_loss_m = AverageMeter("RLLoss", ":.4f")
    reward_m = AverageMeter("Reward", ":.4f")
    reward_std_m = AverageMeter("RewardStd", ":.4f")
    kl_m = AverageMeter("KL", ":.4f")
    progress = ProgressMeter(
        args.train_steps_per_epoch,
        [
            #batch_time_m,
            rl_loss_m,
            reward_m,
            reward_std_m,
            kl_m,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )
    
    model.eval()    # freeze BN
    
    grad_norm = 0
    optimizer.zero_grad()
    train_iterator = iter(dataloader)
    start = end = time.time()
    
    for i in range(args.train_steps_per_epoch * args.accum_freq): # i 是微批次索引
        batch = next(train_iterator)
        batch = {
            k: v.to(args.local_rank, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        data_time_m.update(time.time() - end)

        i_accum = i // args.accum_freq
        global_step = args.train_steps_per_epoch * epoch + i_accum

        if i % args.accum_freq == 0:
            scheduler.step(global_step)
            optimizer.zero_grad()

        batch_size = batch['image'].size(0)
        n_valid_edge_ann = batch['edges'][:, 0, 0].ge(0).sum().item()
        n_valid_coord_ann = batch['coords'][:, 0, 0].ge(0).sum().item()

        def reward_fn(pred_smiles):
            if len(pred_smiles) % len(batch['smiles']):
                raise ValueError('Incomplete completion groups')
            group_size = len(pred_smiles) // len(batch['smiles'])
            gt_smiles = [gt for gt in batch['smiles'] for _ in range(group_size)]
            rewards_per_func = torch.zeros(len(pred_smiles), len(reward_funcs), device=device)
            for j, func in enumerate(reward_funcs):
                rewards_per_func[:, j] = func(pred_smiles, gt_smiles).to(device)
            return rewards_per_func.sum(dim=1).to(device)
        
        with torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.bfloat16):
            #results = model(**batch)
            loss, metric = criterion(model, batch['image'], reward_fn)
            #loss = sum(losses.values())
    
        rl_loss_m.update(loss.item(), batch_size)
        reward_m.update(metric['avg_rewards'], batch_size)
        reward_std_m.update(metric['rewards_std'], batch_size)
        kl_m.update(metric['kl'], batch_size)
        
        if args.accum_freq > 1:
            loss = loss / args.accum_freq
        scaler.scale(loss).backward()
        #loss.backward()
        
        #for name, param in model.named_parameters():
        #    if param.grad is None:
        #        print(f"[unused_parameters:] {name}")
        
        #for name, param in model.named_parameters():
        #    if torch.isnan(param).any():
        #        print(f"[PARAM NaN] {name}")
        #    if param.grad is not None and torch.isnan(param.grad).any():
        #        print(f"[GRAD NaN] {name}")

        if (i + 1) % args.accum_freq == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            #optimizer.step()

            if (i_accum + 1) % args.print_freq == 0:
                if args.world_size > 1:
                    rl_loss_m.all_reduce()
                    reward_m.all_reduce()
                    reward_std_m.all_reduce()
                    kl_m.all_reduce()
                
                if args.local_rank == 0:
                    progress.display(i_accum + 1) 
                    writer.add_scalar("train/total_loss", rl_loss_m.avg, global_step)
                    writer.add_scalar("train/reward", reward_m.avg, global_step)
                    writer.add_scalar("train/reward_std", reward_std_m.avg, global_step)
                    writer.add_scalar("train/kl", kl_m.avg, global_step)
                    writer.add_scalar("train/total_secs_per_batch", batch_time_m.avg, global_step)
                    writer.add_scalar("train/data_secs_per_batch", data_time_m.avg, global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    logger.info(f"lr:{scheduler.get_last_lr()[0]} Runed {timeSince(start, float(i_accum + 1) / args.train_steps_per_epoch)}")
                
                batch_time_m.reset()
                data_time_m.reset()
                rl_loss_m.reset()
                reward_m.reset()
                reward_std_m.reset()
                kl_m.reset()
            
            if args.local_rank == 0 and (i_accum + 1) % args.steps_per_save == 0:
                checkpoint_dict = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                torch.save(checkpoint_dict, os.path.join(args.save_dir, f'epoch_{epoch}_step_{i_accum + 1}.pth'))
            
        # measure elapsed time
        batch_time_m.update(time.time() - end)
        end = time.time()


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
    enable_lora(model)
    
    predictions = {}
    start = end = time.time()
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        n_img = len(batch["idx"])
        #with torch.no_grad():
        #    results = model(**batch)
        #    pred = results['logits'].argmax(dim=-1)
        
        with torch.no_grad():
            # install kv_cache hooks
            kv_cache, hooks = model.install_kv_cache_hooks()

            batch_preds, inter = model.generate(**batch, kv_cache=kv_cache)
            xi = inter['image_features']    # [n_img, H*W, hidden_dim]
            hidden_states = inter['hidden_states']  # [n_img, max_len, hidden_dim]
            
            # remove hooks
            for hook in hooks:
                hook.remove()

            edge_preds = edge_scores = loc_preds = None
            if 'edges' in args.formats:
                atom_indices = [torch.LongTensor(i) for i in batch_preds["indices"]]
                atom_indices = torch.nn.utils.rnn.pad_sequence(atom_indices, batch_first=True, padding_value=PAD_ID).to(hidden_states.device)   # [n_img, max_len]
                atom_indices = atom_indices + model.sample_begin  # convert to the indices in the hidden states
                
                edge_logits = model.decoder.edge_predictor(hidden_states, atom_indices)    # [b, l, l, 7]
                edge_probs = F.softmax(edge_logits, dim=-1)
                valid_lengths = [len(ind) for ind in batch_preds["indices"]]
                edge_preds, edge_scores = get_edge_prediction(edge_probs, valid_lengths)
            if 'coords' in args.formats:
                seq_len = hidden_states.size(1)
                position_embeddings = (model.decoder.embed_cos[:, :seq_len].to(xi.dtype), model.decoder.embed_sin[:, :seq_len].to(xi.dtype)) if args.use_qknorm else None
                for block in model.decoder.loc_predictor.loc_blocks:
                    hidden_states = block(hidden_states, xi, position_embeddings=position_embeddings)
                loc_coords = model.decoder.loc_predictor(hidden_states, atom_indices)[0]  # [b, l, 2]

                loc_preds = [loc_coords[i, :int(valid_lengths[i])].tolist() for i in range(n_img)]

            batch_preds['edges'] = edge_preds
            batch_preds['edge_scores'] = edge_scores
            batch_preds['coords'] = loc_preds
        
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
            if args.local_rank == 0:
                progress.display(i + 1)
                logger.info(f"Runed {timeSince(start, float(i + 1) / len(dataloader))} min")
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
    if args.local_rank == 0:
        logger.info("========== training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_dataset = HybridDataset(args, train_data, tokenizer)
    if args.debug and args.local_rank == 0:
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
                              num_workers=args.num_workers,
                              prefetch_factor=4,
                              persistent_workers=True,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_fn)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ====================================================
    # loop
    # ====================================================
    #criterion = Criterion(args, tokenizer).to(device)
    criterion = GRPOLoss(args).to(device)
    reward_funcs = [tanimoto_reward_fn, stereochem_reward_fn]

    best_score = -np.inf
    best_loss = np.inf

    for epoch in range(start_epoch, args.epochs):

        if args.world_size > 1:
            train_sampler.set_epoch(epoch)
            dist.barrier()

        start_time = time.time()

        # train
        train_one_epoch(
            train_loader, model, criterion, reward_funcs, optimizer, epoch,
            scheduler, scaler, device, writer, args)

        # save
        if args.local_rank == 0:
            checkpoint_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint_dict, os.path.join(args.save_dir, f'epoch_{epoch}.pth'))
        if args.world_size > 1:
            dist.barrier()
        
        # eval
        scores = valid(args, valid_df, tokenizer, model, device, epoch, split='valid')

        if args.local_rank != 0:
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

    if args.local_rank == 0:
        logger.info('Best valid score: {:.4f}'.format(best_score))
    if args.world_size > 1:
        dist.barrier()


def valid(args, data_df, tokenizer, model, device, epoch=0, split='test'):
    if args.local_rank == 0:
        logger.info("========== inference ==========")
        logger.info(data_df.attrs['file'])

    dataset = ValDataset(args, data_df, tokenizer)
    if args.world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size // 2,
                            sampler=sampler,
                            num_workers=args.num_workers,
                            prefetch_factor=4,
                            persistent_workers=True,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=collate_fn)
    predictions = inference(dataloader, model, tokenizer, device, args, epoch)

    # The evaluation and saving prediction is only performed in the master process.
    if args.local_rank != 0:
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
            if args.local_rank == 0:
                logger.info(f'{name} train.shape: {train_data[name].shape}')
        #print_rank_0(f'train.shape: {train_df.shape}')
    if args.do_train or args.do_valid:
        valid_df = pd.read_csv(os.path.join(args.data_path, args.valid_file))
        valid_df.attrs['file'] = args.valid_file
        if args.local_rank == 0:
            logger.info(f'valid.shape: {valid_df.shape}')
    if args.do_test:
        test_files = args.test_file.split(',')
        test_df = [pd.read_csv(os.path.join(args.data_path, file)) for file in test_files]
        for file, df in zip(test_files, test_df):
            df.attrs['file'] = file
            if args.local_rank == 0:
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
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
        save_args(args)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None
    
    args.formats = args.formats.split(',')
    if args.local_rank == 0:
        logger.info(f"Output formats: {args.formats}")

    train_data, valid_df, test_df, tokenizer = load_data(args)

    model = MolsightModel(args, tokenizer)
    model.to(device)

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    if args.local_rank == 0:
        logger.info("Trainabel parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"{name}: {param.shape}")

    if args.train_steps_per_epoch == -1:
        args.train_steps_per_epoch = sum([len(df) for df in train_data.values()]) // (args.batch_size * args.world_size) // args.accum_freq
    num_training_steps = args.epochs * args.train_steps_per_epoch
    args.num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.decoder_lr,
        #optimizer_grouped_parameters,
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
        if args.local_rank == 0:
            logger.info("DDP setup finished")
    
    # load 
    if args.resume:
        assert args.load_path is not None

    num_completed_epoch = 0
    if args.load_path:
        # load checkpoint
        ckpt_path = args.load_path
        if args.local_rank == 0:
            logger.info("=> loading checkpoint from {}".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model_checkpoint = checkpoint["model"]
        if args.world_size <= 1:
            # For DDP, we need to load the model state dict from the module
            model_checkpoint = {
                (k[7:] if k.startswith('module.') else k): v
                for k, v in model_checkpoint.items()
            }
        
        new_state_dict = adapt_state_dict(model, model_checkpoint, verbose=args.local_rank == 0)
        
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if args.local_rank == 0:
            for missing_key in missing_keys:
                assert 'lora_A' in missing_key or 'lora_B' in missing_key
            logger.info(f'Missing keys: {missing_keys}')
            logger.info(f'Unexpected keys: {unexpected_keys}')
        if not args.load_model_only:
            num_completed_epoch = checkpoint.get("epoch", 0) + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint['scheduler'])
    
    if args.do_train:
        train(args, train_data, valid_df, model, optimizer, scheduler, tokenizer, device, writer, num_completed_epoch)

    if args.do_valid:
        scores = valid(args, valid_df, tokenizer, model, device, num_completed_epoch, split='valid')
        if args.local_rank == 0:
            logger.info(json.dumps(scores, indent=4))

    if args.do_test:
        assert type(test_df) is list
        for df in test_df:
            scores = valid(args, df, tokenizer, model, device, num_completed_epoch, split='test')
            if args.local_rank == 0:
                logger.info(json.dumps(scores, indent=4))


if __name__ == "__main__":
    main()
