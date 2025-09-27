import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
from transformers import BertConfig
import random
import logging
from tqdm import trange
import argparse
import utils
from optimization import BertAdam
from evaluate import evaluate
from dataloader import CustomDataLoader
from model import BertForRE, PGD
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--ex_index', type=str, default=1)
parser.add_argument('--corpus_type', type=str, default="CA", help="NYT, WebNLG, NYT*, WebNLG*")
parser.add_argument('--device_id', type=int, default=0, help="GPU index")
parser.add_argument('--epoch_num', required=True, type=int, help="number of epochs")
parser.add_argument('--multi_gpu', action='store_true', help="ensure multi-gpu training")
parser.add_argument('--restore_file', default=None, help="name of the file containing weights to reload")
parser.add_argument('--corres_threshold', type=float, default=0.5, help="threshold of global correspondence")
parser.add_argument('--rel_threshold', type=float, default=0.5, help="threshold of relation judgement")
parser.add_argument('--ensure_corres', action='store_true', help="correspondence ablation")
parser.add_argument('--ensure_rel', action='store_true', help="relation judgement ablation")
parser.add_argument('--emb_fusion', type=str, default="concat", help="way to embedding")
parser.add_argument('--num_negs', type=int, default=4,
                    help="number of negative sample when ablate relation judgement")
def train(model, data_iterator, optimizer, params, ex_params, args):
    model.train()
    loss_avg = utils.RunningAverage()
    loss_avg_seq = utils.RunningAverage()
    loss_avg_mat = utils.RunningAverage()
    loss_avg_rel = utils.RunningAverage()
    loss_avg_adv = utils.RunningAverage()
    pgd = PGD(model)
    K = 3
    t = trange(len(data_iterator), ascii=True)
    for step, _ in enumerate(t):
        batch = next(iter(data_iterator))
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, attention_mask, seq_tags, relations, corres_tags, rel_tags = batch
        loss, loss_seq, loss_mat, loss_rel = model(input_ids, attention_mask=attention_mask,
                                                   seq_tags=seq_tags, potential_rels=relations,
                                                   corres_tags=corres_tags, rel_tags=rel_tags,
                                                   ex_params=ex_params)
        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()
        if params.gradient_accumulation_steps > 1:
            loss = loss / params.gradient_accumulation_steps
        loss.backward()
        pgd.attack(is_first_attack=True)
        for _k in range(K):
            adv_loss, _, _, _ = model(input_ids, attention_mask=attention_mask,
                                     seq_tags=seq_tags, potential_rels=relations,
                                     corres_tags=corres_tags, rel_tags=rel_tags,
                                     ex_params=ex_params, is_adv_training=True)
            if params.n_gpu > 1 and args.multi_gpu:
                adv_loss = adv_loss.mean()
            if params.gradient_accumulation_steps > 1:
                adv_loss = adv_loss / params.gradient_accumulation_steps
            adv_loss.backward()
            if _k != K - 1:
                pgd.attack()
            loss_avg_adv.update(adv_loss.item() * params.gradient_accumulation_steps)
        pgd.restore()
        if (step + 1) % params.gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
        loss_avg.update(loss.item() * params.gradient_accumulation_steps)
        loss_avg_seq.update(loss_seq.item())
        loss_avg_mat.update(loss_mat.item())
        loss_avg_rel.update(loss_rel.item())
        t.set_postfix(loss='{:05.5f}'.format(loss_avg()),
                      loss_seq='{:05.5f}'.format(loss_avg_seq()),
                      loss_mat='{:05.5f}'.format(loss_avg_mat()),
                      loss_rel='{:05.5f}'.format(loss_avg_rel()),
                      loss_adv='{:05.5f}'.format(loss_avg_adv()))
def train_and_evaluate(model, params, ex_params, restore_file=None):
    dataloader = CustomDataLoader(params)
    train_loader = dataloader.get_dataloader(data_sign='train', ex_params=ex_params)
    val_loader = dataloader.get_dataloader(data_sign='val', ex_params=ex_params)
    params.max_epoch = args.epoch_num
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        model, optimizer = utils.load_checkpoint(restore_path)
    model.to(params.device)
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)
    param_optimizer = list(model.named_parameters())
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
    param_downstream = [(n, p) for n, p in param_optimizer if 'bert' not in n]
    no_decay = ['bias', 'LayerNorm', 'layer_norm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.fin_tuning_lr
         },
        {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.fin_tuning_lr
         },
        {'params': [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.downs_en_lr
         },
        {'params': [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.downs_en_lr
         }
    ]
    num_train_optimization_steps = len(train_loader) // params.gradient_accumulation_steps * args.epoch_num
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=params.warmup_prop, schedule="warmup_cosine",
                         t_total=num_train_optimization_steps, max_grad_norm=params.clip_grad)
    best_val_f1 = 0.0
    patience_counter = 0
    for epoch in range(1, args.epoch_num + 1):
        params.current_epoch = epoch
        logging.info("Epoch {}/{}".format(epoch, args.epoch_num))
        train(model, train_loader, optimizer, params, ex_params, args)
        val_metrics, _, _ = evaluate(model, val_loader, params, ex_params, mark='Val')
        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1
        model_to_save = model.module if hasattr(model, 'module') else model
        optimizer_to_save = optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                               'model': model_to_save,
                               'optim': optimizer_to_save},
                              is_best=improve_f1 > 0,
                              checkpoint=params.model_dir)
        params.save(params.ex_dir / 'params.json')
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        if (patience_counter > params.patience_num and epoch > params.min_epoch_num) or epoch == args.epoch_num:
            logging.info("Best val f1: {:05.5f}".format(best_val_f1))
            break
if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(args.ex_index, args.corpus_type)
    ex_params = {
        'ensure_corres': args.ensure_corres,
        'ensure_rel': args.ensure_rel,
        'num_negs': args.num_negs,
        'emb_fusion': args.emb_fusion
    }
    if args.multi_gpu:
        params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        params.n_gpu = n_gpu
    else:
        torch.cuda.set_device(args.device_id)
        print('current device:', torch.cuda.current_device())
        params.n_gpu = n_gpu = 1
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    utils.set_logger(save=True, log_dir=params.ex_dir)
    logging.info(f"Model type:")
    logging.info("device: {}".format(params.device))
    logging.info('Load pre-train model weights...')
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'config.json'))
    model = BertForRE.from_pretrained(config=bert_config,
                                      pretrained_model_name_or_path=params.bert_model_dir,
                                      params=params)
    logging.info('-done')
    logging.info("Starting training for {} epoch(s)".format(args.epoch_num))
    train_and_evaluate(model, params, ex_params, args.restore_file)