import json
import logging
import random
import argparse
from tqdm import tqdm
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from metrics import tag_mapping_nearest, tag_mapping_corres
from utils import Label2IdxSub, Label2IdxObj, SUB_TAG_SIZE, OBJ_TAG_SIZE
import utils
from dataloader import CustomDataLoader
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--ex_index', type=str, default=1)
parser.add_argument('--corpus_type', type=str, default="CA", help="NYT, WebNLG, NYT*, WebNLG*")
parser.add_argument('--device_id', type=int, default=0, help="GPU index")
parser.add_argument('--restore_file', default='last', help="name of the file containing weights to reload")
parser.add_argument('--mode', default='test', help="mode")
parser.add_argument('--mat_threshold', type=float, default=0.5, help="threshold of global correspondence")
parser.add_argument('--rel_threshold', type=float, default=0.5, help="threshold of relation judgement")
parser.add_argument('--ensure_corres', action='store_true', help="correspondence ablation")
parser.add_argument('--ensure_rel', action='store_true', help="relation judgement ablation")
parser.add_argument('--emb_fusion', type=str, default="concat", help="way to embedding")
def plot_model_confusion_matrix(predictions, ground_truths, save_path):
    all_triples = set()
    for pred_list in predictions:
        all_triples.update(pred_list)
    for truth_list in ground_truths:
        all_triples.update(truth_list)
    triple_labels = sorted(list(all_triples))
    y_true = []
    y_pred = []
    for pred_list, truth_list in zip(predictions, ground_truths):
        pred_set = set(pred_list)
        truth_set = set(truth_list)
        for triple in all_triples:
            y_true.append(1 if triple in truth_set else 0)
            y_pred.append(1 if triple in pred_set else 0)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()
def save_training_results(metrics, predictions, ground_truths, params, mark='Val'):
    if not hasattr(params, 'experiment_id'):
        params.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(params.ex_dir, f'training_results_{params.experiment_id}.json')
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {'epochs': []}
    epoch_result = {
        'epoch': params.current_epoch,
        'metrics': metrics,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    results['epochs'].append(epoch_result)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    if params.current_epoch == params.max_epoch:
        cm_path = os.path.join(params.ex_dir, f'confusion_matrix_{params.experiment_id}.png')
        plot_model_confusion_matrix(predictions, ground_truths, cm_path)
def get_metrics(correct_num, predict_num, gold_num):
    p = correct_num / predict_num if predict_num > 0 else 0
    r = correct_num / gold_num if gold_num > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1
    }
def span2str(triples, tokens):
    def _concat(token_list):
        result = ''
        for idx, t in enumerate(token_list):
            if idx == 0:
                result = t
            elif t.startswith('
                result += t.lstrip('
            else:
                result += ' ' + t
        return result
    output = []
    for triple in triples:
        rel = triple[-1]
        sub_tokens = tokens[triple[0][1]:triple[0][-1]]
        obj_tokens = tokens[triple[1][1]:triple[1][-1]]
        sub = _concat(sub_tokens)
        obj = _concat(obj_tokens)
        output.append((sub, obj, rel))
    return output
def evaluate(model, data_iterator, params, ex_params, mark='Val'):
    model.eval()
    rel_num = params.rel_num
    predictions = []
    ground_truths = []
    correct_num, predict_num, gold_num = 0, 0, 0
    all_true_entities = []
    all_pred_entities = []
    all_true_relations = []
    all_pred_relations = []
    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        batch = tuple(t.to(params.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, attention_mask, triples, input_tokens = batch
        bs, seq_len = input_ids.size()
        with torch.no_grad():
            pred_seqs, pre_corres, xi, pred_rels = model(input_ids, attention_mask=attention_mask,
                                                         ex_params=ex_params)
            pred_seqs = pred_seqs.detach().cpu().numpy()
            pre_corres = pre_corres.detach().cpu().numpy()
        if ex_params['ensure_rel']:
            xi = np.array(xi)
            pred_rels = pred_rels.detach().cpu().numpy()
            xi_index = np.cumsum(xi).tolist()
            xi_index.insert(0, 0)
        for idx in range(bs):
            if ex_params['ensure_rel']:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[xi_index[idx]:xi_index[idx + 1]],
                                                 pre_corres=pre_corres[idx],
                                                 pre_rels=pred_rels[xi_index[idx]:xi_index[idx + 1]],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)
            else:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[idx * rel_num:(idx + 1) * rel_num],
                                                 pre_corres=pre_corres[idx],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)
            gold_triples = span2str(triples[idx], input_tokens[idx])
            pre_triples = span2str(pre_triples, input_tokens[idx])
            ground_truths.append(list(set(gold_triples)))
            predictions.append(list(set(pre_triples)))
            correct_num += len(set(pre_triples) & set(gold_triples))
            predict_num += len(set(pre_triples))
            gold_num += len(set(gold_triples))
            true_entities = triples[idx]
            pred_entities = pred_seqs[idx * rel_num:(idx + 1) * rel_num]
            all_true_entities.extend(true_entities)
            all_pred_entities.extend(pred_entities)
            if ex_params['ensure_rel']:
                true_relations = pred_rels[xi_index[idx]:xi_index[idx + 1]]
                pred_relations = pred_rels[xi_index[idx]:xi_index[idx + 1]]
                all_true_relations.extend(true_relations)
                all_pred_relations.extend(pred_relations)
    metrics = get_metrics(correct_num, predict_num, gold_num)
    metrics_str = "; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics:\n".format(mark) + metrics_str)
    save_training_results(metrics, predictions, ground_truths, params, mark=mark)
    return metrics, predictions, ground_truths
if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(ex_index=args.ex_index, corpus_type=args.corpus_type)
    ex_params = {
        'corres_threshold': args.mat_threshold,
        'rel_threshold': args.rel_threshold,
        'ensure_corres': args.ensure_corres,
        'ensure_rel': args.ensure_rel,
        'emb_fusion': args.emb_fusion
    }
    torch.cuda.set_device(args.device_id)
    print('current device:', torch.cuda.current_device())
    mode = args.mode
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed
    utils.set_logger()
    dataloader = CustomDataLoader(params)
    logging.info('Loading the model...')
    logging.info(f'Path: {os.path.join(params.model_dir, args.restore_file)}.pth.tar')
    model, optimizer = utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'))
    model.to(params.device)
    logging.info('- done.')
    logging.info("Loading the dataset...")
    loader = dataloader.get_dataloader(data_sign=mode, ex_params=ex_params)
    logging.info('-done')
    logging.info("Starting prediction...")
    _, predictions, ground_truths = evaluate(model, loader, params, ex_params, mark=mode)
    with open(params.data_dir / f'{mode}_triples.json', 'r', encoding='utf-8') as f_src:
        src = json.load(f_src)
        df = pd.DataFrame(
            {
                'text': [sample['text'] for sample in src],
                'pre': predictions,
                'truth': ground_truths
            }
        )
        df.to_csv(params.ex_dir / f'{mode}_result.csv')
    logging.info('-done')