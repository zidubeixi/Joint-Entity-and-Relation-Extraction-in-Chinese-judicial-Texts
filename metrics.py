def get_chunk_type(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    if tag_name in ["SB-SE", "OB-OE"]:
        return tag_name.split('-')[0], tag_name.split('-')[1]
    elif tag_name in ["SB", "SE", "OB", "OE"]:
        return tag_name, None
    else:
        return tag_name, None
def get_chunks(seq, tags):
    default1 = tags['O']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        if tok == default1:
            if chunk_type is not None:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
        else:
            res = get_chunk_type(tok, idx_to_tag)
            if len(res) == 1:
                continue
            tok_chunk_class, tok_chunk_type = res
            if tok_chunk_class in ["SB-SE", "OB-OE"]:
                entity_type = "S" if "S" in tok_chunk_class else "O"
                chunk = (entity_type, i, i + 1)
                chunks.append(chunk)
            elif tok_chunk_class in ["SB", "OB"]:
                if chunk_type is not None:
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                entity_type = "S" if tok_chunk_class == "SB" else "O"
                chunk_type, chunk_start = entity_type, i
            elif tok_chunk_class in ["SE", "OE"]:
                if chunk_type is not None:
                    chunk = (chunk_type, chunk_start, i + 1)
                    chunks.append(chunk)
                    chunk_type, chunk_start = None, None
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks
def tag_mapping_bie(predict_tags, pre_rels=None, label2idx_sub=None, label2idx_obj=None):
    if predict_tags is None:
        return None
    pred_sub = torch.argmax(predict_tags[0], dim=-1)
    pred_obj = torch.argmax(predict_tags[1], dim=-1)
    pred_sub = pred_sub.cpu().numpy()
    pred_obj = pred_obj.cpu().numpy()
    idx2tag_sub = {v: k for k, v in label2idx_sub.items()}
    idx2tag_obj = {v: k for k, v in label2idx_obj.items()}
    mapped_tags = []
    for i in range(pred_sub.shape[0]):
        sub_tags = [idx2tag_sub.get(tag, 'O') for tag in pred_sub[i]]
        obj_tags = [idx2tag_obj.get(tag, 'O') for tag in pred_obj[i]]
        mapped_tags.append([sub_tags, obj_tags])
    return mapped_tags
def tag_mapping_nearest(predict_tags, pre_rels=None, label2idx_sub=None, label2idx_obj=None):
    rel_num = predict_tags.shape[0]
    pre_triples = []
    for idx in range(rel_num):
        heads, tails = [], []
        pred_chunks_sub = get_chunks(predict_tags[idx][0], label2idx_sub)
        pred_chunks_obj = get_chunks(predict_tags[idx][1], label2idx_obj)
        pred_chunks = pred_chunks_sub + pred_chunks_obj
        for ch in pred_chunks:
            if ch[0] == 'S':
                heads.append(ch)
            elif ch[0] == 'O':
                tails.append(ch)
        if len(heads) != 0 and len(tails) != 0:
            if len(heads) < len(tails):
                heads += [heads[-1]] * (len(tails) - len(heads))
            if len(heads) > len(tails):
                tails += [tails[-1]] * (len(heads) - len(tails))
        for h_t in zip(heads, tails):
            if pre_rels is not None:
                triple = list(h_t) + [pre_rels[idx]]
            else:
                triple = list(h_t) + [idx]
            pre_triples.append(tuple(triple))
    return pre_triples
def tag_mapping_corres(predict_tags, pre_corres, pre_rels=None, label2idx_sub=None, label2idx_obj=None):
    rel_num = predict_tags.shape[0]
    pre_triples = []
    for idx in range(rel_num):
        heads, tails = [], []
        pred_chunks_sub = get_chunks(predict_tags[idx][0], label2idx_sub)
        pred_chunks_obj = get_chunks(predict_tags[idx][1], label2idx_obj)
        pred_chunks = pred_chunks_sub + pred_chunks_obj
        for ch in pred_chunks:
            if ch[0] == 'S':
                heads.append(ch)
            elif ch[0] == 'O':
                tails.append(ch)
        retain_hts = [(h, t) for h in heads for t in tails if pre_corres[h[1]][t[1]] == 1]
        for h_t in retain_hts:
            if pre_rels is not None:
                triple = list(h_t) + [pre_rels[idx]]
            else:
                triple = list(h_t) + [idx]
            pre_triples.append(tuple(triple))
    return pre_triples