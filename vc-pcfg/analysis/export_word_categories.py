"""
Export one row per token with its induced preterminal category.

Goal:
- load or receive a trained model
- run forward_parser(...)
- extract one row per word
- save rows as a CSV/dataframe

To first initialize a model, run:
    cd vc-pcfg
    python as_train.py --data_path ../preprocessed-data/abstractscenes --logger_name analysis/outputs/random_init_test --tiny --visual_mode --init_only --encoder_file "all_as-resn-50.npy"

Run:
    cd vc-pcfg
    python -m analysis.export_word_categories --model_init analysis/outputs/random_init_test/checkpoints/checkpoint.pth.tar --data_path ../preprocessed-data/abstractscenes --output_path analysis/outputs/random_word_categories.csv --use_mean_z
"""

from typing import Any

import argparse
from pathlib import Path
import logging

import pandas as pd
import torch
from torch_struct import SentCFG

from vpcfg.utils import Vocabulary
from vpcfg.as_vocab import get_vocab
from vpcfg.as_dataloader import get_data_iters, set_constant

# Saves rows to CSV and returns DataFrame
def save_rows(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")
    print(df.head())

    return df

# Returns rows given a single batch
def rows_from_batch(captions, lengths, ids, viterbi_preterminals, vocab):
    rows = []

    # Loop through each sentence with idx [b]
    for b in range(len(lengths)):
        sent_len = int(lengths[b])
        sent_id = ids[b]

        # Construct sentence context
        sentence_words = [
            vocab.idx2word[int(captions[b][i])]
            for i in range(sent_len)
        ]
        sentence = " ".join(sentence_words)

        sentence_categories = viterbi_preterminals[b, :sent_len].cpu().tolist()
        if len(sentence_categories) != sent_len:
            raise RuntimeError(
                f"Sentence {sent_id} has length {sent_len}, but "
                f"{len(sentence_categories)} categories were returned."
            )

        for word_index, category in enumerate(sentence_categories):
            word_id = int(captions[b][word_index])
            word = vocab.idx2word[word_id]

            rows.append({
                "sent_id": sent_id,
                "sent_len": sent_len,
                "word_index": word_index,
                "word_id": word_id,
                "word": word,
                "viterbi_preterminal": int(category),
                "left_context": " ".join(
                    sentence_words[:word_index]
                ),
                "right_context": " ".join(
                    sentence_words[word_index + 1:]
                ),
                "sentence": sentence,
                "num_preterminal_assignments": len(
                    sentence_categories
                ),
                "preterminal_matches_length": (
                    len(sentence_categories) == sent_len
                ),
            })

    return rows

def test_rows_from_batch():
    vocab = Vocabulary()
    vocab.add_word("the")
    vocab.add_word("dog")
    vocab.add_word("runs")

    captions = [[1, 2, 3]]
    lengths = [3]
    ids = [42]

    argmax_spans = [[
        (0, 0, 7),
        (1, 1, 8),
        (2, 2, 7),
        (0, 1, 3),
        (0, 2, 6),
    ]]

    rows = rows_from_batch(
        captions=captions,
        lengths=lengths,
        ids=ids,
        argmax_spans=argmax_spans,
        vocab=vocab,
    )

    for row in rows:
        print(row)

    assert len(rows) == 3
    assert [row["word"] for row in rows] == ["the", "dog", "runs"]
    assert all(row["leaf_span_matches_length"] for row in rows)

    return rows

# DataLoader --> batches --> rows --> CSV file / DataFrame
def export_from_loader(model, data_loader, vocab, output_path, log_every=10, use_mean_z=False):
    all_rows = []

    for i, (images, captions, lengths, ids, spans) in enumerate(data_loader):
        if torch.cuda.is_available():
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths).long()
            lengths = lengths.cuda()
            captions = captions.cuda()
        
        viterbi_preterminals = extract_viterbi_preterminals(
            model=model,
            captions=captions,
            lengths=lengths,
            use_mean_z=use_mean_z,
        )

        batch_rows = rows_from_batch(
            captions=captions.detach().cpu(),
            lengths=lengths.detach().cpu() if torch.is_tensor(lengths) else lengths,
            ids=ids,
            viterbi_preterminals=viterbi_preterminals.detach().cpu(),
            vocab=vocab,
        )

        all_rows.extend(batch_rows)

        if i % log_every == 0:
            print(f"[{i}/{len(data_loader)}] rows so far: {len(all_rows)}")
    
    return save_rows(all_rows, output_path)

# Returns the preterminal category selected for every token by the globally optimal derivation
def extract_viterbi_preterminals(model, captions, lengths, *, use_mean_z=False):
    # Compute grammar potentials
    with torch.no_grad():
        params, _ = model.parser(captions, use_mean=use_mean_z)
        params = tuple(param.detach() for param in params)
    
    with torch.enable_grad():
        dist = SentCFG(params, lengths=lengths)
        terminal_usage, _rule_usage, _root_usage, _span_chart = dist.argmax

    terminal_usage = terminal_usage.detach()

    # Shape: [batch_size, max_sentence_length, number_of_preterminals]
    viterbi_preterminals = terminal_usage.argmax(dim=-1)

    for b, raw_length in enumerate(lengths):
        sent_len = int(raw_length)
        selected_per_word = (
            terminal_usage[b, :sent_len]
            .ne(0)
            .sum(dim=-1)
        )

        if not (selected_per_word == 1).all().item():
            raise RuntimeError(
                "Expected exactly one Viterbi preterminal per word, "
                f"but sentence {b} had selection counts "
                f"{selected_per_word.cpu().tolist()}."
            )

    return viterbi_preterminals

    

# Loads model and exports from loader
def main_export_word_categories(opt):
    # Prevents weights only error
    torch.serialization.add_safe_globals([argparse.Namespace])

    checkpoint = torch.load(opt.model_init, map_location="cpu")
    model_opt = checkpoint["opt"]

    vocab = get_vocab(opt.data_path)
    model_opt.vocab_size = len(vocab)

    from vpcfg.model_vis import VGCPCFGs

    # DON'T USE model
    # if not model_opt.visual_mode:
    #     from vpcfg.model import VGCPCFGs
    # else:
    #     from vpcfg.model_vis import VGCPCFGs 

    logger = logging.getLogger(__name__)
    model = VGCPCFGs(model_opt, vocab, logger=logger)
    model.set_state_dict(checkpoint["model"])
    model.eval()

    set_constant(model_opt.visual_mode, model_opt.max_length)

    _, _, sem_test_loader = get_data_iters(
        opt.data_path,
        model_opt.prefix,
        vocab,
        model_opt.batch_size,
        model_opt.workers,
        load_img=model_opt.visual_mode,
        encoder_file=model_opt.encoder_file,
        img_dim=model_opt.img_dim,
        shuffle=False,
        sampler=None,
        tiny=opt.tiny,
        one_shot=model_opt.one_shot,
    )

    export_from_loader(
        model=model,
        data_loader=sem_test_loader,
        vocab=vocab,
        output_path=opt.output_path,
        log_every=opt.log_step,
        use_mean_z=opt.use_mean_z,
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_init", default=None)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", default="outputs/word_categories.csv")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--use_mean_z", action="store_true",
        help=(
            "Use the posterior mean of z instead of sampling z. "
            "This makes category exports deterministic."
        ),
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Test extraction logic
    # rows = test_rows_from_batch()
    # save_rows(rows, "analysis/outputs/test_word_categories.csv")

    opt = parse_args()
    main_export_word_categories(opt)