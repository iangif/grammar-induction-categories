"""
Export one row per token with its induced preterminal category.

Goal:
- load or receive a trained model
- run forward_parser(...)
- extract one row per word
- save rows as a CSV/dataframe

Inside grammar-induction-categories/ run:
    python -m vc-pcfg.analysis.export_word_categories
"""

from pprint import pprint

from ..vpcfg.utils import Vocabulary

def rows_from_batch(captions, lengths, ids, argmax_spans, vocab):
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

        temp = []
        num_leaf_spans = 0
        
        # Loop through each span [a]
        for a in argmax_spans[b]:
            # left word position, right word position, category
            l, r, cat = a

            # Add a row for each leaf
            if l == r and l < sent_len:
                word_id = int(captions[b][l])
                word = vocab.idx2word[word_id]

                left_context = " ".join(sentence_words[:l])
                right_context = " ".join(sentence_words[l + 1:])

                temp.append({
                    "sent_id": sent_id,
                    "sent_len": sent_len,
                    "word_index": l,
                    "word_id": word_id,
                    "word": word,
                    "viterbi_preterminal": cat,
                    "left_context": left_context,
                    "right_context": right_context,
                    "sentence": sentence,
                })

                num_leaf_spans += 1
         
        for row in temp:
            row["num_leaf_spans"] = num_leaf_spans
            row["leaf_span_matches_length"] = num_leaf_spans == sent_len
        rows.extend(temp)
        
    return rows

if __name__ == "__main__":
    captions = [[1, 2, 3]]
    lengths = [3]
    ids = [42]
    argmax_spans = [[
        (0, 0, 7),
        (1, 1, 8),
        (2, 2, 7),
        (0, 1, 3),
        (0, 2, 1)
    ]]
    vocab = Vocabulary()
    vocab.add_word("the")
    vocab.add_word("dog")
    vocab.add_word("runs")

    rows = rows_from_batch(captions, lengths, ids, argmax_spans, vocab)
    pprint(rows)