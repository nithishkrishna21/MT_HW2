# Word Alignment with IBM Models 1 & 2

Statistical word alignment for French-English sentence pairs (Canadian Hansards), implemented
from scratch using Expectation-Maximization.

- **IBM Model 1** (`IBM1_EM.py`): assumes every source word is equally likely to align to any
  target word, ignoring position. Trained with EM over translation probabilities.
- **IBM Model 2** (`IBM2_EM.py`): extends Model 1 with a position-dependent alignment
  probability, conditioned on sentence lengths and word positions.

## Results

Evaluated against gold alignments using precision, recall, and AER (Alignment Error Rate,
lower is better).

| Method | Precision | Recall | AER |
|---|:---:|:---:|:---:|
| Dice coefficient baseline | 0.239 | 0.595 | 0.682 |
| IBM Model 1 + EM | 0.456 | 0.604 | 0.497 |
| IBM Model 2 + EM | 0.578 | 0.790 | 0.354 |

IBM Model 2 outperforms Model 1 on every metric, since incorporating word position gives it a
better signal for alignment beyond translation probability alone.

## Running

```bash
python IBM1_EM.py -n num_sentences -i num_iterations > alignment_file.a
python IBM2_EM.py -n num_sentences -i num_iterations > alignment_file.a
python score-alignments < alignment_file.a
```

- `num_sentences`: how many sentence pairs to train on (1 to 10000)
- `num_iterations`: EM iterations (5 is usually enough to converge)

`EM_IBM1_alignment.a` and `alignment` are the alignment outputs from the two models; the
`*.jpg` files are screenshots of the EM training runs.

**Authors:** Nithish Krishna Shreenevasan, Khushang Zaveri, Avantika Singh
