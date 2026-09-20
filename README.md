# Signs of tokenization awareness in Qwen3-4B

Feature **118230** in the layer-0 transcoder for Qwen3-4B fires on English
compound words — but not on all of them. It fires on the compounds whose
tokenization is *unstable*: one token when preceded by a space, several when
not.

```
newsletter    →  newsletter          firefighter  →  fire · fighter
 newsletter   →   newsletter          firefighter →   firefighter
                  ↑ one token either way              ↑ one token only with the space
```

The words that are stably one token barely register. The words that are stably
several tokens barely register. The ambiguous middle lights up. That looks like
a compensatory signal: at layer 0, before anything semantic has happened, the
model is flagging words whose identity its own tokenizer left unresolved.

📄 **[Read the write-up](https://solidgoldmagikarp.github.io/tokenization-awareness/)** ·
🔬 **[Feature on Neuronpedia](https://www.neuronpedia.org/qwen3-4b/0-transcoder-hp/118230)** ·
✍️ **[Original post on Medium](https://medium.com/@solidgoldmagikarp/a-breakthrough-feature-signs-of-tokenization-awareness-in-llms-058fe880ef9f)**

---

## The result

1,566 closed compounds, grouped by how Qwen3-4B's BPE tokenizer treats them
with and without a leading space:

| Group | Example | With space | Without | n | Mean activation |
|---|---|---|---|---:|---:|
| **True single-token** | `newsletter` | 1 token | 1 token | 82 | 1.22 |
| **Conditional single-token** | `firefighter` | 1 token | 2 tokens | 390 | **6.13** |
| **True multi-token** | `sunflower` | 2 tokens | 2 tokens | 1,093 | 0.03 |

All three pairwise differences are significant (p < 1e-13). The conditional
group's mean is 5× the true-single group's and 200× the true-multi group's.

![Feature activation by tokenization group](results/figures/activation_by_token_type.png)

Word frequency alone does not account for it. Activation is weak for very
frequent compounds, spikes in the middle of the frequency range, and drops off
again for rare ones — the shape you get if the feature tracks *merge history*
rather than commonness.

![Frequency vs activation](results/figures/frequency_vs_activation.png)

Full numbers, including every word that fired:
**[results/reference_run/RESULTS.md](results/reference_run/RESULTS.md)**

## Install

```bash
git clone https://github.com/solidgoldmagikarp/tokenization-awareness.git
cd tokenization-awareness
pip install -e .            # dataset + analysis, CPU only
pip install -e ".[model]"   # adds torch + TransformerLens for the measurement stage
```

Python 3.10+. The measurement stage wants ~9 GB of VRAM for Qwen3-4B in
bfloat16; a free Colab T4 handles the full sweep in about 15 minutes. Every
other stage runs on a laptop.

## Run it

The pipeline is three stages, each writing a file the next one reads, so the
GPU step happens once and the analysis can be re-run freely.

```bash
python scripts/build_dataset.py        # → results/groups.json         (seconds, CPU)
python scripts/measure_activations.py  # → results/activations.json    (~15 min, GPU)
python scripts/analyze.py              # → statistics + results/figures/
```

Two more scripts, both tokenizer-only:

```bash
python scripts/capitalization.py       # → results/capitalization.json
python scripts/build_page_data.py      # → docs/assets/tokens.js, for the write-up
```

Every script takes `--help`. Useful flags:

```bash
python scripts/build_dataset.py --capitalize          # group the capitalized forms
python scripts/measure_activations.py --feature 12345 # point at a different feature
python scripts/measure_activations.py --limit 20      # smoke test before the full sweep
python scripts/analyze.py --equal-var                 # reproduce the original t-tests
```

## Using it as a library

```python
from tokenization_awareness import build_groups, load_compounds
from tokenization_awareness.dataset import load_tokenizer

tokenizer = load_tokenizer()
groups = build_groups(tokenizer, load_compounds())

print(groups.summary())
print(groups.conditional_single[:5])
# ['nevertheless', 'newborn', 'afternoon', 'newcomer', 'folklore']
print(groups.spaced("conditional_single")[:2])
# [' nevertheless', ' newborn']
```

```python
from tokenization_awareness.activations import load_model, measure_activations
from tokenization_awareness.transcoder import load_transcoder

model = load_model()
transcoder = load_transcoder(layer=0)
measure_activations(model, transcoder, [" firefighter", " sunflower"])
```

## Layout

```
tokenization_awareness/
    config.py        model, transcoder, feature index, paths
    dataset.py       word list → the three tokenization groups
    transcoder.py    load transcoder weights; encode MLP inputs to features
    activations.py   run the model, read one feature at the last token
    analysis.py      descriptive stats, t-tests, firing words, Zipf correlations
    plots.py         the two figures
scripts/             the pipeline stages, each with a CLI
data/compounds.txt   1,720 English compounds
results/             regenerated outputs; reference_run/ is committed
notebooks/           the original Colab notebook, unmodified
tests/               pytest suite
docs/                the write-up, served as a GitHub Page
```

## Tests

```bash
pip install -e ".[dev]"
pytest                  # 20 tests
pytest -m "not slow"    # skip the ones that download the tokenizer
```

The slow tests assert the exact group counts (82 / 390 / 1,093), so any change
that quietly alters the dataset fails loudly.

## Publishing the write-up

The page is plain HTML with no build step. In the repository settings, under
**Pages**, choose *Deploy from a branch* → `main` → `/docs`. A `.nojekyll` file
is already in place so GitHub serves the assets untouched.

## Method notes

- **Activations are measured on the space-prefixed form** of every word.
  Spacing is the independent variable, so it is never added silently.
- **The reading is taken at the last token position**, which for a split word
  is the position that has attended to all the others — where a detokenization
  signal would have to appear.
- **The hook point is `blocks.0.ln2.hook_normalized`**: the layer-normalised
  residual stream entering the MLP, which is what the transcoder was trained on.
- **Words are run bare, without context.** That keeps the manipulation clean
  but means the result is about lexical processing, not about how the feature
  behaves mid-sentence.

## Differences from the original notebook

The notebook is preserved unmodified in `notebooks/`. The refactor is faithful
to it — the dataset counts match exactly — with four changes:

1. **Group assignment is per-word, not by set difference.** The original built
   groups by subtracting sets, which made ordering non-deterministic between
   runs. Each word is now classified once from its two token counts, so groups
   are disjoint by construction and order is preserved.
2. **One word is no longer dropped.** `turnstile` is one token bare and two
   with a leading space, so it fell out of every set-difference group. It now
   lands in `bare_single_only` and gets reported.
3. **The capitalization section runs to completion.** It ended in an infinite
   loop in the notebook (a helper was passed the same list as both source and
   accumulator). Results are in
   [RESULTS.md](results/reference_run/RESULTS.md#capitalization).
4. **Welch's t-test by default**, given group sizes spanning an order of
   magnitude and visibly unequal variances. `--equal-var` restores the original
   Student's test; the conclusions are unchanged either way.

## Limitations

The honest list, mostly carried over from the write-up:

- **One feature, one layer, one model.** Nothing here shows the pattern
  generalises beyond Qwen3-4B's layer 0.
- **Transcoders only capture the MLP.** Attention is invisible to this method,
  and reconstruction is imperfect, so "the feature does X" is a claim about the
  transcoder's picture of the model.
- **The tokenizer may be in the training data.** Qwen's tokenizer files are
  public. Memorisation cannot be ruled out — though a feature that generalises
  the pattern to unseen words is still a feature that learned something.
- **1,566 English compounds** from one list. No other language, no other word
  class, no held-out set.
- **Correlational.** No ablation, no causal intervention on the feature.

## Cite

See [CITATION.cff](CITATION.cff).

## Credits

Compound word list from
[proofreadingservices.com](https://www.proofreadingservices.com/pages/compound-words-list).
Transcoders by [Michael Hanna](https://huggingface.co/mwhanna/qwen3-4b-transcoders).
Feature exploration via [Neuronpedia](https://www.neuronpedia.org/)'s Circuit
Tracer. Hooks by [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
Frequency scores by [wordfreq](https://doi.org/10.5281/zenodo.7199437).

Written as a research proposal for Neel Nanda's MATS stream, Summer 2026.

MIT licensed.
