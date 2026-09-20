# Reference run

Measured January 2026 on Qwen3-4B, layer 0, transcoder feature 118230, using
`mwhanna/qwen3-4b-transcoders`. Activations read at the last token position, on
the space-prefixed form of each word, with no surrounding context.

The dataset stage reproduces exactly on any machine (`pytest` asserts the
counts). The activation numbers below come from the original GPU run and are
kept here so the analysis stage has something to check against.

## Dataset

| | count | share of closed |
|---|---:|---:|
| Source list | 1,720 | |
| Closed compounds | 1,566 | 100% |
| True single-token | 82 | 5.2% |
| Conditional single-token | 390 | 24.9% |
| True multi-token | 1,093 | 69.8% |
| Bare-single only | 1 | 0.1% |

The last row is `turnstile` — one token on its own, two with a leading space.
It is the mirror image of the conditional group and too small to analyse, but
it is reported rather than dropped.

## Activation by group

| Group | n | mean | std | non-zero |
|---|---:|---:|---:|---:|
| True single-token | 82 | 1.2202 | 3.1757 | 16 (19.5%) |
| Conditional single-token | 390 | 6.1271 | 5.5718 | — |
| True multi-token | 1,093 | 0.0298 | 0.4295 | 9 (0.8%) |

The conditional group's mean is 5× the true-single group's and 200× the
true-multi group's.

## Group comparisons

Student's t-test, as originally run:

| Comparison | t | p |
|---|---:|---:|
| Conditional > true single | 7.6991 | 8.16e-14 |
| Conditional > true multi | 35.8569 | 3.19e-203 |
| True single > true multi | 11.1031 | 2.63e-27 |

`scripts/analyze.py` defaults to Welch's t-test instead, which is the
appropriate test given the group sizes and unequal variances. Pass
`--equal-var` to reproduce the table above.

## Words that fired

**True single-token — 16 of 82 (19.5%), range 0.0356 to 20.2500**

| word | activation | | word | activation |
|---|---:|---|---|---:|
| heartbeat | 20.2500 | | dropout | 6.1250 |
| headline | 10.0000 | | thumbnail | 5.9688 |
| copyright | 8.6250 | | popover | 5.5312 |
| breadcrumb | 8.0000 | | lifetime | 4.3438 |
| deadline | 7.4688 | | benchmark | 3.4688 |
| weekday | 7.3438 | | bookmark | 2.5312 |
| newsletter | 6.5000 | | sandbox | 2.2031 |
| | | | barcode | 1.6641 |
| | | | passport | 0.0356 |

**True multi-token — 9 of 1,093 (0.8%), range 0.1855 to 9.1250**

| word | activation | tokenization | last token measured |
|---|---:|---|---|
| lighthouse | 9.1250 | ` l` + `ighthouse` | `ighthouse` |
| summertime | 7.5938 | ` summ` + `ertime` | `ertime` |
| peppermint | 6.3438 | ` pepp` + `ermint` | `ermint` |
| lifeguard | 3.2812 | ` lif` + `eguard` | `eguard` |
| skyscraper | 2.2031 | ` skys` + `craper` | `craper` |
| oversleep | 1.9844 | ` overs` + `leep` | `leep` |
| jigsaw | 1.2031 | ` j` + `igsaw` | `igsaw` |
| limelight | 0.6055 | ` lim` + `elight` | `elight` |
| humankind | 0.1855 | ` hum` + `ankind` | `ankind` |

Every one of these nine splits *across* the morpheme boundary rather than at
it — `lighthouse` becomes ` l` + `ighthouse`, not ` light` + `house`. So the
group label is doing something slightly different from what it says: these
words are multi-token, but their final token is a long, near-whole-word chunk,
which is the same kind of unit the conditional group's single token is. The
feature is not confused by them so much as consistent about them.

## Capitalization

Regenerate with `python scripts/capitalization.py` (tokenizer only, no GPU):

| Group | as written | capitalized |
|---|---:|---:|
| True single-token | 82 | 84 |
| Conditional single-token | 390 | 93 |
| True multi-token | 1,093 | 1,383 |
| Bare-single only | 1 | 6 |

Capitalization removes three quarters of the conditional group: 295 of the 390
words become true multi-token, because `Firefighter` was never frequent enough
in the corpus to earn its own merge, while ` firefighter` was.

Eleven words move the other way, into the conditional group, and they are
almost all proper nouns or place-name elements: *Superman, Thanksgiving,
Parkway, Pathfinder, Highland, Highlander, Typeface, Riverside, Keystone,
Wonderland, Snapdragon*. Those are words whose capitalized form is the common
one in text. That is what the merge-frequency account predicts, and it is a
cheap, tokenizer-only check of it.

This section was not completed in the original notebook.
