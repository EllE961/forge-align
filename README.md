<div align="center">

# Forge-Align

**DPO alignment, from scratch, for [Forge](https://github.com/EllE961/forge-llm) -- built to catch reward hacking if
it happens, and to report honestly that, at this scale, something more interesting happened instead.**

![python](https://img.shields.io/badge/python-3.10+-1f1d1b?style=flat-square&labelColor=0d0c0b)
![pytorch](https://img.shields.io/badge/pytorch-1.13%20%7C%202.x-d95926?style=flat-square&labelColor=0d0c0b)
![tests](https://img.shields.io/badge/tests-13%20passing-3987e5?style=flat-square&labelColor=0d0c0b)
![license](https://img.shields.io/badge/license-MIT-1f1d1b?style=flat-square&labelColor=0d0c0b)

<img src="docs/assets/race.gif" width="100%" alt="base model vs. DPO-aligned model generating the same dragon prompt, token by token, with live reward bars">

</div>

Most "I built a language model from scratch" projects stop at pretraining. This one asks the next question: once you
*align* that model to follow an instruction, does it actually get better at it, or does it just get better at
*looking like* it satisfies whatever you're measuring? Both DPO and the measurement that catches it are implemented
here from first principles, on top of [`forge-llm`](https://github.com/EllE961/forge-llm), the 27M-parameter
Llama-style model I trained from scratch in that repo.

**What actually happened, in one sentence:** the textbook reward-hacking curve (proxy up, true quality down) never
clearly showed up -- instead, the most tightly-constrained run (beta=1.0) overfit 28 training pairs so hard it got
*worse* than the untouched base model on both axes, while the least-constrained run (beta=0.02) generalised best and
even came out *more* fluent, not less. Both are real failure/success modes worth knowing about; neither is the one
this repo set out to find, and both are reported as measured, below.

## The setup

- **Task:** constrained story generation -- *"write a short story about a `dragon` that ends `happy`"* -- given to a
  **base model that was never instruction-tuned**. Forge only ever learned to continue TinyStories text, so it
  satisfies these constraints by chance, not by training. That's deliberate: it means alignment here has to teach
  instruction-following itself, not just adjust an already-compliant model's tone.
- **Reward, written from scratch, with no external judge and no human labels:** does the story contain the keyword
  (with simple morphological variants), does its last sentences match the requested tone (a small hand-built
  happy/sad lexicon), is it close to a target length, and is it free of the degenerate repetition loops small LMs
  fall into. Every sub-score is in `[0, 1]`; see `align/reward.py`.
- **The reward deliberately never checks fluency.** Fluency is measured only at evaluation time, as the average
  log-probability of the generated text under the *original, frozen* base model -- a signal DPO never sees during
  training. If DPO pushes the proxy reward up while this independent fluency score falls, that's reward hacking,
  and this repo is built specifically to catch it if it happens, not to hide it if it does.
- **Preference pairs, built without a reward model or human raters:** sample K completions per task from the base
  model, score each with the reward above, and take the best-scoring completion as "chosen" and the worst as
  "rejected" (`align/preference_data.py`). Tasks where every sample scores the same are skipped -- no signal to
  train on, so no pair is manufactured.
- **DPO (Rafailov et al., 2023), from scratch:** no reward model, no PPO rollouts. A frozen reference copy of the
  base model and a trainable policy are compared directly through the closed-form optimal-policy substitution that
  turns reward maximisation into a classification loss (`align/dpo.py`). Trained at four values of `beta`, the
  implicit KL penalty, to see the tradeoff rather than report one cherry-picked run.

```mermaid
flowchart LR
    subgraph base["forge-llm (upstream)"]
        A["27M-parameter base model<br/>(TinyStories, no instruction tuning)"]
    end
    subgraph pairs["Preference data (this repo)"]
        A --> B["sample K completions / task<br/>(constrained story prompts)"]
        B --> C["score with align.reward<br/>keyword · tone · length · diversity"]
        C --> D["chosen = best · rejected = worst"]
    end
    subgraph train["DPO training"]
        D --> E["policy (trainable)<br/>+ reference (frozen copy)"]
        E --> F["DPO loss, beta in {0.02, 0.1, 0.3, 1.0}"]
    end
    subgraph eval["Evaluation"]
        F --> G["proxy reward<br/>(the training signal)"]
        A -. "never trained on" .-> H["true fluency<br/>(held-out log-prob under the frozen base model)"]
        G --> I{"did proxy rise<br/>while fluency fell?"}
        H --> I
    end
```

## Results

All numbers below are measured on the RTX 3080 that produced `forge-llm`'s numbers, re-running the exact commands
in [Quickstart](#quickstart) -- nothing here is hand-picked except the single qualitative example, which is labelled
as such.

**Base model** (re-trained from scratch for this repo, not reused from `forge-llm`'s run): 26.7M parameters, 8,109
steps, 265.7M tokens, 75.1 minutes, best validation loss **1.2716** (perplexity 3.57) -- in line with (slightly
better than) the original run, a sanity check that this copy behaves the same way. **Preference pairs:** 28/28
training tasks and 12/12 held-out tasks produced a usable chosen/rejected pair; none were skipped for lacking signal.

| beta | proxy reward (held out) | true fluency (held out) | vs. base |
|---|---|---|---|
| base model (no DPO) | 0.294 | -2.504 | -- |
| 0.02 | **0.344** (+17.0%) | **-2.360** (better) | better on both axes |
| 0.10 | 0.336 (+14.3%) | -2.542 (worse) | better reward, worse fluency |
| 0.30 | 0.342 (+16.3%) | -2.614 (worse) | better reward, worse fluency |
| 1.00 | 0.287 (-2.4%, worse than base) | -2.629 (worse) | **worse on both axes** |

<img src="docs/assets/beta_sweep.png" width="100%" alt="constraint satisfaction vs. held-out fluency across four DPO beta values">

Reading left to right as beta increases: held-out proxy reward and held-out fluency move *together*, not against
each other. The loosest constraint (beta=0.02) is both the most constraint-satisfying and the most fluent of the
four; the tightest (beta=1.0) is both the least constraint-satisfying -- worse than never aligning at all -- and the
least fluent. That is the opposite of a classic reward-hacking curve, and it is what 28 preference pairs actually
produced.

<img src="docs/assets/overfitting_trajectory.png" width="100%" alt="beta=1.0: proxy reward and fluency rise together then fall together over DPO training steps">

The beta=1.0 trajectory shows why: proxy and fluency rise together for the first 20 steps, then **fall together**
and never recover, ending below the untouched base model on both. Chasing 28 examples too hard with too little
tolerance for drift from the reference doesn't trade fluency for reward -- it loses both, which is overfitting, not
reward hacking. (All four betas land on nearly identical values at step 20; 20 steps at `lr=1e-5` move the weights
little enough that sampling at temperature 0.9 mostly reproduces the same tokens regardless of beta. The runs only
diverge from step 40 on.)

<img src="docs/assets/training_dynamics.png" width="100%" alt="DPO loss, implicit reward margin and pairwise accuracy over training, beta=1.0">

Internally, DPO training "succeeds" by every number it can see: loss collapses to ~0, the implicit reward margin
between chosen and rejected explodes past +20 nats, pairwise training accuracy hits 1.0 -- all within 30 steps, on
28 pairs seen 3 times. That is textbook overfitting of a tiny preference set, and it is exactly why the held-out
numbers above matter more than anything `align/dpo.py` prints while training.

**One concrete before/after**, same prompt and seed, base model vs. the beta=0.02 policy (the strongest run):

> **Prompt:** *Write a short story about a dragon. The story should end happy.*
>
> **Base model:** "...there was a girl named Lucy who loved to cook. She had many meals to eat and she always made
> everyone around her smile. One day, the wind blew hard outside, making everything freeze up. Everything got
> frozen in place! Lucy couldn't help but feel a little bit sad because no one wanted to play with her anymore..."
> *(no dragon; drifts sad despite being asked for happy -- reward 0.24)*
>
> **DPO-aligned (beta=0.02):** "...there was a big black dragon who loved to cook. He lived in a dark cave near the
> sea. Every day at noon, he would go out and make soup for all his friends. One sunny Sunday, a little boy named
> Tim came to the castle. 'Hello,' said Tim with a big smile on his face. 'What are you making?' asked the dragon.
> 'I am cooking something yummy!' said Tim, excited to try it." *(keeps the dragon, stays cheerful throughout --
> reward 0.64)*

<img src="docs/assets/token_shift.png" width="100%" alt="per-token probability shift after DPO for the dragon example above">

The token-level view of that same completion is the one genuinely surprising figure in this repo: the word
**"dragon" itself is colder (less likely) under the aligned model than under the base model**, at the exact position
it appears. DPO didn't simply boost the keyword's own token everywhere -- "who" and "black" immediately around it
got warmer instead. Whatever DPO changed about this model, it was not a lookup-table bump on the one word being
scored; the shift is diffuse across the surrounding context, which is also a small, honest piece of evidence against
the crudest version of "the model just learned to game the keyword check."

### How broad is this, really?

The dragon example above is real and reproducible (`seed=39` in `scripts/record_gif.py`), but it's also the best
one found while looking for a clear illustration -- it is not the average case, and the GIF above says so in its own
numbers. Two figures below exist specifically so the headline table can't be mistaken for the whole story.

<img src="docs/assets/component_breakdown.png" width="100%" alt="mean score per reward component -- keyword, ending, length, diversity -- base vs. each beta">

Averaged over all 12 held-out tasks x 3 samples, the base model mentions the requested keyword only **5% of the
time**; the best DPO runs reach **13-14%** -- roughly a 3x relative improvement, and a real one, but nowhere near
"it reliably follows the instruction." `diversity` is pinned near 1.0 for every model (repetition loops just
aren't this small model's problem at 90 tokens) and `length` barely moves either; essentially all of the measured
effect lives in `keyword` and `ending`, and both remain low in absolute terms. The single dragon example that
scores 100% on keyword is a real sample, not a fabricated one -- it's just in the right tail, not the middle, of
this distribution.

<img src="docs/assets/per_task.png" width="100%" alt="per-task proxy reward, base vs. beta=0.02, all 12 held-out tasks">

Per task, beta=0.02 improves **8 of the 12** held-out (keyword, tone) combinations and regresses on the other 4 --
broad enough to produce the positive mean in the headline table, not so broad that it's uniform. `snow/happy` and
`train/happy` improve the most; `forest/sad`, `cookie/happy` and `star/happy` are better off base. No pattern in
*which* tasks improve jumped out while writing this -- that would be a reasonable thing for a future pass to dig
into, rather than something to claim here without having actually checked.

## Why this design

**Fluency is withheld from the training reward on purpose.** The easy version of this project rewards fluency too,
and the aligned model looks great on every axis you optimised for. The interesting version holds one axis out, so
there's an honest answer to "did alignment actually make this better, or did it just get good at the test."

**The base model is deliberately not instruction-tuned.** An already-compliant model aligned further would make a
less convincing demo -- there would be little room for DPO to visibly change behaviour. Starting from a model that
mostly ignores the instruction gives DPO something real to do, and gives the before/after comparison in the
playground an actual difference to show.

**Best-vs-worst-of-K instead of a learned reward model.** A from-scratch PPO implementation (reward model + rollouts
+ value function + clipped surrogate objective) is a larger, noisier undertaking for a single GPU and a dataset this
small. DPO gets the same Bradley-Terry-optimal policy with one loss function and no rollouts, which is also exactly
why it has become the default for small-scale alignment work since 2023.

**Four betas, not one.** Beta is the one hyperparameter in DPO that directly trades off constraint satisfaction
against staying close to the reference model. Reporting a single run would hide that tradeoff; the sweep in
`configs/dpo-beta-sweep.json` is the point, not an afterthought.

## Repository map

```
align/              reward model, preference-pair sampling, DPO trainer, evaluation, FastAPI server
  reward.py           the proxy reward (keyword / tone / length / diversity), fluency-blind by design
  prompts.py           the 20 keywords x 2 tones = 40 constrained-generation tasks, with a fixed train/eval split
  model_utils.py       thin glue over forge-llm: load checkpoints, sample text, score log-likelihood
  preference_data.py   best-of-K / worst-of-K preference pair construction
  dpo.py               the DPO loss and training loop, from scratch
  evaluate.py          proxy reward vs. held-out fluency, per checkpoint and across the beta sweep
  server.py            playground backend
configs/             one JSON per DPO run, plus the beta-sweep manifest
docs/assets/          README figures and the race.gif, generated by scripts/
scripts/make_figures.py  renders every static figure from results/*.json, same visual system as forge-llm
scripts/record_gif.py   renders the base-vs-aligned token-by-token race GIF, no browser needed
tests/                reward scoring, the DPO loss itself (pure-tensor, no GPU needed), the task split
web/ + align/server.py  the comparison playground: base vs. every aligned model, live reward bars, token-shift view
```

## Quickstart

Needs [`forge-llm`](https://github.com/EllE961/forge-llm) installed and its base model trained/exported first
(see that repo's own `make data tokenizer tokenize train export`).

```bash
pip install "forge-llm @ git+https://github.com/EllE961/forge-llm.git"
pip install -e ".[dev]"

make pairs         # sample + score preference pairs from the base model
make train-sweep    # DPO at beta in {0.02, 0.1, 0.3, 1.0}
make eval-sweep      # proxy reward vs. held-out fluency, every run
make figures          # README figures -> docs/assets/
make gif               # the token-by-token race GIF -> docs/assets/race.gif
make serve               # playground at http://127.0.0.1:8002
make test                # 13 tests, CPU-only, no checkpoint required
```

## Limitations and next steps

- **The ending-tone constraint is much harder to satisfy than the keyword constraint at the generation length used
  here (90 tokens).** In every high-scoring example sampled for this README, the story is still mid-scene at token
  90 -- the keyword lands early and reliably, but the lexicon-based tone check, which only looks at the last one or
  two sentences, rarely has a resolved "happy ending" sentence to look at yet. The `ending` sub-score is close to
  zero even for the best completions shown above. A shorter target length or a tone check that scores the *whole*
  completion, not just its tail, would likely close this gap -- it wasn't changed after the fact to keep the
  reported numbers matched to what was actually run.
- The reward is lexicon-based and English-only; a dragon story that says "overjoyed" instead of "happy" scores zero
  on tone even though a human would call it a clear pass. That's a real limitation of rule-based rewards, not swept
  under the rug -- it's also exactly the kind of gap a learned reward model exists to close, at the cost of needing
  one.
- 40 tasks and a 27M-parameter base model keep this tractable on one GPU, but mean the preference dataset is small --
  28 training pairs. That smallness is very likely *why* beta=1.0 overfits instead of hacking the reward: there
  isn't enough data for the policy to find a cheap, generalising exploit, only enough to memorise the training set.
  Reward hacking in the textbook sense tends to show up with more data and more optimisation pressure, not less;
  this result should be read as what happens at this scale, not a claim that it generalises to larger ones.
- A natural follow-up: train a small learned reward model on the same best/worst-of-K data, and see whether a
  *learned* reward is harder or easier to overfit than this rule-based one at the same data scale.

## References

- Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.*
- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback* (InstructGPT / RLHF).
- Gao, L., Schulman, J., Hilton, J. (2022). *Scaling Laws for Reward Model Overoptimization* -- the reward-hacking
  curve this repo's evaluation harness was built to look for (`docs/assets/beta_sweep.png`,
  `docs/assets/overfitting_trajectory.png` show what was found instead, at this data scale).
- Eldan, R., Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*

## License

MIT, see [`LICENSE`](LICENSE).
