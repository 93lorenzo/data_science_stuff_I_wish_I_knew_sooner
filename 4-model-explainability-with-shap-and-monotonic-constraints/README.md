# Data Science stuff I wish I knew sooner: Model Explainability with SHAP and Monotonic Constraints

> Part of the series *Data Science stuff I wish I knew sooner* — practical insights that I learned during my experience

---

## Context

For a long time my idea of "explaining a model" was a feature importance bar chart, screenshotted
into a slide, mentioned once, and never opened again. It answers "what does the model use overall,"
and stops right there. The question that actually comes up in a review — "why did the model say
**this** about **this specific case**" — has no answer in a bar chart.

That gap is what [SHAP](https://github.com/shap/shap) closes: instead of one importance number per
feature, you get a number **per feature, per row**, and those numbers add up exactly to the model's
prediction. Global summary and single-case explanation from the same, consistent accounting.

This article trains an XGBoost classifier on the Breast Cancer Wisconsin dataset, extracts SHAP
values on the test set, and walks through the SHAP plots I reach for most: **beeswarm** (global
overview), **heatmap** and **dependence** plots (debugging a specific feature or subgroup), **bar
chart** (ranked importance), and **force** / **waterfall** plots (one prediction, fully decomposed).
Then it uses what the beeswarm plot surfaces — a feature whose relationship with risk isn't as clean
as it should be — to introduce **monotonic constraints**: a way to tell XGBoost "I already know which
direction this feature should push the prediction, don't second-guess it," verified with scikit-learn's
partial dependence / ICE tooling.

The companion notebook
[`4-Model-Explainability-With-SHAP-and-Monotonic-Constraints.ipynb`](4-Model-Explainability-With-SHAP-and-Monotonic-Constraints.ipynb)
implements every step end to end, with real numbers from a real (if small and clean) dataset — no
synthetic examples built to make the point look better than it is.

---

## Dataset

[Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset),
loaded directly from `sklearn.datasets` — 569 rows, 30 numerical features describing cell nuclei
from digitised breast mass images (`mean`, `standard error`, and `worst` value for radius, texture,
perimeter, area, smoothness, concavity, ...). The target is flipped from scikit-learn's default so
that **`1` means malignant** — the class you actually want to catch — which keeps every SHAP value
in the "pushes risk up / pushes risk down" direction you'd intuitively expect.

An XGBoost classifier (300 trees, depth 4, learning rate 0.05) trained on an 80/20 stratified split
reaches **0.993 test PR AUC** (average precision). Malignant cases are the minority class here
(~37% of rows), so PR AUC is the honest metric to report — it's graded against the positive class
directly, with the prevalence itself as its baseline, rather than ROC AUC's false-positive rate,
which stays flattering even for a mediocre model because it's computed against the larger negative
class. (See the [previous article on classification metrics](../2-classification-metrics) for the
full argument.) This dataset is clean and well separated, which is exactly why it's a good teaching
example: any messiness we find in the explanations is coming from the model's learned splits, not
from noisy labels.

---

## 1. What a SHAP value actually is

SHAP borrows the **Shapley value** from cooperative game theory: treat each feature as a player
contributing to the prediction, and split that prediction fairly across the players based on their
average marginal contribution across every possible ordering in which they could be revealed. The
result is additive:

```
f(x) = base_value + Σ shap_value(feature_i)
```

`base_value` is the average model output over the training set (what you'd predict knowing nothing
about the row). Each SHAP value moves the prediction up or down from there, and they always sum back
to the model's exact output — not an approximation.

For tree ensembles, `shap.TreeExplainer` computes these values **exactly**, in polynomial time, by
exploiting the tree structure — instead of the exponential brute-force game-theory computation, or
the sampling-based approximation that model-agnostic explainers (`KernelExplainer`) need. This is why
SHAP and gradient boosting are used together so often: for trees, "exact and interpretable" isn't a
trade-off you have to make.

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
```

---

## 2. The global view: the beeswarm plot

Every dot is one test-set row. Its x-position is the SHAP value (how much that feature pushed *this*
prediction); its colour is the feature's own value (red = high, blue = low); features are ranked
top-to-bottom by overall impact.

```python
shap.plots.beeswarm(shap_values)
```

![Beeswarm plot](img/beeswarm.png)

`mean concave points` is the single most influential feature, and a clean one — high values (red)
push hard toward malignant, low values (blue) push hard toward benign, with almost no overlap between
the two clusters. Most of the top features look like this.

`worst texture` doesn't. It's the only top feature with dots of *both* colours scattered across the
*whole* x-axis range — a visible sign that its relationship with risk is less tidy than the purely
geometric features. Not every feature the model relies on behaves as tidily as the top one.

The beeswarm plot is a great overview, but it can't answer two questions that come up constantly once
you're actually debugging a model: *do groups of similar patients get explained the same way?* and
*is this specific feature's relationship with risk actually as clean as it looks, or is another
feature quietly interacting with it?*

### 2a. Spotting patterns across patients: the heatmap plot

`shap.plots.heatmap` puts every test-set row along the x-axis (clustered so similarly-explained rows
sit next to each other) and every feature along the y-axis, colouring each cell by that feature's SHAP
value for that row. The thin line along the top traces `f(x)` — the model's actual output — for each
column, so explanation patterns and output level move together.

```python
shap.plots.heatmap(shap_values)
```

![Heatmap plot](img/heatmap.png)

Two clear blocks emerge: a left block where `mean concave points`, `area error`, `worst concave
points`, `worst radius`, and friends are all firmly red (pushing malignant) and `f(x)` sits high, and
a much wider right block where the same features are blue and `f(x)` sits low — the model learning one
dominant, coherent axis of risk. `worst texture`'s row is visibly messier than its neighbours even
within those blocks, which is the heatmap surfacing the same observation as the beeswarm plot, from a
different angle. A heatmap with several distinct, unexplained blocks (rather than one dominant
pattern) is often a sign the model is doing something different for a subgroup you haven't identified
yet — worth checking before shipping a model.

### 2b. Debugging one feature at a time: the dependence plot

`shap.plots.scatter` plots one feature's value against its own SHAP value — "does this feature's
effect actually look like what I'd expect." Colour it by SHAP's automatically-selected strongest
interacting feature and it doubles as an interaction detector: if the colour band isn't uniform at a
given x position, the feature is doing something in combination with another one, not on its own.

```python
shap.plots.scatter(shap_values[:, "mean concave points"], color=shap_values)
```

![Dependence plot](img/dependence.png)

`mean concave points` shows a clean two-cluster jump from strongly-negative to strongly-positive SHAP
values, matching the beeswarm's story. SHAP picked `worst compactness` as the strongest interaction:
within the malignant cluster, higher `worst compactness` (pink) tends to sit at the upper end of the
SHAP range — a secondary effect invisible in the beeswarm or bar plot, since both collapse every
feature to its own axis. This combination — one feature's own effect, plus whichever other feature
modulates it — is worth checking for every important feature before trusting a model in production.

---

## 3. Ranked importance: the bar plot

The bar plot is the beeswarm collapsed to one number per feature: the mean absolute SHAP value across
the test set. You lose direction and distribution, but for a "what matters most" answer with no room
for anything richer, it's the right tool.

```python
shap.plots.bar(shap_values)
```

![Bar plot](img/bar.png)

Same ranking as the beeswarm, as it must be — same numbers, different aggregation. Worth noticing:
the "sum of 21 other features" bar is the *largest* bar in the chart. No individual feature outside
the top 9 matters much on its own, but the long tail isn't negligible collectively — typical for
tabular data with many correlated measurements (here, `mean`/`error`/`worst` versions of the same 10
underlying cell properties).

---

## 4. Explaining one patient: the force plot and the waterfall plot

Beeswarm, heatmap, bar, and dependence are all *global* views — they describe the model's behaviour
across the whole test set. The force plot and the waterfall plot are *local*: they explain **one**
prediction, which is what actually answers "why did the model say this about this patient." Both are
direct visualisations of the additive equation from section 1 — they show the same numbers, laid out
differently.

We deliberately pick a **borderline** case — a confident prediction doesn't need either plot to be
believable.

### The force plot

Contributions laid out left-to-right as arrows on a single axis, ending at `f(x)`. Compact, good for a
quick read of the overall balance of push and pull.

```python
shap.plots.force(shap_values[i], matplotlib=True)
```

![Force plot for a single, borderline prediction](img/force_single_prediction.png)

`worst concave points` and `mean concave points` — both alarmingly high for this patient — push hard
toward malignant. `worst texture` — unremarkable for this patient — pulls back toward benign. The
tug-of-war lands the model at **64.8 % malignant**: correctly flagged as high-risk, with honest,
visible uncertainty rather than false confidence either way. That's a far more useful answer than a
bare "malignant: 65 %" — it tells you *which* measurements are driving the concern.

### The waterfall plot

Same numbers, read top-to-bottom instead, ordered by magnitude, with every value labelled explicitly.

```python
shap.plots.waterfall(shap_values[i])
```

![Waterfall plot for the same prediction](img/waterfall.png)

Identical story, but with two more contributors spelled out that the force plot's compact layout had
folded into "21 other features": `worst concavity` (+0.53) and `worst smoothness` (+0.49) also lean
toward malignant, while `area error` (-0.43) and `worst area` (-0.41) lean back toward benign. For a
quick visual the force plot is fine; for a record you'd attach to a case file, the waterfall plot is
the more complete one.

---

## 5. When you already know the direction: monotonic constraints

Everything above *explains* a model trained the ordinary way. This section changes how it's trained.

A gradient-boosted tree learns its splits purely by minimising loss on the training sample. Nothing
stops it from learning that risk goes up, then down, then up again as a feature increases — noise in
a sparse region is enough. For a purely predictive feature that's often harmless. But for a feature
where you already know the real-world direction from domain knowledge — more concave cell boundaries
cannot make a tumour *less* suspicious, more income cannot make a borrower *less* creditworthy, all
else equal — a local reversal isn't signal the model found, it's noise the model overfit to. And it's
exactly the kind of thing that costs trust the moment a clinician or auditor notices it: *"why did
the risk score go down when this measurement got worse?"*

XGBoost's `monotone_constraints` lets you rule this out entirely: declare, per feature, that the
prediction must be non-decreasing (`1`), non-increasing (`-1`), or left alone (`0`) as that feature
increases, holding everything else fixed. It's enforced structurally while the trees are built, not
patched on afterwards.

```python
monotone_constraints = {col: 0 for col in X.columns}
monotone_constraints["mean texture"] = 1  # risk must be non-decreasing in this feature

model_mono = xgb.XGBClassifier(**BASE_PARAMS, monotone_constraints=monotone_constraints)
model_mono.fit(X_train, y_train)
```

We constrain `mean texture`: texture variability is a well-established diagnostic signal in this
domain, it's a top-10 feature by SHAP (important enough that a hidden reversal matters), and "more
irregular texture shouldn't lower the estimated risk" is a defensible clinical prior.

### The cost: essentially nothing

| Model | Test PR AUC (average precision) |
|---|---|
| Unconstrained | 0.9926 |
| Monotonic constraint on `mean texture` | 0.9923 |

A difference in the fourth decimal place. That's the pattern to expect when constraining a feature
that was already mostly behaving — you're not fighting the data, you're removing a handful of local
reversals the model didn't need in the first place.

### Does it change anything for a real patient?

Averaged over the whole test set, "risk vs. texture" was already trending upward before the
constraint, so a global summary wouldn't show much difference. The constraint's actual guarantee is
narrower and more useful: for **any single patient**, sweeping this one feature up while holding
everything else about them fixed, the prediction can now only go up.

This is precisely the distinction scikit-learn's partial dependence tooling draws:

- **Partial Dependence (PDP)** — the *average* effect of a feature on the prediction, marginalising
  over every other feature. This is "interpretability" in the usual global sense: on average, across
  the population, what does this feature do?
- **Individual Conditional Expectation (ICE)** — the same sweep for one specific row, with every other
  feature held at that row's actual values. This is what the feature does to *this patient's*
  predicted label, specifically.

The PDP can look perfectly smooth while individual ICE curves underneath it still misbehave — the
average hides exactly the kind of local reversal we're trying to catch. We use scikit-learn's
[`partial_dependence`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html)
(`kind="both"`) to compute both at once, for one chosen patient, before and after the constraint.

![Partial dependence and ICE curve before and after the monotonic constraint](img/ice_monotonic_constraint.png)

There it is, in both the printed violation counts and the black highlighted curve: in the unconstrained
model, this patient's predicted risk climbs as `mean texture` rises — until it crosses roughly 22,
where it **drops** from a peak of 30.6 % back down to 24.6 % and flatlines there for every higher
value, including this patient's actual measurement (23.29). Holding their other 29 features fixed, a
*worse* texture reading would have made the model *less* worried about them. Meanwhile the dashed red
**average** (PDP) line barely notices — it keeps trending gently upward the whole time, because most
of the grey ICE lines around it are well-behaved. That's exactly the trap: the global, "interpretability"
view says everything is fine; the individual, per-label view for this one patient says otherwise.

With the constraint, the same sweep is a clean, non-decreasing step function throughout — zero
violations, as guaranteed. This patient is, in fact, malignant — and the constrained model's estimate
for them (36.6 %) is meaningfully higher than the unconstrained one (24.6 %), even though both still
sit under a naive 50 % cutoff. Fixing the local reversal didn't just make the curve prettier; it moved
a genuinely-at-risk patient's score in the right direction.

---

## What to do about it

Monotonic constraints aren't a free correctness upgrade to apply everywhere. They encode *domain
knowledge you trust more than the training data* in a specific region, so:

1. **Only constrain what you'd defend to a domain expert.** "I'm confident the real relationship is
   monotonic" is the bar — not "the dependence plot looks a little jagged." Constraining a feature
   whose true relationship *isn't* monotonic will only hurt the model.
2. **Check held-out performance before and after**, on a metric that matches the problem (PR AUC here,
   not ROC AUC). A well-chosen constraint on a feature that was already mostly monotonic should cost
   you close to nothing, as it did here. A large drop means you constrained the wrong feature, or the
   wrong direction.
3. **Look at ICE curves for individual rows, not just the average PDP.** The PDP can hide exactly the
   local reversal you're trying to catch — a smooth average is not proof that every individual
   prediction behaves.
4. **Constraints are per-feature and independent.** Apply `monotone_constraints` only to the handful
   of features you have a real prior on; leave the rest to learn freely.
5. **It doesn't fix everything.** A monotonic feature still can't be trusted to extrapolate sensibly
   *beyond the range it was trained on* — trees still flatline past the training boundary, as covered
   in [the previous article in this series](../3-boosting-regression-limits). Monotonic constraints
   and range monitoring solve two different problems.

For regulated or high-stakes domains — credit scoring, medical risk, insurance pricing — where a
stakeholder can reasonably ask "why would more of a bad thing ever lower the risk score?", this is a
cheap way to make sure the model never has to answer that question.

---

## Running the notebook

```bash
cd 4-model-explainability-with-shap-and-monotonic-constraints
uv sync
uv run jupyter notebook
```

Or, in VS Code: open the `.ipynb` file, select the `.venv` kernel from the kernel picker.

---

## References

- [SHAP documentation](https://shap.readthedocs.io/)
- [A Unified Approach to Interpreting Model Predictions (Lundberg & Lee, 2017)](https://arxiv.org/abs/1705.07874)
- [XGBoost — Monotonic Constraints](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html)
- [scikit-learn — `average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [scikit-learn — Partial Dependence and Individual Conditional Expectation plots](https://scikit-learn.org/stable/modules/partial_dependence.html)
