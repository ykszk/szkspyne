# Shape Alignment for Active Shape Models

`point_data.py` builds an Active Shape Model (ASM) from annotated landmarks: the shapes are
aligned to a common frame, then PCA is applied to the aligned shapes. This document describes
the two alignment methods (`--alignment reference` and `--alignment procrustes`), what they
store in the model, and how the fitter in [scolrs](https://github.com/ykszk/scolrs) uses it.

## Input

Each annotation becomes a `MovablePoints`:

- **points**: all landmarks of `settings.ASM_LABEL_NAMES`, concatenated label by label
  ($n$ points; `point_counts` gives the number per label)
- **reference points**: one point for each of `settings.ANCHOR_LABEL_NAMES` (`None` if missing)

The label sets depend on the settings: with `USE_NECK_SETTINGS=1` the neck labels are used,
with reference points AnteriorDens, PosteriorDens, Lamina1, C3_BL and C3_BR.

All samples must have the same point counts, and template selection needs every reference
point of every sample.

## Similarity Transform

Both methods align with a similarity transform (scale $s$, rotation $\mathbf{R}$,
translation $\mathbf{t}$), estimated by least squares (Umeyama, `skimage.transform.SimilarityTransform.estimate`):

$$\min_{s, \mathbf{R}, \mathbf{t}} \sum_{i} \left\| s\mathbf{R}\mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \right\|^2$$

where $\mathbf{p}_i$ are the source points and $\mathbf{q}_i$ the corresponding destination points.

## Template Selection

The template defines the frame of the model (its position, scale and orientation).
`MovablePoints.select_template` picks a typical sample:

1. Take sample 0 as a tentative template.
2. For every sample $k$ (sample 0 included, giving the identity), estimate the similarity
   transform from its reference points to those of sample 0, and describe it by
   $\boldsymbol{\theta}_k = (s_k, t_{x,k}, t_{y,k}, \phi_k)$.
3. Return the sample with the smallest Mahalanobis distance from the mean parameters:

$$k^* = \arg\min_k \sqrt{(\boldsymbol{\theta}_k - \bar{\boldsymbol{\theta}})^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\theta}_k - \bar{\boldsymbol{\theta}})}$$

where $\bar{\boldsymbol{\theta}}$ and $\boldsymbol{\Sigma}$ are the mean and covariance of the parameters.

Both methods use the same template.

## Reference Alignment (`--alignment reference`, default)

Each shape is aligned to the template by the similarity transform estimated from its
**reference points only**, and the transform is applied to all of its points:

$$\mathbf{x}_k^{\text{aligned}} = T_k(\mathbf{x}_k), \qquad T_k = \arg\min_{T} \sum_{r} \left\| T(\mathbf{r}_{k,r}) - \mathbf{r}^{\text{template}}_r \right\|^2$$

The model stores:

- `template_points`: the template's points
- `template_reference`: the template's reference points

Only the few reference points are aligned, so the rest of the shape keeps its position, scale
and rotation relative to them. PCA then models this pose variation as shape variation: in the
neck model built this way, about 75% of the variance was pose.

## Procrustes Alignment (`--alignment procrustes`)

Generalized Procrustes analysis aligns each shape by **all of its points** to the mean shape:

1. Start with the mean shape $\bar{\mathbf{x}}$ equal to the template's points.
2. Align every shape to $\bar{\mathbf{x}}$ with the least-squares similarity over all points.
3. Average the aligned shapes into a new mean, and align the new mean to the template's points.
   This keeps the mean in the template's frame (otherwise it can shrink, rotate or drift).
4. Repeat from step 2 until the mean moves by less than $10^{-3}$ (at most 100 iterations;
   a warning is logged if it does not converge).
5. Align every shape to the final mean.

The model stores:

- `template_points`: the mean shape
- `template_reference`: the mean of the reference points, each mapped by its own shape's
  transform (missing reference points are skipped)

Pose is removed from the shape variation (about 2% of the variance remains pose in the neck model),
so fewer modes describe the same shapes: 95% of the variance needs 15 modes in the neck model.

## Model Output

PCA is applied to the flattened aligned shapes $[x_1, y_1, \ldots, x_n, y_n]$ of all samples.
Since alignment removes four degrees of freedom (scale, rotation and translation), the last
components have zero variance and an arbitrary basis.

The JSON (`--json`) contains:

| Key | Content |
|---|---|
| `mean`, `components`, `explained_variance`, `explained_variance_ratio` | PCA (scikit-learn) |
| `template_points`, `template_reference` | see the two methods above |
| `labels`, `reference_labels`, `point_counts` | landmark definition |
| `alignment` | `reference` or `procrustes` |
| `dataset_size` | number of shapes |
| `timestamp` | build time (UTC, ISO 8601) |
| `config` | fitting configuration given by `--config`, if any |

The signs of the components depend on the scikit-learn version, so models built with
different versions may differ in sign while describing the same shapes.

## How the Fitter Uses the Model

The fitter (`scolrs::asm`) treats both kinds of model the same way:

1. **Initial pose**: the similarity transform from the reference points detected in the image
   to `template_reference` (at least two are needed) places the model in the image.
2. **ICP initialization**: alternates between matching the model to the heatmap peaks and
   updating the shape parameters and a similarity pose.
3. **Heatmap fitting**: optimizes the shape parameters with the pose fixed, using
   increasingly less smoothed heatmaps.

With reference alignment, `template_reference` is consistent with how the training shapes were
aligned, and the shape modes depend on the reference points carrying the pose. With Procrustes
alignment, `template_reference` only gives the initial pose, and the pose update of the ICP
initialization corrects it (without that update, fitting fails).

## Fitting Configuration

`--config` takes a TOML or JSON file with a possibly partial fitting configuration (the
`AsmConfig` of scolrs), stored in the model as `config`. The fitter uses it over its defaults; a
configuration file given with `deepscol --asm model.json,config.toml` and `ASM__*`
environment variables override it in turn. Settings that depend on the alignment, such as the
number of modes and the regularization weight, thus travel with the model.

The neck model uses [`conf/asm_fit_neck.toml`](../conf/asm_fit_neck.toml):

```toml
[mode]
variance = 0.95

[fit]
lambda = 0.005
```

## Choosing the Alignment

- **Neck**: Procrustes alignment, used by `task_build_pca` in `dodo_neck.py`.
- **AP and lateral**: reference alignment. The reference points at both ends of the spine
  (C7 and S) help choose between the models with different numbers of vertebrae; with
  Procrustes alignment, the correct model was chosen less often when ASM was forced.

## Usage

```sh
USE_NECK_SETTINGS=1 python -m szkspyne.asm.point_data pca_model.npz \
    --input train.ndjson --labelme \
    --alignment procrustes --config szkspyne/conf/asm_fit_neck.toml \
    --json pca_model.json
```
