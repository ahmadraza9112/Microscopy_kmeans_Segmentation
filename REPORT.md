# Academic Project Report

## Unsupervised Microscopy Image Segmentation Using K-Means Clustering

**Type:** Independent introductory machine-learning project | **Year:** 2026  
**Course topic alignment:** BIME740413 / Machine Learning — unsupervised learning, clustering, image preprocessing, statistical evaluation.  
**Status:** Implemented and locally executed. The project is **not confirmed** as any prior course's final submission.

### Abstract

This introductory project investigates unsupervised pixel-intensity clustering for microscopy-image segmentation. Three methods were implemented: Otsu thresholding and K-Means clustering with either two or three clusters. A publicly accessible quantitative-phase microscopy sample from scikit-image was used for qualitative illustration. Since that image does not include an independent segmentation annotation, a separate controlled benchmark of six synthetic microscopy-like images was generated with known binary reference masks. The benchmark evaluated Dice, intersection over union (IoU), precision and recall, and compared clean and noisy conditions. For three clean synthetic examples, mean Dice was approximately 0.997 for Otsu and K-Means-2 and 0.954 for K-Means-3. For three noisy examples, the three methods obtained mean Dice between 0.992 and 0.993. The real-cell image showed broadly similar outlines across methods, but its segmentation accuracy could not be measured. These results demonstrate a reproducible unsupervised workflow, rather than state-of-the-art cell segmentation. On these simple images, increasing the number of clusters did not provide a consistent benefit.

### 1. Motivation and objectives

Microscopy image segmentation separates structures of interest from image background. For a beginner studying machine learning, intensity-based clustering provides a transparent way to understand unsupervised learning. K-Means does not require labelled examples during fitting. The project asks: **How does intensity-based K-Means compare with a traditional thresholding baseline, and does using more clusters help?** Its objectives were to implement a pipeline, demonstrate it on microscopy data, evaluate masks where ground truth is available, and document limitations without inflating the results.

### 2. Data

**Real image.** We used `skimage.data.cell()`, a quantitative phase microscopy image of one cell floating in saline. The input image was resized from 660×550 to 330×275 pixels using anti-aliasing. This sample is documented in scikit-image and is associated with Müller et al. (2018). **No real-image ground truth was provided or created.**

**Controlled artificial images.** Six grayscale microscopy-style images were generated at 256×256 pixels. Each contains four bright ellipse-like objects at approximately fixed locations, with deterministic geometric variations, blur, a nonuniform background and Gaussian noise. Three clean samples use a lower noise level and weaker intensity gradient; three noisy samples use stronger noise and gradient. The corresponding binary reference masks are saved in memory during evaluation. The images represent simplified *shapes*, not genuine cells.

### 3. Methodology

All images are normalized to pixel intensities between 0 and 1. A Gaussian smoothing filter with sigma = 1 pixel is applied before segmentation. The methods are:

1. **Otsu:** Choose a global grayscale threshold from the smoothed image and classify pixels above it as bright foreground.
2. **K-Means (K=2):** Cluster grayscale intensities into two groups; select the cluster with the highest mean intensity as foreground.
3. **K-Means (K=3):** Repeat with three intensity groups and select the brightest cluster.

K-Means minimizes the squared distance from each pixel intensity to its assigned cluster mean:

\[
J = \sum_{i=1}^{N}\left(x_i-\mu_{z_i}\right)^2,
\]

where \(x_i\) is pixel intensity, \(z_i\) is cluster assignment, and \(\mu_{z_i}\) is the assigned cluster centroid. A fixed random seed of 42 is used for initialization; five random initializations (`n_init=5`) are tried by scikit-learn. To keep runtime low, K-Means fits its centroids using every fourth pixel and then predicts labels for the entire image. The input masks are not used to train the model, nor to choose any settings.

All methods undergo **identical morphological cleanup**: remove components smaller than approximately 35 pixels, close small discontinuities with a disk of radius two, and fill small holes. This makes comparisons less dependent on isolated pixels. Exactly the same parameters are used on the real image and synthetic examples.

### 4. Evaluation

For synthetic images only, predictions were compared against the generated binary reference masks. Defining true positives (TP), false positives (FP) and false negatives (FN) as foreground pixels:

\[
\mathrm{Dice}=\frac{2TP}{2TP+FP+FN}, \quad
\mathrm{IoU}=\frac{TP}{TP+FP+FN}.
\]

\[
\mathrm{Precision}=\frac{TP}{TP+FP},\quad
\mathrm{Recall}=\frac{TP}{TP+FN}.
\]

Runtime was recorded as an approximate one-run wall-clock measurement. For the unannotated real sample, we report only the fraction of pixels classified as foreground and the predicted outlines. No Dice or IoU is computed for the real sample.

### 5. Experiments and results

**Experiment 1 — Real microscopy sample.** All three approaches identified a similar bright region corresponding to the visible cell interior. The foreground fraction was approximately **3.2%** for each method. This is a *descriptive observation*, not real-image accuracy.

**Experiment 2 — Controlled synthetic masks.** The table summarizes means over three scenes per condition. Full per-scene results and standard deviations are in `results/synthetic_pixel_metrics.csv` and `results/summary.csv`.

| Synthetic condition | Method | Mean Dice | Mean IoU |
|---|---|---:|---:|
| Clean | Otsu | 0.9974 | 0.9947 |
| Clean | K-Means-2 | 0.9973 | 0.9947 |
| Clean | K-Means-3 | 0.9536 | 0.9112 |
| Noisy | Otsu | 0.9929 | 0.9858 |
| Noisy | K-Means-2 | 0.9929 | 0.9860 |
| Noisy | K-Means-3 | 0.9921 | 0.9843 |

**Experiment 3 — Noise comparison.** All methods retained high pixel-level agreement on the specified artificial data. Using three intensity groups performed substantially worse in the clean condition, likely because selecting only the brightest group omitted pixels within some dimmer objects. This explanation is plausible from how foreground is assigned, but was not experimentally isolated or statistically tested. No superiority of K-Means over Otsu was established.

![Real microscopy outlines](figures/real_microscopy_comparison.png)

![Synthetic comparison](figures/synthetic_noisy_comparison.png)

![Dice chart](figures/synthetic_dice_comparison.png)

### 6. Interpretation

This project shows that unsupervised clustering can segment high-contrast, bright objects on a relatively uniform background. However, Otsu is simpler and gives almost identical scores on these examples. K-Means-3 does not automatically lead to better segmentation. The useful scientific result is a clear **method comparison with measured limitations**, rather than claiming an improvement where there is none.

### 7. Limitations and next steps

The quantitative metrics come exclusively from **six simplified synthetic scenes**, not expert-labeled microscopy images. The real demonstration includes only one isolated cell; its visible bright interior may not coincide exactly with the biological cell's true boundary, especially around a dark halo. Objects do not overlap or touch in the synthetic set. The method assumes foreground is brighter than background, uses only one grayscale intensity feature per pixel, and does not model spatial structure explicitly. The preprocessing and evaluation settings were fixed for educational clarity. The observations should not be extrapolated to fluorescence microscopy, pathology, photoacoustic images or clinical imaging. A meaningful extension would evaluate genuine annotated nuclei masks from BBBC039 or a similar open benchmark, without changing method settings after observing test labels.

### 8. Reproducibility

Use Python 3.10+ and install the requirements in `requirements.txt`. Run `python run.py` from the project root. The program regenerates figures and CSV metrics, records package versions in `results/environment.json`, and uses deterministic seeds for dataset synthesis and K-Means. Numerical scores are expected to be close across supported library versions; runtime measurements are machine-dependent.

### References

1. Müller, P., Schürmann, M., Girardo, S., Cojoc, G., and Guck, J. (2018). Accurate evaluation of size and refractive index for spherical objects in quantitative phase imaging. *Optics Express*, 26(8), 10729–10743. https://doi.org/10.1364/OE.26.010729
2. scikit-image. *Example data: `skimage.data.cell()`*. https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.cell
3. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*.
4. Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66. https://doi.org/10.1109/TSMC.1979.4310076
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
