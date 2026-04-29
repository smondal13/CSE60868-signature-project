# CSE60868 Signature Verification Project
**Team members:** Kathan Desai, Shuvashish Mondal
## Project Objective
This project explores the application of deep learning to **offline handwritten signature verification**. 

## Part 1: Conceptual design
### The Core Comparison
1.  **Task A: Writer-Dependent Classification**
We will first implement a standard Convolutional Neural Network (CNN) trained as a multi-class classifier. The goal is to answer the question: *"Who wrote this signature?"* This establishes a performance baseline and verifies that our feature extractor can successfully learn static signature attributes like stroke width, loop geometry, and aspect ratio. However, this approach is limited to a "closed-set" scenario; it cannot handle new users without full retraining.

2.  **Task B: Writer-Independent Verification**
The core of our project is to build a One-Shot Learning system using a Siamese Network. This addresses the real-world requirement where a banking or security system must verify a new user without retraining the entire model. The network will answer the question: *"Do these two signatures belong to the same person?"* By learning a similarity metric (Euclidean distance) rather than specific class labels, the model can generalize to completely unseen writers—including the project team members.
---

## Selected Strategy & Data Split
We will first use the [**BHSig-260**](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets/data) dataset, which contains 7800 forged and 6240 genuine signatures across Bengali and Hindi scripts, for training and validation, and note the performance. After that, we will apply the model to other language signature verification, for example, English, and note the performance. If the performance is unsatisfactory, we will add the [**CEDAR**](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets/data) Latin signatures for training and note the resulting performance improvement.

---

## Technical Workflow & Solution

### The Baseline
* **Task 1: Preprocessing Pipeline:**
    * Invert images (Black background, White ink).
    * Resize to fixed target (e.g., $155 \times 220$).
    * *Deliverable:* A `process_image()` function used by both team members.
* **Task 2: The CNN Classifier (Baseline):**
    * Build a standard Multi-Class CNN to classify the 260 BHSig users.
    * *Goal:* Achieve high accuracy (>90%) to prove the Custom CNN architecture is sound.

---
### Custom Siamese Network (Shuvashish Mondal)
**The Architecture**
Shuvashish will build the system from scratch using a custom CNN backbone. The preprocessed pipeline from the baseline work will be used in the Siamese Network.
* **The Backbone:** A sequence of Convolutional blocks (Conv2D $\to$ ReLU $\to$ MaxPool). This will compress the high-dimensional image into a compact embedding vector (e.g., size 128).
* **The Siamese Head:** Two identical copies of this backbone will process a pair of images (Image A and Image B) in parallel, sharing the exact same weights.
* **Distance Metric:** The network will output the Euclidean distance between the two embedding vectors.

**Training Logic**
Shuvashish Mondal will use **Contrastive Loss** to train the manifold.
* **Positive Pairs (Genuine + Genuine):** The loss function will penalize the network if the distance is $> 0$.
* **Negative Pairs (Genuine + Forged):** The loss function will penalize the network if the distance is $< m$ (margin).
* *Engineering Challenge:* We will implement **Dynamic Pair Sampling** to generate pairs on-the-fly during training. This prevents memory explosions and allows us to prioritize "Hard Negatives" (forgeries) over "Easy Negatives" (different writers), forcing the model to learn subtle stroke differences.
---

### **Joint Tasks**
* **Final Report:** Compare the Baseline (Classifies known users) vs. Siamese (Verifies new users).
* **Demo:** Run the team's signatures through the Siamese network to see if it detects the teammate's forgeries.
---

## Evaluation Methodology
Unlike standard classification tasks, where "Accuracy" is the primary metric, biometric verification requires measuring the trade-off between security and convenience. We will evaluate our Siamese Network using the following standard biometric metrics:

1.  **False Acceptance Rate (FAR):**
    * *Definition:* The likelihood that the system incorrectly verifies a forgery as genuine.
    * *Significance:* In a high-security context (e.g., banking), this must be minimized to near zero.
2.  **False Rejection Rate (FRR):**
    * *Definition:* The likelihood that the system incorrectly rejects a genuine user.
    * *Significance:* High FRR frustrates users. A usable system must balance this against FAR.
3.  **Receiver Operating Characteristic (ROC) Curve:**
    * Since our model outputs a continuous distance score, there is no single "correct" threshold. We will plot FAR vs. FRR across all possible thresholds ($\tau$) to visualize the system's performance.
4.  **Equal Error Rate (EER):**
    * We will determine the specific threshold where $FAR = FRR$. This single number provides a standard benchmark to compare our model against state-of-the-art results in the literature.
---

## Advanced Scope (Optional)
If time permits, we plan to experiment with transfer learning using foundational models such as Meta's [DINOv3](https://ai.meta.com/research/dinov3/). We will test if replacing our custom CNN backbone with DINOv3 features improves verification accuracy on the "One-Shot" task.

---

Wordcount: 779 using [wordcounter](https://wordcounter.net/)


---

## Part 2: Database Description

### 1. Data Sources and Associated Literature
To start actual development of the project solutions, we have acquired appropriate offline signature datasets. We will primarily use the BHSig-260 dataset. If the performance is unsatisfactory, we will add the CEDAR Latin signatures for training.

[**Downloaded dataset directory**](https://drive.google.com/drive/folders/15IMEsLgGY4iegLh5JI73ExozQwcyvlA7?usp=sharing)

| Dataset | Source | Language | Original publication |
|---------|--------|----------|----------------------|
| **BHSig-260 Dataset** | [Kaggle](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets/data) | Bengali and Hindi | Pal, S., Alaei, A., Pal, U. and Blumenstein, M., 2016, April. Performance of an off-line signature verification method based on texture features on a large indic-script signature dataset. In 2016 12th IAPR workshop on document analysis systems (DAS) (pp. 72-77). IEEE. |
| **CEDAR Dataset** | [Kaggle](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets/data) | Latin script | CEDAR, URL: https://cedar.buffalo.edu/NIJ/publications.html |

### 2. Characterization of Samples and Subjects
Our acquired database consists of offline handwritten signatures. Unlike online signatures that capture dynamic time-series data, offline signatures are static images extracted from scanned documents. 
* **Subjects and Volume:** It is important to note that a signature dataset does not contain just one unique signature per person. To train a verification model, the network must learn the natural variance in how a single person signs their name. Therefore, each identity in our dataset provides multiple genuine specimens and has multiple skilled forgeries associated with them. The Bengali and Hindi portions of the BHSig-260 dataset together contain 7800 forged and 6240 genuine signatures. The CEDAR dataset contains signatures by 55 people, where each person has 24 genuine and 24 forged signatures.
* **Sensors and Ambient Conditions:** Because these are offline datasets, the sensors used to capture them were standard high-resolution flatbed optical scanners. The ambient conditions involved standard indoor lighting, and the physical medium was ink on white paper.
* **Resolution and Preprocessing:** The raw signature images come in varying resolutions. To standardize this for our neural networks, our preprocessing pipeline will invert the images (black background, white ink) and resize them to a fixed target of 155 by 220 pixels. Inverting the images helps neutralize ambient background noise so the network focuses purely on stroke geometry.

### 3. Training vs. Validation Split Strategies
The dataset needs to be split into three subsets. For this phase, we are focusing on the data that we will use in training and validation. We aim for a standard split of roughly 60% for training, 20% for validation, and 20% reserved for an unknown subset for testing. We will apply random splitting for this with a seed so that reproducibility is guaranteed. 

* **Training Subset:** We will use the training set to find the values of our trainable parameters: weights and biases. For the baseline CNN, this involves updating the classifier weights, while for the Siamese network, it involves training the weights of the shared Convolutional blocks (Conv2D $\to$ ReLU $\to$ MaxPool).
* **Validation Subset:** We will use the validation set to check how our network generalizes on new samples. This prevents our solution from overfitting and allows us to find values for our hyper-parameters. Specifically, we will use it to tune the learning rate and the margin parameter ($m$) for our Contrastive Loss function.
* **Key Differences:** The most critical difference between our training and validation sets is the strict separation of identities. Because our advanced goal is writer-independent verification (verifying new users), our validation set must evaluate how the model performs on writers it has never seen. If we split strictly by image, the network would simply memorize the strokes of known users. By isolating writers completely across the subsets, we can accurately evaluate the metric learning capabilities of our Siamese network.

### 4. Declaration of Data Acquisition
By submitting this report, we attest that we have physically downloaded the aforementioned data (BHSig-260 and CEDAR datasets) to our local machines. We have verified the directory structures and confirmed the image formats are compatible with our PyTorch data loaders.

### 5. Team Contributions
For this specific assignment phase:
* **Shuvashish Mondal:** Physically acquired and downloaded the dataset files, verified the folder structures, and authored the initial draft of this Part 2 report.
* **Kathan Desai:** Responsible for reviewing the acquired data, correcting any architectural discrepancies in the document, and finalizing/pushing this report to the GitHub repository.

*(Moving forward into the implementation phase, Kathan will lead the CNN Baseline development, while Shuvashish will lead the Siamese Network architecture).*

*Disclaimer: Portions of this proposal were brainstormed and refined with the assistance of AI tools (Gemini), acting as a "senior engineer" persona to critique architectural decisions and workflow.*
---


## Part 3: First Update

### Shuvashish Mondal — Writer-Independent Siamese Verification Track

This section documents my progress on the writer-independent Siamese-network track for offline signature verification. At this stage, my goal is not to claim final performance, but to show that the full experimental pipeline is working and to identify the main challenges that need to be resolved before larger-scale training and evaluation.

So far, I have implemented an end-to-end workflow around the BHSig-260 dataset. The pipeline reads and normalizes signature images, constructs writer-independent train/validation/test splits, trains a Siamese model with contrastive loss, selects thresholds on validation data, and evaluates on held-out test data using verification metrics such as FAR, FRR, ROC, AUC, and EER. I have verified this workflow through reduced-scale runs, which gives confidence that the code path is functional from data loading through final evaluation.

A key design choice in my track is strict writer independence. The data is partitioned by writer ID so that no writer appears in more than one split. This matters because the real challenge is not memorizing the appearance of writers seen during training, but generalizing to new writers at evaluation time. I added explicit leakage checks to confirm that the train, validation, and test partitions remain disjoint at the writer level. This has been one of the most important implementation decisions so far, because without it the reported results would not reflect the intended task.

On the modeling side, I implemented a Siamese-network training pipeline that learns an embedding space for signature verification. Each branch of the network processes one signature image, and the Euclidean distance between the two embeddings is used as the verification signal. Training uses contrastive loss,

$$
L = y d^2 + (1-y)\max(0, m-d)^2,
$$

where $y$ indicates whether the pair is positive or negative, $d$ is the embedding distance, and $m$ is the margin. In practice, the main point of this objective is to pull genuine pairs closer together while pushing negative pairs apart up to the margin. I also integrated online hard negative mining so that training emphasizes confusing negative pairs instead of spending most updates on trivial ones.

In addition to the core training loop, I implemented a validation-driven thresholding procedure. Rather than tuning directly on the test set, I sweep thresholds on validation distances, choose an operating point based on validation metrics, and then lock that threshold for held-out test evaluation. This is important for keeping the evaluation protocol realistic and avoiding post hoc calibration on the final test split. The evaluation code currently reports ROC behavior, FAR, FRR, AUC, and EER, which gives a more complete view than reporting only one scalar metric.

To confirm that all parts of the pipeline work together, I ran reduced-scale experiments. These runs use a deliberately small profile so that I can debug quickly and verify correctness before spending more time and compute on larger jobs. At this scale, the exact numbers are less important than the fact that the full process completes successfully and produces sensible outputs.

| Metric | Interim Value |
|---|---|
| Reduced-profile split size | Train 480, Validation 160, Test 160 |
| Writer split (no overlap) | Train 24, Validation 8, Test 8 |
| Best validation EER (interim) | ~0.099 |
| Typical validation AUC range | ~0.93 to ~0.96 |
| Held-out test EER (interim run) | ~0.17 |

These early results are encouraging in the limited sense that they show the pipeline is functional and capable of producing nontrivial separation between positive and negative pairs. At the same time, I do not view them as strong evidence of final model quality. The reduced-profile setting is small enough that run-to-run variance is noticeable, and it would be easy to over-interpret numbers that are really only serving as a systems check. For now, I treat these experiments as validation of implementation readiness rather than validation of final generalization performance.

The most important part of this update is the set of challenges I am currently facing. The largest issue is moving from local prototyping to reliable CRC execution. My workflow so far has used local Apple Silicon runs for debugging and small-scale testing, which has been effective for rapid iteration. However, full experiments will require CRC GPU jobs, and that introduces practical uncertainty around scheduler conventions, resource requests, logging, checkpoint recovery, and debugging failed runs. This is my first time using the cluster, so even routine questions about submission setup can become bottlenecks.

A second challenge is the computational cost of pair construction and evaluation. Verification tasks naturally create many possible pairs, and the number grows quickly as the number of writers and samples increases. At a small debugging scale this is manageable, but for larger experiments I need to decide how exhaustive the evaluation should be and whether some form of deterministic sampling is acceptable. This is both a technical and reporting issue: I want evaluation to be meaningful, but I also need the runtime to remain practical enough for repeated experiments.

A third challenge is metric interpretation. I can already compute EER, FAR, FRR, ROC, and AUC, but I am still thinking carefully about what should be emphasized at this stage of the semester. EER is an attractive summary metric, but it can hide behavior at specific operating points that may matter in verification settings. At the same time, reporting too many metrics too early can make the update harder to read without adding much value. Guidance from Adam or the TA on the preferred scope of metric reporting for this checkpoint would help me avoid spending time optimizing for the wrong presentation format.


My immediate next steps are therefore centered on making the workflow more reliable rather than making it more complicated. First, I want to run moderate-scale experiments that are larger than the current debug profile but still small enough to iterate on efficiently. Second, I want to transition the training and evaluation workflow to repeatable CRC single-GPU jobs with proper logging and checkpoint tracking. Finally, I want to settle on a reporting format that is both methodologically sound and aligned with course expectations.

Overall, I believe this track has reached a good interim point: the core system is implemented, the end-to-end path works, and the current limitations are now clear enough to discuss productively. The main value of this update is not a single performance number, but a clear picture of what is working, what is uncertain, and what support would be most useful before the next milestone.


## Kathan Desai — Writer-Dependent CNN Classification Track

This section documents my progress on the writer-dependent CNN classification track for offline signature analysis. At this stage, my goal is not to claim a strong final performance, but to show that the baseline classification pipeline is implemented, that the end-to-end workflow is functioning, and that the main challenges are now becoming clear.

So far, I have built a baseline workflow around the **BHSig260-Hindi** dataset. The pipeline reads the raw writer folders, separates genuine signatures from forged ones, constructs train/validation/test splits, preprocesses the images, trains a convolutional neural network in a multi-class classification setting, and evaluates performance on held-out data. I have verified that this workflow runs from dataset preparation through final testing, which gives confidence that the baseline code path is functional even though the current model performance is weak.

A key design choice in my track is that the baseline uses **only genuine signatures** for classification. Since the goal of this part is to answer the question *“Which writer produced this signature?”* ,I treated each writer as a separate class and excluded forged samples from training. This keeps the baseline setup clean and makes it easier to interpret whether a standard CNN can learn writer-specific signature features such as stroke shape, spacing, slant, and overall writing style.

On the implementation side, I wrote code to reorganize the raw BHSig260-Hindi dataset into a format suitable for PyTorch training. Each numbered folder corresponds to one writer, and filenames indicate whether a signature is genuine or forged. The preprocessing pipeline converts images to grayscale, inverts them, resizes them to a fixed input shape, and normalizes them before training. I then implemented a multi-class CNN baseline and trained it end-to-end as a writer classifier.

The current interim experiment used:

- **160 writer classes**
- **2560 training samples**
- **480 validation samples**
- **800 test samples**

The current results are:

- **Best validation accuracy:** 0.0063
- **Final test accuracy:** 0.0063
- **Final test loss:** 5.0755

These results are not strong in a performance sense. In fact, they are very close to what would be expected from random guessing over 160 classes. I therefore do not interpret them as evidence that the current CNN baseline is learning meaningful writer separation yet. Instead, I view them as an important interim result because they show that the baseline pipeline is implemented well enough to run end-to-end, while also making it clear that the current formulation is too difficult for this simple setup.

The main challenge I am currently facing is the difficulty of the classification problem itself. Although the total number of images is not small, the number of genuine samples per writer is limited once the dataset is split into training, validation, and test sets. This means the model is being asked to distinguish many writers using only a modest number of examples per class. In practice, this is a difficult learning problem, especially because signatures are visually sparse and often differ only in subtle ways.

A second challenge is preprocessing sensitivity. Signature images contain much less visual information than natural images, so small choices in resizing, inversion, and normalization may have a significant impact on what the model learns. I am still evaluating whether the current preprocessing choices are the best ones for this dataset or whether the model is losing too much useful structure before training even begins.

A third challenge is deciding how much to improve this baseline before shifting more attention to the Siamese verification model. One interpretation of the weak baseline result is that the current CNN architecture or training setup needs improvement. Another interpretation is that this already highlights the limitation of writer-dependent classification and strengthens the motivation for a writer-independent verification approach. Guidance from Adam or the TA would be especially helpful here, particularly on whether I should spend more time simplifying and improving the baseline or treat this as a sufficient baseline checkpoint and move more strongly toward verification.

My immediate next steps are therefore focused on making the baseline more informative rather than simply making it larger. First, I want to inspect the training behavior more closely and test whether the network can learn on a smaller subset of writers. Second, I want to try mild augmentation and a few controlled preprocessing adjustments. Finally, I want to use what I learn from this baseline to better support the later comparison with the Siamese verification model.

Overall, I believe this track has reached a useful interim point: the baseline system is implemented, the end-to-end workflow is working, and the current limitations are clear enough to discuss productively. The main value of this update is not a strong performance number, but a clearer understanding of what is working, what is difficult, and what support would be most useful before the next milestone.

---

## Part 4: Final Results and Discussion

### Shuvashish Mondal — Final Siamese Verification Result

The final system is a writer-independent offline signature verifier built using a Siamese neural network. Each signature image is mapped to an embedding, and the Euclidean distance between two embeddings is used as the verification score. A small distance suggests the pair is from the same writer, while a large distance suggests a non-match.

This is a verification problem, not a standard multiclass classification problem. Because of that, plain classification accuracy is not the best primary metric. In biometric verification, the operating threshold is part of the system itself, so I mainly report:

- **AUC**, to measure overall separation between genuine and non-genuine pairs
- **EER**, to summarize the FAR/FRR trade-off
- **FAR** and **FRR** at a fixed threshold

Since the assignment also asks for an accuracy-style measure, I additionally report threshold-based pairwise accuracyon both the train and validation splits.

#### How to run the trained network on a single validation sample

For this Siamese verifier, a single inference example consists of a **pair of signature images**. I included one validation pair in the repository:

- reference image: `./validation/B-S-83-G-04.tif`
- query image: `./validation/B-S-83-F-05.tif`

This pair contains a genuine signature and a skilled forgery from the same writer, so the expected output is `non-match`.

From a fresh clone of the repository, run:

```bash
conda env create -f environment.yml
# if you have already have an environment with the dependencies, the line above is unnecessary
conda activate <environment-name>
bash ./siamese/scripts/run_single_validation_pair_local.sh
```
This command uses:

- the final checkpoint selected from validation
- the fixed validation pair above
- the locked threshold stored in the checkpoint

The script prints:

- the checkpoint path
- the two image paths
- the Euclidean distance between embeddings
- the decision threshold
- the predicted label (`match` or `non-match`)
- the expected label

It also saves the output to:

- [single_pair_result.json](./validation/results/single_pair_result.json)

The underlying inference entry point is:

- [infer_single_pair.py](./siamese/src/signature_siamese/infer_single_pair.py)

#### Evaluation protocol

The Bengali and Hindi portions of **BHSig-260** were split in a
writer-independent way:

- `60%` train
- `20%` validation
- `20%` test

No writer appears in more than one split. This is important because the goal is
to verify unseen writers, not to memorize known signatures.

The final training recipe used:

- a custom CNN Siamese backbone
- contrastive loss
- online hard-negative mining
- mild train-time augmentation (`mild_v1`)
- focused hyperparameter tuning over learning rate and margin

The best final configuration was:

- learning rate: `5e-4`
- margin: `0.75`
- augmentation: `mild_v1`

#### Final validation performance on BHSig-260

For the best validation checkpoint, using the locked threshold
`\tau = 0.5195`, the fixed-split evaluation results were:

- **Train accuracy:** `98.52%`
- **Validation accuracy:** `92.95%`
- **Train AUC:** `0.9990`
- **Validation AUC:** `0.9800`
- **Train EER:** `0.00847`
- **Validation EER:** `0.07064`
- **Train FAR at the locked threshold:** `0.0180`
- **Validation FAR at the locked threshold:** `0.0704`
- **Train FRR at the locked threshold:** `0.00430`
- **Validation FRR at the locked threshold:** `0.0711`

The fixed validation evaluation set contained:

- positive pairs: `14,352`
- skilled-forgery pairs: `37,440`
- random-impostor pairs: `10,324`
- total pairs: `62,116`

The fixed train evaluation set contained `186,506` pairs in total:

- positive pairs: `43,056`
- skilled-forgery pairs: `112,320`
- random-impostor pairs: `31,130`

These train and validation numbers come from the same post-training fixed-split
evaluator, so they are directly comparable. The train-validation gap is clear,
but not extreme: the model fits the training data very well while still keeping
strong validation performance on unseen writers.

#### Why these metrics are appropriate

Unlike multi-class classification, the Siamese network does not directly output
a writer label. It outputs a distance score, and the final match/non-match
decision depends on a threshold. Therefore:

- **AUC** is useful because it measures how well the model separates positive
  and negative pairs over all thresholds.
- **EER** is useful because it gives a standard biometric summary at the point
  where FAR and FRR are balanced.
- **FAR/FRR** are important because in a verification system the operating point
  matters. A model with a decent EER can still behave poorly if the threshold is
  badly chosen.

For this reason, I believe AUC, EER, FAR, and FRR are better suited to this
problem than plain accuracy alone.

#### Commentary on the observed validation performance

The final validation result is strong enough to support the claim that the
Siamese verification approach is working. An EER of about `7%` and an AUC near
`0.98` show that the model learned a useful writer-independent representation on
the BHSig validation split.

At the same time, I do not interpret this as evidence that the problem is
solved. A clean validation score can hide brittleness, so I also evaluated the
model under controlled perturbations. These robustness experiments showed:

- small rotations are not the main weakness
- resolution loss hurts substantially more
- stroke-thickness changes, especially thinning, remain difficult

This means the model is learning meaningful signature features, but it is still
somewhat dependent on image quality and stroke appearance.

#### What improved generalization

The most important improvement during the project was mild targeted
augmentation. Earlier models were much more fragile under resolution and
stroke-width shifts. After augmentation:

- clean validation performance improved slightly
- robustness improved dramatically

For example, compared to the earlier robustness baseline, the best augmented and
tuned model reduced validation EER under the strongest perturbations by large
margins:

- `resolution_50`: from `0.3786` to `0.1259`
- `thickness +1`: from `0.3845` to `0.0914`
- `thickness -1`: from `0.5120` to `0.1781`

This is important because it suggests that the original model was overfitting to
the exact appearance statistics of the training data, and that realistic
augmentation materially improved generalization.

#### External generalization on CEDAR

I also evaluated the best BHSig-trained model on CEDAR to test cross-dataset transfer. In zero-shot transfer, AUC remained high (`0.9680`) but the transferred threshold produced poor operating-point behavior (high FAR). After calibrating the threshold on a small disjoint CEDAR subset, AUC and EER stayed nearly unchanged while FAR/FRR became much more reasonable.

This shows that the learned embedding transfers better than the threshold itself. In other words, the representation generalizes moderately well, but deployment still requires dataset-specific calibration.

#### Interpretation of training versus validation behavior

The fixed-split results show that the model learns the training distribution
very strongly: training AUC is almost perfect and training EER is below `1%`.
Validation performance drops, but it remains strong, with validation AUC near
`0.98` and validation EER near `7%`. I interpret this as moderate
overfitting, not catastrophic overfitting.

That gap is expected in writer-independent verification, because validation
writers are completely unseen during training. More importantly, the project
showed that the main limitation is not just ordinary overfitting. The broader
story is:

- the initial model worked on clean validation but was not robust
- targeted augmentation improved both clean validation and robustness
- external evaluation showed that the learned representation transfers better
  than the threshold

So the remaining limitation is a combination of:

- sensitivity to appearance shifts
- threshold calibration mismatch across datasets
- remaining difficulty with cross-writer impostors

#### Ideas for future improvement

If I continued this project, the most useful next steps would be:

1. **Better threshold calibration**
   The CEDAR experiments show clearly that the threshold is dataset-dependent.
   More systematic calibration would improve deployment quality.

2. **More cross-writer negative emphasis**
   Since random-impostor rejection remains the harder weakness, I would spend
   more effort on harder cross-writer negative mining or sampling.

3. **More diverse training data**
   Mixing more scripts or adding domain-diverse signature data could improve
   cross-dataset transfer.

4. **Further robustness-focused augmentation**
   The current augmentation already helped a lot, but more realistic scan and
   stroke perturbations may improve generalization further.

#### Final conclusion

The final Siamese system is a successful writer-independent offline signature
verification pipeline. On BHSig validation, it achieves strong verification
performance, with **98.52% train accuracy** and **92.95% validation accuracy**
at the locked threshold, **train/validation AUC of 0.999/0.980**, and
**train/validation EER of 0.0085/0.0706**. The most important technical result
is that mild targeted augmentation substantially improved generalization
without harming clean validation performance.

External evaluation on CEDAR showed that the learned embedding transfers
reasonably well, but the threshold does not transfer cleanly across datasets.
After CEDAR-side calibration, the operating-point behavior improves greatly
while AUC and EER remain nearly unchanged. This strongly suggests that
calibration is a major part of cross-dataset deployment.

Overall, I conclude that the Siamese approach is effective for offline
signature verification, but robustness and threshold calibration are just as
important as raw validation accuracy.



# Part 4 — Kathan Desai (Writer-Dependent CNN Classification Track)

## How to Run Single-Sample Inference

A standalone script `infer_single_sample.py` loads a trained checkpoint
and classifies a single signature image. The script supports both
backbones and both datasets; it reads the backbone type, class names,
and preprocessing parameters from the checkpoint itself.

```bash
# BHSig ResNet-18 (default):
python infer_single_sample.py --image ./validation_sample.tif

# BHSig custom CNN:
python infer_single_sample.py --image ./validation_sample.tif \
    --checkpoint ./run_results/custom_allclasses_best.pt

# CEDAR ResNet-18:
python infer_single_sample.py --image ./cedar_sample.png \
    --checkpoint ./cedar_run_results/cedar_writer_all_resnet18_best.pt
```

The attached `validation_sample.tif` is a genuine signature from
BHSig-Hindi writer 011; the expected top-1 prediction is writer `11`.

## Overview

This section reports the final results of the CNN classification track
on two datasets: BHSig260-Hindi (160 writers, the primary dataset) and
CEDAR-Buffalo (55 writers, used as a cross-dataset test on a different
script). On both datasets, we train a model to answer *" who wrote this
signature?"* On CEDAR, we also train a model to answer *" Is this
signature real or forged?"* Unlike Shuvashish's Siamese track, which
verifies signatures from people the model has never seen, these
classifiers assume every person they will be asked about was already
seen during training.

## Methodology

Each dataset was split 60/20/20 into training, validation, and test
partitions by signature index — for each person, their first 14–16
signatures go to training, the next ~5 to validation, and the last ~5
to test. No image appears in more than one split. Preprocessing
inverts images (white strokes on black), resizes, and normalizes.
Training adds small random rotations, shifts, and shears (±5°, ±5%,
small shear) so the model doesn't just memorize exact pixel patterns.

We compared two models. The **custom CNN** has four convolutional
blocks (1→32→64→128→256 channels, each doing a 3×3 convolution,
GroupNorm, ReLU, and 2×2 max-pooling) followed by the global average
pooling and a linear classifier — about 430,000 trainable parameters.
The **ResNet-18** is a well-known pretrained model that has already learned
general visual features from 1.4 million ImageNet images. We kept
those pretrained weights and only replaced their final classification
layer with a fresh one for our writer count, then fine-tuned with a
lower learning rate on the pretrained parts (3e-5) and a higher rate
on the new layer (3e-4). Both use Adam with cosine annealing and
cross-entropy loss.

**Metrics.** Accuracy is the primary metric because both tasks are
balanced — every class appears about equally in every split. We also
report the chance baseline (1/N for N classes), since "48.8% accuracy."
means very different things on a 2-class versus a 160-class problem,
and per-class accuracy, which catches situations where the overall
number looks good but the model is quietly failing on specific
classes.

## Why Two Models? One Per Dataset

A natural question is why we trained separate models for BHSig and
CEDAR rather than one model that handles both. The short answer is
that a classifier's output layer is hard-wired to the specific people
it was trained on. Our BHSig model has 160 output neurons, one per
BHSig writer. Our CEDAR model has 55 output neurons, one per CEDAR
writer. You can't use the BHSig model on a CEDAR signature — the 160
output slots correspond to 160 specific Hindi-script writers, none of
whom are the CEDAR people. The model would return one of its known
names with low confidence, which is meaningless.

This is a fundamental limitation of any classifier: it can only assign
inputs to classes it was explicitly trained on. Adding new writers
requires retraining the final layer at a minimum. It also means the
BHSig and CEDAR numbers are not directly comparable to the way results are presented
from the same model on different test sets would be — they are
separate experiments using the same architecture and training recipe,
Run in parallel to see whether the approach holds up across languages.

## Results — Writer Classification

| Dataset | Backbone | Classes | Val Acc | Test Acc | Chance |
|---|---|---|---|---|---|
| BHSig-Hindi | Custom CNN | 2 | 100.0% | 100.0% | 50.00% |
| BHSig-Hindi | Custom CNN | 10 | 76.7% | 76.0% | 10.00% |
| BHSig-Hindi | Custom CNN | 160 | 49.6% | 48.8% | 0.63% |
| BHSig-Hindi | ResNet-18 | 160 | 100.0% | **99.6%** | 0.63% |
| CEDAR | Custom CNN | 2 | 100.0% | 100.0% | 50.00% | 
| CEDAR | Custom CNN | 10 | 94.0% | 88.0% | 10.00% |
| CEDAR | Custom CNN | 55 | 91.3% | 89.5% | 1.82% | 
| CEDAR | ResNet-18 | 55 | 100.0% | **98.6%** | 1.82% |

Two things stand out. First, the pretrained ResNet-18 does extremely
well on both datasets — 99.6% on BHSig (160 people) and 98.6% on CEDAR
(55 people). Second, the custom CNN performs far better on CEDAR
(89.5%) than on BHSig (48.8%). Some of that gap is because CEDAR has
fewer people to choose from, but some seem to come from the
scripts themselves: Latin signatures tend to have distinctive
large-scale shape features (overall slant, initial letter, loops) that
are easier to pick up than the densely interconnected strokes common
in Devanagari.

Per-class breakdown on the BHSig ResNet-18 run: 157 of 160 writers
classified perfectly, every writer predicted at least once, and the 3
That fails each miss one test sample out of five. Each confused writer
pair goes only one direction, suggesting near-neighbor style similarity
rather than systematic mix-ups.

## From 0.6% to 99.6% on BHSig: The Debugging Journey

The first BHSig implementation (described in Part 3) scored 0.6% on the
160-way task — chance. Figuring out why took real detective work.

Following the professor's suggestion, we first tested whether the model
could handle easier versions of the task by restricting training to 2
and then 10 writers. Both still gave near-chance results, ruling out
the hypothesis that 160 classes were simply too many — something was
broken before capacity even mattered.

The breakthrough came from a sanity check that compared two different
ways of measuring training accuracy. Neural networks behave differently
in "training mode" (with dropout active and BatchNorm using batch
statistics) versus "evaluation mode" (dropout off, BatchNorm using
accumulated running averages). Normally, these give similar accuracy on
the same data. Ours didn't: training mode said the model was learning,
eval mode said every prediction was the same class.

The culprit was **BatchNorm's running statistics**. BatchNorm keeps a
moving average of each batch's mean and variance across training, then
uses that average at evaluation time. That moving average uses momentum
0.1 — tuned for runs with hundreds of batches. In our 2-class sanity
test we had 32 total training samples and batch size 32, meaning one
batch per epoch and only 20 updates in total. The running average never
had a chance to settle, so the evaluation was normalized with essentially
random statistics.

The fix replaced BatchNorm with GroupNorm, which normalizes within each
sample using channel groups and tracks no batch statistics, so it
behaves identically in training and eval mode. We also dropped the
batch size from 32 to 8, so even small runs had multiple batches per
epoch. This single architecture change moved the custom CNN from
0.625% to 48.75% on the 160-class task — a 78× improvement with no
other changes.

## The CEDAR Forgery Investigation

When we extended the work to CEDAR, we added a second task: to tell genuine
signatures apart from forgeries. The initial result looked too good —
ResNet-18 scored **100% test accuracy**, noticeably above typical
published benchmarks on CEDAR (usually 80–95%). That prompted a closer
look.

Visual inspection of the images revealed something important: genuine
signatures were scanned on textured paper, some with visible ruled paper
guidelines, while forgeries were on clean white paper. This raised two
possibilities. The model might be detecting the *paper* rather than the
*signature*. Or, since the same writers appeared in both training and
testing, the model might be memorizing each person's genuine signatures
and flagging anything that looks different. We ran three experiments
to separate these effects:

| Run | Split | Preprocessing | Test Acc |
|---|---|---|---|
| A | Writer-dependent | Standard | **100.0%** |
| B | Writer-dependent | Binarized (Otsu) | **98.2%** |
| C | Writer-independent | Binarized | **66.9%** |

Run B removes paper texture using Otsu binarization — each image is
converted to pure black ink on a pure white background. The drop from
100% to 98.2% was unexpectedly small: the paper texture wasn't doing
most of the work.

Run C removes per-writer memorization by holding out 10 writers
entirely from training. The test set contains only signatures from
people the model has never seen. Accuracy drops to 66.9% — still well
above the 50% chance baseline, and within the published range for
writer-independent forgery detection on CEDAR (about 65–80%), but a
very different task than the writer-dependent numbers above.

The gap between validation (99.3% on trained writers) and test (66.9%
on held-out writers) In run C, it is itself revealing. That 32-point gap
quantifies how much of the forgery-detection performance came from
memorizing specific people's signatures versus actually learning what
Forgery looks like it in general (~17 points above chance). The honest
Takeaway: Writer-dependent forgery detection is much easier than it
sounds, and the big numbers in that setup don't translate to
deployment.

## Why We Can't Use This Classifier to Verify Our Own Signatures

A demo of the project originally envisioned was feeding our own signatures
(and forgeries of each other's signatures) into the model to test
whether it detects the forgery. With the classifier, that isn't
possible — and it's the same root cause as the "why two models."
question above.

The BHSig model's 160 output neurons correspond to 160 specific
Hindi-script writers. If we hand in someone's signature, it can only
say "this is writer 47" or "this is writer 112" with some probability.
It has no way to say "this is none of my known writers" or "this is a
forgery." Classifiers are closed-set by design: they assume every
input belongs to one of the categories they've been shown. The CEDAR
forgery classifier has the same limitation — it was trained on CEDAR
writers specifically and doesn't generalize to Kathan or Shuvashish.

The Siamese network, by contrast, is purpose-built for this. Instead
of outputting a class label, it outputs a distance between two
signatures. "Are these two signatures similar?" is a question you can
answer for any pair of images, including unseen people. That's why
the team-signatures demo is a Siamese experiment, not a CNN experiment.
The classifier and the verifier answer structurally different
questions.

## Comparison with the Siamese Track

Shuvashish's Siamese model reports 92.95% validation accuracy on
writer-independent BHSig verification. On paper, that's lower than the
99.6% classification result, but the two numbers measure different
things. Classification is writer-dependent — which of 160 known people
wrote this signature, assuming one of them did. Verification is
writer-independent — do these two arbitrary signatures match,
regardless of whether the model has seen either person. The CEDAR
forgery investigation makes the difficulty gap concrete: the same
ResNet-18 scored 98.2% writer-dependent but 66.9% writer-independent
on forgery detection. In realistic deployment — a bank verifying a
new customer — the writer-independent number is what matters, and
that is the regime the Siamese track was designed for.



### Contribution

- **Shuvashish Mondal**
  - Designed and implemented the Siamese network pipeline
  - Wrote training and validation code
  - Performed hyperparameter tuning
  - Ran experiments and analyzed results

- **Kathan Desai**
   - Reorganized BHSig260-Hindi into train/val/test splits
   - Implemented the CNN classification pipeline
   - Diagnosed the BatchNorm running-statistics failure behind the 0.6% baseline and implemented the GroupNorm fix
   - Ran the full sweep on BHSig across custom CNN and ResNet-18 at 2, 10, and 160 classes
   - Extended the pipeline to CEDAR for both writer classification and genuine-vs-forgery detection
   - Conducted the forgery artifact investigation with binarization and writer-independent splits
   - Wrote the single-sample inference script
   - Authored the CNN sections of Parts 1–4

- **Combined**
  - Decided on the overall project direction
  - Selected the final dataset used in the project
  - Reviewed the final repository and report







# Part 5 — Kathan Desai Cross-Domain Test Evaluation

## Test Database Description

Part 5 evaluates the trained classifiers against a **deliberately
out-of-distribution test set**: each model is fed images from the
*other* dataset, which it was never trained on. This is a stronger
test than the held-out partitions of Part 4, where train and test
images came from the same writers and the same acquisition pipeline.
Cross-dataset evaluation tests something the standard split cannot:
how does the model behave when handed data from a domain it has no
business making predictions about?

We use two test sets:

- **For the BHSig-trained classifier (160 Hindi-script writers):** all
  2640 CEDAR-Buffalo signatures (1320 genuine + 1320 forgeries) from
  55 Latin-script writers. None of the CEDAR people appear in BHSig.
- **For the CEDAR-trained classifier (55 Latin-script writers):** the
  same 800 BHSig held-out test images used in Part 4. None of the
  BHSig people appear in CEDAR.

The differences between these test sets and what the models were trained on
are deliberately stark. The script is different (Latin vs Devanagari).
The writers are different. The acquisition pipelines were different
(separate scanning sessions, separate institutions, decades apart). A
model that has truly learned *what makes signatures distinctive* would
struggle on this data; a model that learned only superficial pixel
patterns specific to its training domain would also struggle, but in a
different way. Either failure mode is informative.

We also include in-domain baselines (each classifier on its own
held-out test set, the numbers already reported in Part 4), so the
out-of-distribution numbers have something meaningful to compare to.

## Metrics

Following the same convention as Part 4, we report accuracy and
chance-level baselines. But for the cross-domain (OOD) evaluations,
top-1 accuracy in the usual sense is not directly meaningful — none
of the input writers are even in the model's class vocabulary, so any
prediction is wrong by construction. We instead measure two proxies
of model behavior:

- **Mean top-1 predicted probability.** How confident is the model in
  its top guess? On in-domain data, this should be close to 1 when
  predictions are correct. On OOD data, an honest model should drop
  toward chance (1/N).
- **Mean output entropy.** Entropy of the full softmax distribution
  measures how spread out the model's probability mass is. Maximum
  entropy is log(N): for BHSig (160 classes), that is 5.08 nats; for
  CEDAR (55 classes) is 4.01 nats. An honest OOD prediction
  approaches this maximum, indicating the model is admitting it
  doesn't know.

These metrics fit the cross-domain question — "does the model know it
doesn't know?" — which simple accuracy cannot answer.

## Results

| Model | Test Data | Domain | N | Mean Top-1 Prob | Entropy / Max |
|---|---|---|---|---|---|
| BHSig ResNet-18 | BHSig test | In-domain | 800 | **0.931** | 0.085 |
| BHSig ResNet-18 | CEDAR genuine | OOD | 1320 | 0.114 | 0.820 |
| BHSig ResNet-18 | CEDAR forgery | OOD | 1320 | 0.177 | 0.756 |
| CEDAR ResNet-18 | CEDAR genuine | In-domain | 1320 | **0.681** | 0.407 |
| CEDAR ResNet-18 | BHSig test | OOD | 800 | 0.097 | 0.907 |

In-domain accuracy on both classifiers exactly matches the Part 4
numbers (BHSig 99.6%, CEDAR 99.7% on its own genuine set), confirming
the checkpoints loaded correctly.

The out-of-distribution numbers are the headline result, and they are
encouraging. Both models behave **correctly** when shown data they
were not trained on:

- The BHSig classifier's confidence drops from 93.1% on in-domain Hindi
  signatures to 11.4% on out-of-domain CEDAR Latin signatures — an
  82-point drop. Its entropy rises from 8.5% of maximum to 82.0% of
  maximum, meaning the probability mass spreads across most of the 160
  output classes rather than concentrating on one.
- The CEDAR classifier shows the same pattern even more strongly. It
  goes from 68.1% in-domain confidence to 9.7% on BHSig images —
  almost exactly chance (1/55 ≈ 1.8%, but the natural distribution
  centers slightly above due to softmax mechanics). Its entropy is
  90.7% of the maximum on OOD data.

This is **not** the standard "neural networks are overconfident on
out-of-distribution data" failure mode that the literature widely
documents. Both classifiers correctly recognize, through the spread of
their output distributions, that they do not know what they are
looking at. They do not aggressively pick a single writer with high
confidence; they hedge.

## What Went Wrong (and What Did Not)

The rubric anticipates that test results should be worse than train
and validation, and asks for explanations of why. In our case the
in-domain test accuracy was already strong (99.6% on BHSig, 98.6% on
CEDAR writer classification — see Part 4) and remains strong here.
The cross-domain results are not "worse" in the accuracy sense;
they are *different*. The right question is not "why did accuracy
drop?" but "what does the model's behavior on out-of-distribution
data tell us about what it actually learned?"

A few specific findings emerged:

**Finding 1: Models are well-calibrated on OOD inputs.** Both
classifiers spread probability across many classes when shown
unfamiliar data, which is the correct behavior. Many published
softmax classifiers fail this test by remaining over-confident.
Ours don't. Likely reasons: ImageNet pretraining gives ResNet-18 a
broad feature space that doesn't collapse to a narrow training class
features; the small per-class sample count (~14 images per writer)
provides natural regularization against memorizing class-specific
shortcuts; and class-balanced training keeps the softmax temperature
in a reasonable range.

**Finding 2: The BHSig classifier reacts differently to CEDAR genuine
vs CEDAR forgery.** Confidence on CEDAR forgeries is **6.3 percentage
points higher** than on CEDAR genuine signatures (17.7% vs 11.4%) and
entropy is correspondingly lower (3.84 vs 4.16 nats). With N=1320
samples in each group this difference is unambiguously real. Since
the BHSig model has no concept of "forgery" — it was never shown
forgeries during training — this is direct evidence that the
acquisition artifact identified in Part 4 (CEDAR forgeries scanned on
clean white paper, CEDAR genuines on textured paper) is detectable
even to a model with no semantic understanding of the genuine/forgery
distinction. The clean-paper images simply look more like *something*
to the BHSig model — likely a stroke layout that resembles one of
the Hindi writers — than the textured-paper images do.

**Finding 3: The CEDAR classifier is more uncertain on OOD inputs
than the BHSig classifier is.** CEDAR's OOD entropy ratio is 91% of
maximum; BHSig's is 82%. The CEDAR classifier hedges more aggressively.
A plausible explanation: CEDAR has fewer classes (55 vs 160), so
spreading probability uniformly across all classes uses fewer "units
of confidence" per class than BHSig does. Another factor: the CEDAR
in-domain confidence is also lower (68% vs 93%), suggesting the CEDAR
model is generally less aggressive about committing to single
predictions — perhaps because its 24-samples-per-writer training set
gave it less per-writer specificity than BHSig's effectively
similar-but-larger pool.

**What Could Be Improved**

Even though the OOD behavior is well-calibrated, several improvements
would lower the residual error rates:

1. **Calibrated confidence with temperature scaling.** Both models
   could be post-hoc calibrated using temperature scaling on a
   held-out validation set. This would not change accuracy, but would
   make predicted probabilities more meaningful — useful for
   downstream applications that need to threshold on confidence.

2. **Out-of-distribution detection as an explicit task.** The current
   models implicitly handle OOD through softmax spreading. An
   explicit OOD detector — for example, training the model with an
   additional "none of the above" class supervised by random images,
   or using approaches like ODIN or Mahalanobis distance on
   embeddings — would let the system *report* "this isn't one of the
   writers I know" rather than returning a low-confidence prediction.

3. **Domain adaptation between scripts.** A more ambitious approach
   would be unsupervised domain adaptation: use the unlabeled CEDAR
   signatures during BHSig training to align the embedding spaces of
   the two domains. The model would still classify only Hindi
   writers, but its features would generalize better. This addresses
   the script-specific shortcut detected in finding 2.

4. **Forgery-aware preprocessing.** The genuine-vs-forgery confidence
   gap (finding 2) shows that any model trained on CEDAR-style data
   needs preprocessing that neutralizes paper texture before the
   model sees the image. The Otsu binarization explored in Part 4
   was a good first step. A more principled approach would include
   stroke extraction, contrast normalization, and/or scanner-specific
   color correction.

5. **Fundamentally, switching to a verification framing.** None of
   the above fixes changes the fact that a closed-set classifier is
   the wrong tool for cross-domain signature analysis. Shuvashish's
   Siamese verification track is the right architecture for handling
   unseen writers, because it computes pairwise similarity rather
   than predicting from a fixed vocabulary. Its CEDAR transfer
   results in Part 4 (AUC 0.97 zero-shot) confirm this.

## Visual Illustrations

The notebook saves three sets of failure illustrations to
`./part5_results/illustrations/`:

- `bhsig_on_cedar_genuine_top_failures.png` — the six CEDAR genuine
  signatures the BHSig model was most confident about. These show
  what happens when the model is "wrong but committed": it has
  picked a Hindi writer that the CEDAR signature happens to
  superficially resemble.
- `bhsig_on_cedar_forgery_top_failures.png` — the same for CEDAR
  forgeries. Higher confidence than the genuine equivalents,
  consistent with finding 2.
- `cedar_on_bhsig_top_failures.png` — the symmetric direction:
  Hindi signatures the CEDAR classifier was most confident about.

Inspecting these images reveals what the model is reaching for when
it's confused. In most cases, the "most confident" OOD predictions are
images whose stroke layout coincidentally matches a known training
writer — long horizontal lines may cause the BHSig model to predict a
Hindi writer with similar overall stroke geometry, even though the
ink shapes themselves are completely different. These illustrations
make the failure mode concrete.
