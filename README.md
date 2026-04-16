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

### Contribution

- **Shuvashish Mondal**
  - Designed and implemented the Siamese network pipeline
  - Wrote training and validation code
  - Performed hyperparameter tuning
  - Ran experiments and analyzed results

- **Kathan Desai**

- **Combined**
  - Decided on the overall project direction
  - Selected the final dataset used in the project
  - Reviewed the final repository and report
