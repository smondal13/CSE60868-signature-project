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
We will first use the [**BHSig-260**](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets/data) dataset, which contains 3000 forged and 2400 genuine signatures of Bengali and Hindi scripts, for training and validation, and note the performance. After that, we will apply the model to other language signature verification, for example, English, and note the performance. If the performance is unsatisfactory, we will add the [**CEDAR**](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets/data) Latin signatures for training and note the resulting performance improvement.

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
* **Subjects and Volume:** It is important to note that a signature dataset does not contain just one unique signature per person. To train a verification model, the network must learn the natural variance in how a single person signs their name. Therefore, each identity in our dataset provides multiple genuine specimens and has multiple skilled forgeries associated with them. The BHSig-260 dataset contains 3000 forged and 2400 genuine signatures. The CEDAR dataset contains signatures by 55 people, where each person has 24 genuine and 24 forged signatures.
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