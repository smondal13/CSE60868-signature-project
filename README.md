# CSE60868 Signature Verification Project

## Project Objective
This project implements a **One-Shot Learning** system for offline handwritten signature verification using Siamese Networks. The goal is to build a system that can verify signatures for users it has never seen before (Writer-Independent).

### Core Tasks
1.  **User-Specific Verification (Writer-Dependent):**
    * Train and test on a closed set of users (e.g., specific Kaggle IDs).
    * *Question:* "Is this image consistent with User A's training samples?"
2.  **Writer-Independent Verification (Siamese One-Shot):**
    * Train on a large background dataset (e.g., BHSig260 + CEDAR).
    * Test on completely unseen users (e.g., Our own signatures).
    * *Question:* "Given Reference Image X and Query Image Y, are they from the same writer?"

---

## Datasets Available Online
*Reference list of all datasets evaluated for this project.*

1. **[Divyansh Rai Dataset](https://www.kaggle.com/datasets/divyanshrai/handwritten-signatures)**
   - Handwritten dataset of 30 people.
   - Contains 5 genuine and 5 forged signatures from each person.

2. **[BHSig-260 Dataset](https://www.kaggle.com/datasets/ankita22053139/cedarbhsig-260)**
   - Contains 3000 forged and 2400 genuine signatures.
   - Includes Bengali and Hindi scripts.

3. **[GPDS 1-150 Dataset](https://www.kaggle.com/datasets/adeelajmal/gpds-1150)**
   - Contains 2400 genuine and 2400 forged signatures for training.
   - Contains test signatures of 125 persons.

4. **[GPDS 300 Dataset](https://service.tib.eu/ldmservice/dataset/?tags=Handwritten+Signatures)**
   - 300 writers, with 24 genuine and 30 skilled forgeries per writer.

5. **[TC4 Dataset Link](https://iapr-tc4.org/signature-datasets/)**
   - Repository containing URLs and references of ~15 datasets.

6. **[AKASH GUNDU Dataset](https://www.kaggle.com/datasets/akashgundu/signature-verification-dataset)**
   - Signatures of 1372 people.
   - Each person has 10 genuine and 10 forged signatures in labeled folders.

7. **[Manish Vem Dataset](https://www.kaggle.com/datasets/manishvem/signatures-dataset)**
   - Signatures of 1487 unique individuals.
   - Each person has genuine and forged signatures.

8. **[CEDAR (Ishani Kathuria Dataset)](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets)**
   - Signatures by 55 people written in **Latin** script.
   - Each person has 24 genuine and 24 forged signatures.
   - Also contains BHSig260-Bengali and BHSig260 Hindi datasets.

---

## Selected Dataset Strategy
We have selected the following high-quality academic benchmarks from the list above to ensure valid baselines and cross-lingual generalization.

### Primary Datasets (Training & Validation)
1.  **BHSig-260 Dataset** (Dataset #2)
    * *Role:* Main training set to learn general geometric stability and stroke dynamics (High volume).
2.  **CEDAR / Ishani Kathuria** (Dataset #8)
    * *Role:* **Crucial for English generalization.** We will split this dataset to ensure the model learns Latin script features before testing on our own signatures.

### Supplemental Datasets
* **Akash Gundu Dataset** (Dataset #6) will be used if the model requires more pre-training volume.

---

## Work Division

### **Team Member 1: Infrastructure & Baseline**
* **Data Pipeline:** Implement the `PyTorch Dataset` class.
    * *Coordination Point:* Define the `__getitem__` return format early (e.g., returns `(image1, image2, label)`).
    * *Splitting:* Ensure strictly disjoint user splits (see "Technical Workflow" below).
* **Preprocessing:**
    * Binarization (Otsu’s Thresholding).
    * Inverted Normalization (Background=0, Ink=1).
    * Fixed size resizing (e.g., $155 \times 220$).
* **Baseline Model:** Standard CNN classifier (Multi-class classification) to establish a performance floor on the "Writer-Dependent" task.

### **Team Member 2: Siamese Architecture & Evaluation**
* **Pair Generation Logic:**
    * *Anchor-Positive:* Genuine A + Genuine A.
    * *Anchor-Negative (Easy):* Genuine A + Genuine B.
    * *Anchor-Negative (Hard):* Genuine A + Forged A.
* **Model Architecture:**
    * Implement Siamese Network with shared weights.
    * Backbone: Custom CNN or ResNet-18 (modified for 1-channel input).
* **Loss Function:** Implement **Contrastive Loss**.

### **Joint Tasks**
* **Evaluation:**
    * Compute **FAR** (False Acceptance Rate) and **FRR** (False Rejection Rate).
    * Plot the **ROC Curve** to find the optimal distance threshold $\tau$.
* **Real-World Test:**
    * Collect 5 genuine and 5 forged signatures from team members.
    * Run them through the model (without training on them) to prove One-Shot capability.

---

## Technical Workflow

### 1. Data Ingestion & Splitting Strategy
To ensure the model works on English signatures (Team Test) despite being trained heavily on Hindi/Bengali (BHSig260), we will use the following split:

* **Training Set:** All BHSig260 (260 users) + First 40 users of CEDAR.
    * *Goal:* Learn stroke features from both Indic and Latin scripts.
* **Validation Set:** Users 41–50 of CEDAR.
* **Test Set (Unseen):** Last 5 users of CEDAR + **Team Members' Custom Signatures**.

### 2. Preprocessing Pipeline
Since we are building "from scratch," we need robust features.
1.  **Grayscale & Invert:** Signatures are white-on-black in tensor form ($0$ is background).
2.  **Centering:** Compute the center of mass of the pixels and center the signature in the canvas.
3.  **Augmentation (Crucial):**
    * Random Affine (Rotation $\pm 10^{\circ}$, Shear $\pm 5^{\circ}$).
    * *Do not* flip signatures (signatures are not symmetric).

### 3. Model Architecture Strategy
* **Backbone:** ResNet-18 (lightweight) or Custom 4-Layer CNN.
    * *Constraint:* Input channels = 1.
    * *Output:* Embedding vector of size 128 or 256.
* **Metric:** Euclidean Distance between embeddings.

### 4. Training Loop
* **Loss:** Contrastive Loss.
    $$L = (1-Y) \frac{1}{2} D^2 + (Y) \frac{1}{2} \{ \max(0, m - D) \}^2$$
* **Optimizer:** Adam ($lr=1E-3$ or $1E-4$).
* **Batching:** Ensure every batch contains a mix of Positive and Hard Negative pairs.
