# CSE60868 Signature Verification Project
Disclaimer: Written with the help of Gemini and ChatGPT

## Project Objective
This project compares two deep learning approaches for signature verification to demonstrate the difference between traditional classification and the "One-Shot Learning" capability required for real-world biometrics.

### The Core Comparison
1.  **Task A: Writer-Dependent Classification (The Baseline)**
    * *Question:* "Who wrote this?" (Closed-Set)
    * *Model:* Standard CNN Classifier.
    * *Limit:* Can only recognize users present in the training set.
    * *Goal:* Establish a performance baseline for feature extraction.

2.  **Task B: Writer-Independent Verification (The Advanced Goal)**
    * *Question:* "Do these two signatures match?" (Open-Set)
    * *Model:* Siamese Network with Metric Learning.
    * *Advantage:* Can verify **completely new users** (e.g., the project team) without retraining.
    * *Goal:* Verify signatures for users the model has never seen before.

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

## Selected Strategy & Data Split
To ensure the model succeeds on the team's English signatures (Demo Phase) despite the training data being primarily Indic scripts, we will use a mixed split strategy.

* **Training Set:** **All BHSig-260** + **CEDAR (Users 1–30)**
    * *Reason:* We must include some English/Latin signatures in training so the model learns features like loops and ascenders, which are less common in Indic scripts.
* **Validation Set:** **CEDAR (Users 31–45)**
    * *Reason:* Used to tune hyperparameters and save the best model.
* **Test/Demo Set:** **CEDAR (Users 46–55)** + **Team Members' Custom Signatures**
    * *Reason:* Completely unseen users to prove "One-Shot" capability.

### Supplemental Datasets
* **Akash Gundu Dataset** (Dataset #6) will be used if the model requires more pre-training volume.

---


## Work Division

### **Team Member 1: The Baseline & Infrastructure**
* **Task 1: Preprocessing Pipeline:**
    * Invert images (Black background, White ink).
    * Resize to fixed target (e.g., $155 \times 220$).
    * *Deliverable:* A `process_image()` function used by both team members.
* **Task 2: The CNN Classifier (Baseline):**
    * Build a standard Multi-Class CNN to classify the 260 BHSig users.
    * *Goal:* Achieve high accuracy (>90%) to prove the Custom CNN architecture is sound.

### **Team Member 2: The Siamese Network**
* **Task 1: Dynamic Pair Sampling (Crucial):**
    * *Engineering Constraint:* Do **not** generate all possible pairs (millions). Generate pairs **on-the-fly** inside the `__getitem__` method.
    * *Ratio Strategy:*
        * **Positive Pairs (50%):** Genuine A + Genuine A.
        * **Negative Pairs (50%):** Prioritize **Hard Negatives** (Genuine A + Forged A) over Easy Negatives (Genuine A + Genuine B) to force the model to learn subtle differences.
* **Task 2: Siamese Architecture:**
    * Reuse the CNN from Team Member 1 as the backbone (remove the final classification layer).
    * Implement Euclidean Distance logic and Contrastive Loss.

### **Joint Tasks**
* **Final Report:** Compare the Baseline (Classifies known users) vs. Siamese (Verifies new users).
* **Demo:** Run the team's signatures through the Siamese network to see if it detects the teammate's forgeries.

---

## Technical Workflow

### Phase 1: The Architecture (Custom CNN)
To demonstrate "built from scratch" capability and avoid over-engineering, we will use this custom architecture:

1.  **Conv2D** (1 $\to$ 32 filters) + ReLU + MaxPool
2.  **Conv2D** (32 $\to$ 64 filters) + ReLU + MaxPool
3.  **Conv2D** (64 $\to$ 128 filters) + ReLU + MaxPool
4.  **Flatten** $\to$ **Dense** (256) $\to$ **Dense** (128 - Embedding Vector)

### Phase 2: Training (Metric Learning)
* **Loss Function:** Contrastive Loss.
    * *Logic:* If pairs are the same, minimize distance. If different, maximize distance until it hits a margin $m$.
* **Hyperparameters:**
    * Margin ($m$): 1.0
    * Learning Rate: $1e-3$ (Adam Optimizer)
    * Batch Size: 32 or 64 (depending on RAM).

### Phase 3: Evaluation Metrics
We will report standard biometric metrics:
1.  **FAR (False Acceptance Rate):** How often do we accidentally trust a forgery?
2.  **FRR (False Rejection Rate):** How often do we accidentally reject a genuine signature?
3.  **ROC (Receiver Operating Characteristic) Curve:** Plot FAR vs. FRR to find the optimal threshold $\tau$.

## Advanced tasks (optional):
* Use a foundational model, such as models from meta [DINO 3](https://ai.meta.com/research/dinov3/), and use transfer learning to see how our work performs when we add those on top of DINO3.
