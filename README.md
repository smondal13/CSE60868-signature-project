# CSE60868 Signature Verification Project
**Team members:** Kathan Desai, Shuvashish Mondal

## Project Objective
This project explores the application of deep learning to **offline handwritten signature verification**. 

### The Core Comparison
1.  **Task A: Writer-Dependent Classification**
We will first implement a standard Convolutional Neural Network (CNN) trained as a multi-class classifier. The goal is to answer the question: *"Who wrote this signature?"* This establishes a performance baseline and verifies that our feature extractor can successfully learn static signature attributes like stroke width, loop geometry, and aspect ratio. However, this approach is limited to a "closed-set" scenario; it cannot handle new users without full retraining.

2.  **Task B: Writer-Independent Verification**
The core of our project is to build a One-Shot Learning system using a Siamese Network. This addresses the real-world requirement where a banking or security system must verify a new user without retraining the entire model. The network will answer the question: *"Do these two signatures belong to the same person?"* By learning a similarity metric (Euclidean distance) rather than specific class labels, the model can generalize to completely unseen writers—including the project team members.
---

## Selected Strategy & Data Split
We will first use the **BHSig-260** dataset, which contains Indic language signatures, for training and validation, and note the performance. After that, we will apply the model to other language signature verification, for example, English, and note the performance. If the performance is unsatisfactory, we will add the CEDAR Latin signatures for training and note the resulting performance improvement.

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
## Advanced Scope (Optional)
If time permits, we plan to experiment with transfer learning using foundational models such as Meta's [DINOv3](https://ai.meta.com/research/dinov3/). We will test if replacing our custom CNN backbone with DINOv3 features improves verification accuracy on the "One-Shot" task.

## Datasets Available Online
*Reference list of all datasets collected for this project.*

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

*Disclaimer: Portions of this proposal were brainstormed and refined with the assistance of AI tools (Gemini and ChatGPT), acting as a "senior engineer" persona to critique architectural decisions and workflow.*
