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
   - Contains 3000 forged and 2400 genuine signatures.
   - Includes Bengali and Hindi scripts.
## Selected Strategy & Data Split
We will first use the **BHSig-260** dataset, which contains 3000 forged and 2400 genuine signatures of Bengali and Hindi scripts, for training and validation, and note the performance. After that, we will apply the model to other language signature verification, for example, English, and note the performance. If the performance is unsatisfactory, we will add the CEDAR Latin signatures for training and note the resulting performance improvement.

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

*Wordcount: 779 using [wordcounter](https://wordcounter.net/)*

*Disclaimer: Portions of this proposal were brainstormed and refined with the assistance of AI tools (Gemini and ChatGPT), acting as a "senior engineer" persona to critique architectural decisions and workflow.*
