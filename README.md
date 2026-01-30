# CSE60868-signature-project
This repo will contain the semester project of CSE60868.
The project has two parts:
1. **User‑specific verification:** Given a stored genuine signature of Student A and a test signature, is this a genuine A or a forgery of A?
2. **Generic forged vs genuine:** Given a signature image with no user ID, is it a forgery of whoever it claims to be, or genuine?

## Datasets
1. **[Divyansh Rai Dataset](https://www.kaggle.com/datasets/divyanshrai/handwritten-signatures)**
   - Handwritten dataset of 30 people
   - Contains 5 genuine and 5 forged signatures from each person

2. **[BHSig-260 Dataset](https://www.kaggle.com/datasets/ankita22053139/cedarbhsig-260)**
   - Contains 3000 forged and 2400 genuine signatures

3. **[GPDS 1-150 Dataset](https://www.kaggle.com/datasets/adeelajmal/gpds-1150)**
   *Note: Description not available in the dataset*
   - Contains 2400 genuine and 2400 forged signatures for training.
   - Contains test signatures of 125 persons

4. **[GPDS 300 Dataset](https://service.tib.eu/ldmservice/dataset/?tags=Handwritten+Signatures)**
   *Note: Could not see the dataset*
    - 300 writers, with 24 genuine and 30 skilled forgeries per writer 

5. **[TC4 Dataset Link](https://iapr-tc4.org/signature-datasets/)**
    - Contains URLs and references of ~15 datasets

6. **[AKASH GUNDU Dataset](https://www.kaggle.com/datasets/akashgundu/signature-verification-dataset)**
  *Note: Has good description*
    - This contains signature of 1372 people
    - Each person has 10 genuine and 10 forged signatures in labeled folders.

7. **[Manish Vem Dataset](https://www.kaggle.com/datasets/manishvem/signatures-dataset)**
    *Note: Has good description*
    - Contains signature of 1487 unique individuals
    - Each person  has genuine and forged signatures
   
8. **[Ishani Kathuria Dataset](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets)**
  *Note: Has good description*
    - This dataset contains signatures by 55 people written in **Latin** script.
    - Each person has 24 genuine and 24 forged signatures.
    - It also contains BHSig260-Bengali and BHSig260 Hindi datasets


## Suggested division of work
- **Person 1:** preprocessing pipeline and standard CNN classifier (baseline).
- **Person 2:** pair generation logic and Siamese model.
- **Together:** design evaluation metrics (false accept/false reject rates) and experiments on (a) Kaggle users, (b) your own signatures.

## Workflow
1. Finalize which dataset(s) to use
2. Process the data
   - Different datasets come at different resolutions and background conditions, so define one pipeline in PyTorch: grayscale → crop to ink region → resize → normalize.
4. Pretrain on Big mixed data (Can we use ResNet for that?)
   - Train the Siamese network on large multi‑writer datasets (e.g., GPDS/MCYT/BHsig, or Kaggle Akash Gundu + Signatures Dataset) to learn general “signature similarity” features.
5. Pair construction across datasets
  - Positive pairs: two genuine signatures from the same writer (within each dataset).
  - Negative pairs: genuine from writer A vs genuine from writer B, or genuine vs forged for the same writer, where available.
  - Mix pairs from all datasets in training so the network sees many writing styles.
6. 
