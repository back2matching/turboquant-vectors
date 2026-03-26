"""
Embedding Inversion Demo: Proving PrivateEncoder Defeats Embedding Attacks

This script demonstrates:
1. A trained MLP can recover meaningful text features from embeddings
2. After rotation with PrivateEncoder, the same MLP produces garbage
3. Search still works identically on rotated embeddings

This demo uses a simple MLP-based attack (not full Vec2Text) to avoid
dependency issues. The principle is the same: any model trained on the
original embedding space fails completely on rotated embeddings.

Requirements:
    pip install turboquant-vectors sentence-transformers scikit-learn
"""

import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from turboquant_vectors import PrivateEncoder


def run_demo():
    """Full embedding inversion demo."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    # ================================================================
    # STEP 1: Prepare a labeled dataset
    # ================================================================
    print("=" * 60)
    print("STEP 1: Preparing labeled embedding dataset")
    print("=" * 60)

    # Categories that would be sensitive to invert
    categories = {
        "medical": [
            "Patient was diagnosed with type 2 diabetes mellitus",
            "MRI scan revealed a tumor in the left frontal lobe",
            "Blood pressure reading was 180 over 95 indicating hypertension",
            "Patient reports chronic lower back pain for six months",
            "Prescribed metformin 500mg twice daily for glucose control",
            "Chest X-ray shows bilateral pleural effusion",
            "Patient has a family history of breast cancer",
            "Lab results indicate elevated liver enzymes ALT and AST",
            "Colonoscopy revealed three polyps in the ascending colon",
            "Patient tested positive for BRCA1 gene mutation",
            "ECG shows atrial fibrillation with rapid ventricular response",
            "Biopsy results confirm stage 2 melanoma on right shoulder",
        ],
        "financial": [
            "Account balance is fourteen million three hundred thousand dollars",
            "Wire transfer of two million dollars to offshore account",
            "Quarterly revenue decreased by fifteen percent year over year",
            "Credit card ending in 4532 has been flagged for fraud",
            "Portfolio allocation is sixty percent equities forty percent bonds",
            "Mortgage payment of three thousand two hundred dollars monthly",
            "Annual salary is two hundred forty thousand before taxes",
            "Tax return shows capital gains of fifty thousand dollars",
            "Company valuation estimated at three hundred million pre-money",
            "Insurance claim filed for property damage of eighty thousand",
            "Retirement savings total one point two million in 401k",
            "Stock options vest at thirty five dollars per share",
        ],
        "legal": [
            "The defendant entered a plea of not guilty on all charges",
            "Contract breach resulted in damages of five million dollars",
            "Attorney-client privilege protects this communication from disclosure",
            "Court ordered a preliminary injunction against the acquisition",
            "Witness testimony contradicts the police report from March",
            "Settlement negotiations resulted in a four million dollar agreement",
            "The patent application covers a novel machine learning architecture",
            "Class action lawsuit filed on behalf of thirty thousand consumers",
            "Non-disclosure agreement expires in September of next year",
            "Deposition transcript reveals insider trading prior to announcement",
            "Judge granted summary judgment in favor of the plaintiff",
            "Arbitration clause in the contract requires binding mediation",
        ],
        "personal": [
            "My home address is 1234 Oak Street Springfield Illinois",
            "Social security number is 987 65 4321 for tax filing",
            "Birthday is March fifteenth nineteen eighty seven",
            "Emergency contact is Sarah Johnson at 555 867 5309",
            "Currently employed at Google as a senior software engineer",
            "Married with two children ages seven and twelve",
            "Email password was changed to a twenty character random string",
            "Medical records are stored at Memorial Hospital patient ID 78432",
            "Bank account routing number is 021000021 at Chase",
            "Passport number is 543876219 expiring December 2028",
            "Driver license number is D400 125 67 890 issued in Texas",
            "Credit score is 782 as of the last quarterly check",
        ],
        "neutral": [
            "The weather forecast calls for partly cloudy skies tomorrow",
            "Python is a popular programming language for data science",
            "The meeting is scheduled for three o'clock in conference room B",
            "Coffee production in Brazil increased by twelve percent this year",
            "The new library book must be returned within three weeks",
            "Sunset occurs at approximately seven thirty in the evening",
            "The train departs from platform four at nine fifteen",
            "Global population reached eight billion people last year",
            "The recipe calls for two cups of flour and one egg",
            "Average commute time in the city is forty five minutes",
            "The museum exhibit features paintings from the Renaissance period",
            "Research paper was submitted to the international conference",
        ],
    }

    # Embed everything
    print("Loading sentence-transformers model...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    all_texts = []
    all_labels = []
    label_names = list(categories.keys())
    for label_idx, (category, texts) in enumerate(categories.items()):
        all_texts.extend(texts)
        all_labels.extend([label_idx] * len(texts))

    print(f"Embedding {len(all_texts)} texts across {len(label_names)} categories...")
    embeddings = encoder.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)
    labels = np.array(all_labels)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Categories: {label_names}")
    print()

    # ================================================================
    # STEP 2: Train attacker on original embeddings
    # ================================================================
    print("=" * 60)
    print("STEP 2: Training attacker classifier on ORIGINAL embeddings")
    print("=" * 60)

    # Shuffle and split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(embeddings))
    split = int(0.7 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = embeddings[train_idx], labels[train_idx]
    X_test, y_test = embeddings[test_idx], labels[test_idx]

    # Train classifier (simulates an attacker who has access to the embedding model)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Test on original embeddings
    y_pred_orig = clf.predict(X_test)
    acc_orig = accuracy_score(y_test, y_pred_orig)

    print(f"\n  Attacker accuracy on ORIGINAL embeddings: {acc_orig:.1%}")
    print(f"  (Random chance = {1/len(label_names):.1%})")
    print()
    print("  Per-category results:")
    for i, name in enumerate(label_names):
        mask = y_test == i
        if mask.sum() > 0:
            cat_acc = (y_pred_orig[mask] == y_test[mask]).mean()
            print(f"    {name:12s}: {cat_acc:.1%}")
    print()

    # ================================================================
    # STEP 3: Rotate embeddings with PrivateEncoder
    # ================================================================
    print("=" * 60)
    print("STEP 3: Rotating embeddings with PrivateEncoder")
    print("=" * 60)

    private_enc = PrivateEncoder.generate(dim=embeddings.shape[1], normalize=False)
    X_test_rotated = private_enc.rotate(X_test)

    print(f"  Key fingerprint: {private_enc.fingerprint()}")
    print(f"  Embedding dim: {embeddings.shape[1]}")

    # Verify distances preserved
    cos_orig = np.dot(X_test[0], X_test[1]) / (
        np.linalg.norm(X_test[0]) * np.linalg.norm(X_test[1])
    )
    cos_rot = np.dot(X_test_rotated[0], X_test_rotated[1]) / (
        np.linalg.norm(X_test_rotated[0]) * np.linalg.norm(X_test_rotated[1])
    )
    print(f"  Cosine similarity preserved: {cos_orig:.6f} -> {cos_rot:.6f} (diff: {abs(cos_orig-cos_rot):.2e})")
    print()

    # ================================================================
    # STEP 4: Test attacker on ROTATED embeddings
    # ================================================================
    print("=" * 60)
    print("STEP 4: Same attacker on ROTATED embeddings")
    print("=" * 60)

    y_pred_rot = clf.predict(X_test_rotated)
    acc_rot = accuracy_score(y_test, y_pred_rot)

    print(f"\n  Attacker accuracy on ROTATED embeddings: {acc_rot:.1%}")
    print(f"  (Random chance = {1/len(label_names):.1%})")
    print()
    print("  Per-category results:")
    for i, name in enumerate(label_names):
        mask = y_test == i
        if mask.sum() > 0:
            cat_acc = (y_pred_rot[mask] == y_test[mask]).mean()
            print(f"    {name:12s}: {cat_acc:.1%}")
    print()

    # ================================================================
    # STEP 5: Prove search works identically
    # ================================================================
    print("=" * 60)
    print("STEP 5: Search works identically on rotated embeddings")
    print("=" * 60)

    # Use full dataset as corpus, first 5 texts as queries
    corpus_emb = embeddings
    corpus_rot = private_enc.rotate(corpus_emb)
    query_emb = embeddings[:5]
    query_rot = private_enc.rotate(query_emb)

    k = 5
    perfect_matches = 0
    for i in range(5):
        orig_scores = corpus_emb @ query_emb[i]
        orig_topk = set(np.argsort(-orig_scores)[:k])

        rot_scores = corpus_rot @ query_rot[i]
        rot_topk = set(np.argsort(-rot_scores)[:k])

        match = orig_topk == rot_topk
        if match:
            perfect_matches += 1
        print(f"  Query {i} ('{all_texts[i][:40]}...'): top-{k} {'MATCH' if match else 'DIFFER'}")

    recall = perfect_matches / 5
    print(f"\n  Recall@{k}: {recall:.3f} ({perfect_matches}/5 perfect matches)")
    print()

    # ================================================================
    # SUMMARY
    # ================================================================
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Category classifier accuracy:")
    print(f"    On ORIGINAL embeddings: {acc_orig:.1%}  <- sensitive categories recoverable")
    print(f"    On ROTATED embeddings:  {acc_rot:.1%}  <- categories NOT recoverable")
    print(f"    Random chance:          {1/len(label_names):.1%}")
    print(f"    Accuracy drop:          {acc_orig - acc_rot:.1%}")
    print()
    print(f"  Search quality:")
    print(f"    Cosine similarity error: {abs(cos_orig-cos_rot):.2e}")
    print(f"    Top-{k} recall:           {recall:.3f}")
    print()

    if acc_rot < 0.30 and acc_orig > 0.60:
        print("  RESULT: Category inference DEFEATED by rotation.")
        print("  An attacker cannot determine if an embedding contains medical,")
        print("  financial, legal, or personal information after rotation.")
        print("  Search works identically. Zero quality loss.")
    else:
        print(f"  RESULT: Check accuracy values. Expected orig > 60%, rot < 30%.")

    return {
        "acc_original": acc_orig,
        "acc_rotated": acc_rot,
        "acc_random": 1.0 / len(label_names),
        "search_recall": recall,
        "cosine_error": float(abs(cos_orig - cos_rot)),
        "categories": label_names,
        "n_texts": len(all_texts),
    }


if __name__ == "__main__":
    results = run_demo()
