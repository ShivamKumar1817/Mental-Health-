# Mental Health Text Classifier with Data Augmentation

Ever wish you had more training data? This script has a trick up its sleeve — it takes your existing mental health dataset and cleverly multiplies it before training, helping the model learn better without you having to collect a single extra label.

Under the hood, it's a straightforward but effective combo: augmented text data fed into a TF-IDF + Logistic Regression pipeline. Nothing fancy, but it works surprisingly well.

---

## Getting Started

You'll need Python 3.8 or newer, then just grab the dependencies:

```bash
pip install pandas numpy scikit-learn joblib
```

That's it — no GPU, no Docker, no fuss.

---

## What Data Does It Need?

Drop a file called `Combined Data.csv` into the same folder as the script. It just needs two columns:

| Column      | What goes here                             |
|-------------|--------------------------------------------|
| `statement` | The raw text (what someone wrote/said)     |
| `status`    | The label (e.g. anxiety, depression, etc.) |

Any rows with missing values are quietly skipped, so don't worry about cleaning those up manually.

---

## How It Works

### Step 1 — Augmenting the Training Data

Rather than training on the raw dataset as-is, the script first expands the training split to about **3× its original size** by generating two extra variants of every sample. Each variant is created by randomly applying one of these tweaks:

- **Word Swap** — shuffles ~10% of word pairs around in the sentence
- **Word Deletion** — randomly drops ~15% of words
- **Keep as-is** — sometimes no change is the right move

Crucially, this only happens to the **training data** — the test set is left completely untouched, so your evaluation scores stay honest.

### Step 2 — Training the Model

Once the data is ready, a two-step pipeline kicks off:

- **TF-IDF** turns the text into numeric features, capturing both single words and two-word phrases (up to 25,000 features total)
- **Logistic Regression** does the actual classification, tuned with `C=2.0` and given up to 2,000 iterations to converge

It'll use all your CPU cores to speed things up, but fair warning — with 3× the data, training does take noticeably longer than you might expect.

### Step 3 — Evaluating & Saving

After training, the model is tested on that untouched 20% test split and prints out a full classification report so you can see exactly how it's performing per class. The final model is then saved as:

```
mental_health_model_large.pkl
```

Keep that file — it's what you'll load for predictions later.

---

## Running It

```bash
python augment_and_train.py
```

Just make sure `Combined Data.csv` is sitting in the same directory first.

---

## What You'll See

```
Loading original data...
Original dataset size: 10000
Splitting data...
Augmenting training data to synthetically increase the dataset size...
Training dataset size AFTER augmentation: 24000 statements!
Test dataset size remains strictly: 2000 statements
Building and training pipeline...
Training expanded model (this will take longer now because data is 3x larger)...
Evaluating new improved model...

Classification Report (Augmented Dataset):
              precision    recall  f1-score   support
...
Saving large model to mental_health_model_large.pkl...
Complete!
```

---

## Using the Saved Model

Once training is done, loading the model for predictions is dead simple:

```python
import joblib

pipeline = joblib.load('mental_health_model_large.pkl')
predictions = pipeline.predict(["I've been feeling really hopeless lately."])
print(predictions)
```

No need to re-run the whole training process — just load and go.
