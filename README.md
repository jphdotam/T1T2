# T1T2

1. Label your data

2. Export the training data using those labels

3. Train the network

---

For the labelling system (shown below) run `./_run/label.py`

![Labelling system](https://raw.githubusercontent.com/jphdotam/T1T2/master/labelling/ui/labelui2.gif)

---

Exporting the data is simply done by running `./_run/export_png_masks.py`

---

To train an HRNet, run `./_run/train.py`, after making sure you've exported your labels wiut

Early results from:

TRAIN: 102 examples over 20 patients

TEST:  37 examples over 6 patients

Visualisations every 10 epochs are visible in the ./output/vis directory (Columns T1/T2/truth/prediction).

![Epoch 720](https://github.com/jphdotam/T1T2/raw/master/output/vis/002/720.png)
