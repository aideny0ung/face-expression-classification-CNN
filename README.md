Classify facial expression with a CNN and pytorch.

This project was for the 2026 data-dive with the Data Science club at UIUC and I had everything in a personal directory. I created this repo to show off my work.


Please look at the slides (datadive26slides.pdf) which I included in the repo. I go over my entire data-cleaning pipeline as well as CNN architecture.


A quick summary of it:

I used the FER-2013 dataset to train a model to classify images into six facial expression categories.

The dataset is very messy and I exploed methods including: low/high variance filtering (didn't work it, only found "weird" faces"; edge-desnity filtering (removed several corrupted or problematic images), duplicate removal, addressed class size imbalance , attempted to address mislabeling (finding highest loss images after a round of training)



I may or may not add more details later.
