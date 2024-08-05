# Bert Natural Language Inference

This paper describes the development and evaluation of three BERT models for Natural Language Inference (NLI) tasks using a subset of the FEVER dataset. Models are evaluated on two test sets: a simpler one and an adversarial one. Each model is trained on both the original dataset and an augmented one using various techniques. Results show that all models outperform baselines, with the best one achieving an accuracy of 76.87% on the simpler test-set and 68.25% accuracy on the adversarial test set. Findings suggest data augmentation enhances model performance on complex examples, and that choosing the right model is crucial for handling complexity.



# Scripts instructions
    
This are some simple instructions on how to run the scripts to generate adversarial data, train and test the models, compute baselines and generate data plots.

Run as first command `pip install -r requirements.txt` to install the required libraries.

**NOTE:** to train on adversarial data, first run the `1836098-augment.py` script to generate the adversarial train-set and validation-set.

```bash
python 1836098-augment.py                                            # Generate adversarial data

python 1836098-main.py train --data original    --model roberta      # Train Roberta model with original data
python 1836098-main.py test  --data original    --model roberta      # Test Roberta model with original data
python 1836098-main.py train --data adversarial --model roberta      # Train Roberta model with adversarial data
python 1836098-main.py test  --data adversarial --model roberta      # Test Roberta model with adversarial data

python 1836098-main.py train --data original    --model roberta_df   # Train Roberta_df model with original data
python 1836098-main.py test  --data original    --model roberta_df   # Test Roberta_df model with original data
python 1836098-main.py train --data adversarial --model roberta_df   # Train Roberta_df model with adversarial data
python 1836098-main.py test  --data adversarial --model roberta_df   # Test Roberta_df model with adversarial data

python 1836098-main.py train --data original    --model deberta      # Train Deberta model with original data
python 1836098-main.py test  --data original    --model deberta      # Test Deberta model with original data
python 1836098-main.py train --data adversarial --model deberta      # Train Deberta model with adversarial data
python 1836098-main.py test  --data adversarial --model deberta      # Test Deberta model with adversarial data

python 1836098-main.py baselines                                     # Compute baselines

python 1836098-plots.py                                              # Generate data plots
```


