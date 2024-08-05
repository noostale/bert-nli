import random
import torch
import json
import spacy
import numpy as np
import gc
import subprocess
import nltk

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import gender_guesser.detector as gender
from negate import Negator
from tqdm import tqdm
from datasets import load_dataset


class DataAugmentation:
    
    def __init__(self, dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        
        print("Loading models...")
        
        # Initialize negator model
        print("Loading negator model...")
        self.negator = Negator(fail_on_unsupported=True)
        
        # Initialize gender detector
        print("Loading gender detector model...")
        self.gd = gender.Detector()
                
        # Load synonym augmenter using words from WordNet
        print("Loading synonym augmenter...")
        self.aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.4)
        
        # Load random character deletion augmenter
        print("Loading random character deletion augmenter...")
        self.rand_rem = nac.RandomCharAug(action="delete")

        """ OTHER METHODS THAT TAKE TOO LONG
        # Load back translation augmenter
        self.back_translation_aug = naw.BackTranslationAug(
            from_model_name='Helsinki-NLP/opus-mt-en-de',
            to_model_name='Helsinki-NLP/opus-mt-de-en',
            device='cuda'
        )

        # Load abstractive summarization augmenter
        self.abstractive_aug = nas.AbstSummAug(model_path='t5-base', device = 'cuda')
        """

        

        print("Models loaded")

    """ OTHER METHODS THAT TAKE TOO LONG
    # Abstractive summarization augmentation
    def abstractive_summarization_augmenter(self, item):
        new_item = {}

        augmented_text_hypothesis = self.abstractive_aug.augment(item['hypothesis'])
        
        new_item['premise'] = item['premise']
        new_item['hypothesis'] = augmented_text_hypothesis[0]
        new_item['label'] = item['label']
        new_item['augmentation'] = 'abstractive_summarization_augmenter'

        return new_item

    
    # Back translate the hypothesis (from English to German and back to English)
    def back_translate(self, item):
        new_item = {}

        augmented_text_hypothesis = self.back_translation_aug.augment(item['hypothesis'])
        
        new_item['premise'] = item['premise']
        new_item['hypothesis'] = augmented_text_hypothesis[0]
        new_item['label'] = item['label']
        new_item['augmentation'] = 'back_translate'
        

        return new_item
    
    # Replace some words with their synonyms from WordNet using WSD from dataset
    def synonym_replacement(self, item):
        def get_generic_term(word, synset_name):
            if synset_name == 'O':  # Check for 'O' synset and ignore it
                return None
            try:
                synset = wn.synset(synset_name)
                lemmas = synset.lemmas()
                if not lemmas:
                    return None
                
                return lemmas[0].name()
            except Exception as e:
                #print(f"Error retrieving synset {synset_name}: {e}")
                return None
        
        new_item = {}
        wsd_data = item.get('wsd', {})
    
        premise_changed, hypothesis_changed = False, False
        
        new_premise = item['premise']
        new_hypothesis = item['hypothesis']
    
        for entry in wsd_data.get('premise', []):
            generic_term = get_generic_term(entry['text'], entry.get('nltkSynset'))
            
            if generic_term and generic_term != entry['text']:
                new_premise = new_premise.replace(entry['text'], generic_term)
                premise_changed = True
    
        for entry in wsd_data.get('hypothesis', []):
            generic_term = get_generic_term(entry['text'], entry.get('nltkSynset'))
                                            
            if generic_term and generic_term != entry['text']:
                new_hypothesis = new_hypothesis.replace(entry['text'], generic_term)
                hypothesis_changed = True
    
        if not premise_changed and not hypothesis_changed:
            return item
    
        new_item['premise'] = new_premise
        new_item['hypothesis'] = new_hypothesis
        new_item['label'] = item['label']
        new_item['augmentation'] = 'synonym_replacement'
        return new_item
    """


    # Repace some words in the premise with their synonyms using WordNet and library nlpaug
    def substitute_premise_words_using_wordnet(self, item):

        new_item = {}
        premise = item['premise']
        
        new_premise = self.aug.augment(premise)

        new_item['premise'] = new_premise[0]
        new_item['hypothesis'] = item['hypothesis']
        new_item['label'] = item['label']
        new_item['augmentation'] = 'substitute_premise_words_using_wordnet'
        return new_item
    
    def change_dates_in_hypothesis(self, item):
        def replace_temporal_expressions(doc):
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

            new_tokens = []
            replaced = False  # Flag to track if any replacement is done

            for token in doc:
                if token.ent_type_ == "DATE":
                    text = token.text
                    if text in days:
                        new_tokens.append(random.choice([d for d in days if d != text]))
                        replaced = True
                    elif text in months:
                        new_tokens.append(random.choice([m for m in months if m != text]))
                        replaced = True
                    elif text.isdigit() and (1 <= int(text) <= 31):
                        new_tokens.append(str(random.randint(1, 31)))
                        replaced = True
                    else:
                        new_tokens.append(token.text)
                else:
                    new_tokens.append(token.text)

            return new_tokens, replaced

        new_item = {}
        new_item['premise'] = item['premise']

        hypothesis = item['hypothesis']
        doc = nlp(hypothesis)
        new_hypothesis_tokens, replaced = replace_temporal_expressions(doc)

        if not replaced:
            return item

        new_hypothesis = " ".join(new_hypothesis_tokens)
        new_item['hypothesis'] = new_hypothesis

        # Change the label accordingly if needed
        if item['label'] == "ENTAILMENT":
            new_item['label'] = "CONTRADICTION"
        elif item['label'] == "CONTRADICTION":
            new_item['label'] = "CONTRADICTION"
        elif item['label'] == "NEUTRAL":
            new_item['label'] = "NEUTRAL"

        new_item['augmentation'] = 'change_dates_in_hypothesis'
        return new_item
    
    # Replace names found using NER in the hypothesis with random names
    def replace_names_in_hypothesis(self, item):
        def replace_person_names(tokens):
            random_names = random_names = [
                "John Doe", "Jane Smith", "Alice Johnson", "Bob Brown", "Charlie Davis",
                "Daniel Wilson", "Emily Clark", "Frank Thomas", "Grace Lewis", "Hannah Walker",
                "Isaac Young", "Jessica Hall", "Kevin Allen", "Laura Scott", "Matthew King",
                "Nathan Wright", "Olivia Baker", "Patrick Harris", "Quinn Morgan", "Rachel Cooper",
                "Steven Edwards", "Tina Mitchell", "Ulysses Martinez", "Victoria Roberts", "William Phillips",
                "Xander Turner", "Yvonne Parker", "Zachary Stewart", "Amber Hughes", "Brian Green",
                "Catherine Adams", "David Nelson", "Erica Carter", "Fiona White", "George Thompson",
                "Helen Perez", "Ian Collins", "Julia Ramirez", "Kyle Rogers", "Lily Reed",
                "Michael Campbell", "Nina Simmons", "Oscar Gray", "Paula Butler", "Quincy Foster",
                "Rebecca Gonzalez", "Samuel Henderson", "Teresa Bryant", "Umar Ramirez", "Vanessa Fisher",
                "Walter Mills", "Xena Ford", "Yusuf Kelly", "Zoey Graham", "Aaron Barnes",
                "Brenda Jenkins", "Caleb Wood", "Diana Hunt", "Ethan Palmer", "Faith Black"
            ]

    
            new_tokens = []
            for token in tokens:
                if token.ent_type_ == 'PERSON':
                    new_tokens.append(random.choice(random_names))
                else:
                    new_tokens.append(token.text)
            return new_tokens
    
        new_item = {}
        new_item['premise'] = item['premise']
    
        doc = nlp(item['hypothesis'])
        new_hypothesis_tokens = replace_person_names(doc)
    
        new_hypothesis = " ".join(new_hypothesis_tokens)
        new_item['hypothesis'] = new_hypothesis
    
        # Change the label accordingly if needed
        if item['label'] == "ENTAILMENT":
            new_item['label'] = "CONTRADICTION"
        elif item['label'] == "CONTRADICTION":
            return item
        elif item['label'] == "NEUTRAL":
            new_item['label'] = "NEUTRAL"
    
        new_item['augmentation'] = 'replace_names_in_hypothesis'
        return new_item



    # Infer the gender of the first person mentioned in the premise using NER and replace the hypothesis accordingly
    def gender_guesser(self, item):
        def get_names_genders(sentence):
            doc = nlp(sentence)
            proper_names = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ == 'PERSON']
            genders = {name: self.gd.get_gender(name.split()[0]) for name, label in proper_names}
            return proper_names, genders
        
        new_item = {}
        new_item['premise'] = item['premise']
        new_item['hypothesis'] = item['hypothesis']
        new_item['label'] = item['label']
        
        proper_names, genders = get_names_genders(item['premise'])     
        
        if len(genders) > 0:
            if genders[proper_names[0][0]] in ('male', 'female'):
                if random.random() > 0.5:
                    new_item['hypothesis'] = f"{proper_names[0][0]} is a {genders[proper_names[0][0]]}"
                    new_item['label'] = "ENTAILMENT"
                else:
                    new_item['hypothesis'] = f"{proper_names[0][0]} is not a {genders[proper_names[0][0]]}"
                    new_item['label'] = "CONTRADICTION"
        
        new_item['augmentation'] = 'gender_guesser'
        return new_item


    # shuffle the order of sentences in the hypothesis
    def shuffle_sentence_order(self, item):
        new_item = {}
        new_item['premise'] = item['premise']
        
        # Shuffle the order of sentences in the hypothesis
        hypothesis = item['hypothesis']
        sentences = hypothesis.split('. ')
        if len(sentences) > 1:
            random.shuffle(sentences)
            new_hypothesis = '. '.join(sentences)
        else:
            new_hypothesis = hypothesis
        
        new_item['hypothesis'] = new_hypothesis
        new_item['label'] = item['label']
        new_item['augmentation'] = 'shuffle_sentence_order'
        return new_item

    # Remove random characters from the premise to increase noise robustness
    def remove_random_characters(self, item):
        new_item = {}
        
        premise = item['premise']
        augmented_text = self.rand_rem.augment(premise)
        
        new_item['premise'] = augmented_text[0]
        new_item['hypothesis'] = item['hypothesis']
        new_item['label'] = item['label']
        return new_item


    # Insert random phrases in the premise and hypothesis to increase noise robustness and avoid overfitting
    def random_phrase_insertion(self, item):
        def insert_random_phrase(text, phrases, insert_prob=0.3):
            sentences = text.split('. ')
            new_sentences = []
            
            for sentence in sentences:
                new_sentences.append(sentence)
                if random.random() < insert_prob:
                    random_phrase = random.choice(phrases)
                    new_sentences.append(random_phrase)
            
            return '. '.join(new_sentences)
        
        random_phrases = [
            "In addition,", "Moreover,", "Interestingly,", "As a matter of fact,", "Consequently,",
            "Surprisingly,","Nevertheless,", "Furthermore,", "On the other hand,", "For example,",
            "In contrast,", "As such,", "To illustrate,", "For instance,", "On the contrary,",
            "In other words,", "As a result,", "Thus,", "Hence,"
        ]
        
        new_item = {}
        new_item['premise'] = insert_random_phrase(item['premise'], random_phrases)
        new_item['hypothesis'] = insert_random_phrase(item['hypothesis'], random_phrases)
        new_item['label'] = item['label']
        new_item['augmentation'] = 'random_phrase_insertion'
        return new_item

    # Remove random words from the premise (10%) to increase noise robustness
    def random_words_remover(self, item):
        def remove_random_words(text, remove_prob=0.1):
            words = text.split()
            new_words = [word for word in words if random.random() > remove_prob]
            return ' '.join(new_words)
        
        new_item = {}
        new_item['premise'] = remove_random_words(item['premise'])
        new_item['hypothesis'] = item['hypothesis']
        new_item['label'] = item['label']
        new_item['augmentation'] = 'random_words_remover'
        return new_item


    # Negate the hypothesis and change the label accordingly
    def negate_sentence(self, item):
        new_item = {}
        new_item['premise'] = item['premise']
        
        try:
            new_item['hypothesis'] = self.negator.negate_sentence(item['hypothesis'])
        except RuntimeError:
            return item
        
        if item['label'] == "ENTAILMENT":
            new_item['label'] = "CONTRADICTION"
        elif item['label'] == "CONTRADICTION":
            new_item['label'] = "ENTAILMENT"
        elif item['label'] == "NEUTRAL":
            new_item['label'] = "NEUTRAL"
        
        new_item['augmentation'] = 'negate_sentence'
        return new_item

    
    def create_augmented_samples(self, item, single_random = True):
        # Augmentation methods dictionary
        augmentations = {
            
            # Methods that change the meaning of the samples or infer new information
            'negate_sentence': self.negate_sentence,                                                       #works
            'gender_guesser': self.gender_guesser,                                                         #works
            'replace_names_in_hypothesis': self.replace_names_in_hypothesis,                               #works
            'change_dates_in_hypothesis': self.change_dates_in_hypothesis,                                 #works
            
            # Methods that modify the samples or add noise but do not change the meaning
            'substitute_premise_words_using_wordnet': self.substitute_premise_words_using_wordnet,         #works
            'shuffle_sentence_order': self.shuffle_sentence_order,                                         #works
            'random_phrase_insertion': self.random_phrase_insertion,                                       #works
            'remove_random_characters': self.remove_random_characters,                                     #works
            'random_words_remover': self.random_words_remover, #70%                                        #works
            
            #'synonym_replacement': self.synonym_replacement,                                              #works but duplicate
            #'back_translate': self.back_translate,                                                        #takes too long
            #'abstractive_summarization_augmenter': self.abstractive_summarization_augmenter,              #takes too long
        }

        if (single_random == True):
            # Select a random augmentation method
            selected_augmentations = random.sample(list(augmentations.values()), 1)
        
            # Apply the selected augmentation methods
            augmented_item = selected_augmentations[0](item)

            return [augmented_item] if augmented_item else []
        else:
            # Add each sample augmented with each method
            augmented_samples = []
            for name, method in augmentations.items():
                augmented_sample = method(item)
                #if augmented_sample:
                #    augmented_samples.append(augmented_sample)
            
            return augmented_samples


    def augment_and_save_dataset(self, file_path, target_count):
        augmented_data = []

        print(f"Creating a dataset with {target_count} samples...")
        
        # Augment data
        while len(augmented_data) < target_count:
            for item in tqdm(self.dataset):
                augmented_samples = self.create_augmented_samples(item)
                for augmented_sample in augmented_samples:
                    if augmented_sample['premise'] != item['premise'] or augmented_sample['hypothesis'] != item['hypothesis']:
                        augmented_data.append(augmented_sample)
                        
                        # Check if target count is reached
                        if len(augmented_data) >= target_count:
                            break
                # Check if target count is reached
                if len(augmented_data) >= target_count:
                    break

        
        # Save augmented data
        with open(file_path, 'w', encoding="utf-8") as f:
            for sample in augmented_data:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

    
    def augment_and_save_dataset_eq(self, file_path, target_count, val = False):
        if val == True:
            label_counts = {"CONTRADICTION": 775, "ENTAILMENT": 821, "NEUTRAL": 692}
        else:
            label_counts = {"CONTRADICTION": 12331, "ENTAILMENT": 31128, "NEUTRAL": 7627}
        
        augmented_data = []

        print(f"Creating a dataset with {target_count} samples for each label...")
        

        print("CONTRADICTION:", label_counts["CONTRADICTION"], " so i will add", target_count - label_counts["CONTRADICTION"], "samples")
        print("ENTAILMENT:", label_counts["ENTAILMENT"], " so i will add", target_count - label_counts["ENTAILMENT"], "samples")
        print("NEUTRAL:", label_counts["NEUTRAL"], " so i will add", target_count - label_counts["NEUTRAL"], "samples")
    
        # Augment data
        while (label_counts["ENTAILMENT"] < target_count or label_counts["CONTRADICTION"] < target_count or label_counts["NEUTRAL"] < target_count):
            for item in tqdm(self.dataset):
                augmented_samples = self.create_augmented_samples(item)
                for augmented_sample in augmented_samples:
                    if label_counts[augmented_sample['label']] < target_count:

                        if augmented_sample['premise'] != item['premise'] or augmented_sample['hypothesis'] != item['hypothesis']:
                            augmented_data.append(augmented_sample)
                            label_counts[augmented_sample['label']] += 1

                            
                if label_counts["ENTAILMENT"] >= target_count and label_counts["CONTRADICTION"] >= target_count and label_counts["NEUTRAL"] >= target_count:
                    break
        
        # Save augmented data
        with open(file_path, 'w', encoding="utf-8") as f:
            for sample in augmented_data:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')



def main():
    print("Loading dataset...")
    dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli").shuffle(seed=42)
    
    print("Creating data augmentor...")
    augmentor = DataAugmentation(dataset['train'])
    
    print("Augmenting and saving dataset...")
    augmentor.augment_and_save_dataset_eq("augmented_train_data.jsonl", 35000)
    print("\nAugmented data saved as 'augmented_train_data.jsonl'")
    
    print("Augmenting and saving val-set...")
    augmentor = DataAugmentation(dataset['validation'])
    augmentor.augment_and_save_dataset_eq("augmented_val_data.jsonl", 2000, val = True)
    print("\nAugmented data saved as 'augmented_val_data.jsonl'")



if __name__ == "__main__":
    
    print("DATA AUGMENTATION FOR NLI TASK - FRASCA EMANUELE 1836098\n")
    print("Setting up the environment...")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    gc.enable()

    # Load the spaCy model and nltk wordnet corpus
    try:
        nlp = spacy.load("en_core_web_sm")
    except IOError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    
    nlp = spacy.load("en_core_web_sm")
    
    nltk.download('wordnet')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Setting up finished\n")
    main()