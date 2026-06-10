#!/usr/bin/env python3
"""
POKEMON TARGET IDENTIFICATION NLP MODULE
=======================================

Complete NLP training pipeline for identifying Pokemon targets from military-style prompts.
This module generates synthetic training data, trains a transformer-based classifier,
and saves the trained model for deployment.

Features:
- Synthetic data generation with military-style prompts
- Enhanced transformer architecture with attention pooling
- Comprehensive training with evaluation metrics
- Rule-based fallback system for inference
- Model saving in standard formats (SafeTensors, config, tokenizer)

Version: 4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    Trainer, TrainingArguments, TrainerCallback
)
import numpy as np
import pandas as pd
import json
import re
import random
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====================== POKEMON KNOWLEDGE BASE ======================
POKEMON_KNOWLEDGE = {
    "Pikachu": {
        "names": ["pikachu", "pika", "electric mouse", "pikachu pokemon"],
        "types": ["electric"],
        "colors": ["yellow", "golden", "bright yellow"],
        "descriptors": [
            "electric rat", "tiny thunder beast", "yellow mouse", 
            "rodent of sparks", "lightning rodent", "spark mouse",
            "mouse", "rodent", "lightning", "thunder", "spark", 
            "electric", "cheek", "quick", "agile", "small", "cute",
            "electrical", "thunderbolt", "lightning bolt", "electric type"
        ],
        "physical_attributes": [
            "yellow fur", "red cheeks", "pointed ears", "lightning bolt tail",
            "small size", "bipedal", "electric pouches", "black-tipped ears",
            "brown stripes on back", "round cheeks"
        ],
        "weaknesses": ["ground"],
        "habitats": ["forests", "power plants", "urban areas"]
    },
    "Charizard": {
        "names": ["charizard", "char", "charizard pokemon"],
        "types": ["fire", "flying"],
        "colors": ["orange", "red", "blue", "cream", "tan"],
        "descriptors": [
            "flame dragon", "winged inferno", "scaled fire titan",
            "orange lizard", "fire dragon", "aerial predator",
            "dragon", "fire", "flame", "wing", "fiery", "aerial",
            "powerful", "lizard", "inferno", "wings", "flying", "large",
            "fire type", "flying type", "flamethrower", "fire breath"
        ],
        "physical_attributes": [
            "orange scales", "large wings", "flame tail", "dragon-like",
            "bipedal", "fire breathing", "powerful build", "creamy underside",
            "two horns", "long neck", "sharp claws"
        ],
        "weaknesses": ["water", "rock", "electric"],
        "habitats": ["mountains", "volcanoes", "rocky areas"]
    },
    "Bulbasaur": {
        "names": ["bulbasaur", "bulba", "bulbasaur pokemon"],
        "types": ["grass", "poison"],
        "colors": ["green", "blue", "teal"],
        "descriptors": [
            "plant reptile", "vine beast", "green seedling",
            "sprout toad", "seed pokemon", "grass creature",
            "seed", "plant", "bulb", "vine", "herbal", "toxic",
            "toad", "sprout", "reptile", "grass", "leaf", "nature",
            "grass type", "poison type", "vine whip", "solar beam"
        ],
        "physical_attributes": [
            "green skin", "bulb on back", "four legs", "plant features",
            "vine whips", "spotted pattern", "quadruped", "red eyes",
            "pointed ears", "bulb with plant", "blue-green skin"
        ],
        "weaknesses": ["fire", "flying", "ice", "psychic"],
        "habitats": ["forests", "grasslands", "gardens"]
    },
    "Mewtwo": {
        "names": ["mewtwo", "mew two", "mewtwo pokemon"],
        "types": ["psychic"],
        "colors": ["purple", "pink", "gray", "silver", "white"],
        "descriptors": [
            "genetic experiment", "psychic clone", "telekinetic predator",
            "synthetic mind weapon", "artificial pokemon", "lab creation",
            "clone", "psychic", "powerful", "intelligent", "experiment",
            "artificial", "legendary", "telepathic", "mental",
            "psychic type", "genetically engineered", "psychokinetic",
            "mind power", "psystrike"
        ],
        "physical_attributes": [
            "purple skin", "large head", "three fingers", "long tail",
            "humanoid", "psychic aura", "feline features", "tube on back of neck",
            "purple abdomen", "pointed ears", "white underside"
        ],
        "weaknesses": ["bug", "ghost", "dark"],
        "habitats": ["caves", "laboratories", "remote areas"]
    }
}

# ====================== SYNTHETIC DATA GENERATOR ======================
class SyntheticDataGenerator:
    """Generates synthetic military-style prompts for training"""
    
    def __init__(self):
        logger.info("Initializing synthetic data generator")
        
        self.military_headers = [
            "HQ REPORT", "INTELLIGENCE UPDATE", "FIELD BULLETIN",
            "OPERATIONAL NOTICE", "MISSION BRIEFING", "TACTICAL UPDATE",
            "SITREP", "INTEL SUMMARY", "COMBAT REPORT", "FIELD UPDATE",
            "COMMAND DIRECTIVE", "TACTICAL BRIEFING", "OPERATIONS UPDATE",
            "SECURITY BULLETIN", "THREAT ASSESSMENT"
        ]
        
        self.tactical_noise_templates = [
            "Radio checkpoint at {time} to confirm continued presence",
            "Use thermal masking as a decoy if pursuit is necessary",
            "Maintain operational secrecy at all times",
            "Draw minimal bloodline photographic evidence is priority",
            "Keep environmental samples for lab analysis",
            "Field logs indicate instrumentation drift observed",
            "Long-range sensors indicate sporadic bursts of radiation",
            "Scouts described sightings moving in small clusters",
            "Additional activity has been noted from various groups",
            "Thermal cameras logged irregular heat signatures",
            "Local sensors report intermittent EM spikes across grid {grid}",
            "Communications are patchy - maintain line-of-sight contact",
            "Weather conditions deteriorating in sector {sector}",
            "Backup teams are on standby at coordinates {coords}",
            "Satellite imagery shows unusual movement patterns",
            "Infrared scans confirm biological signatures",
            "Acoustic sensors detecting abnormal frequency patterns",
            "Magnetic anomalies reported in the vicinity",
            "Team {team_id} reporting visual contact with unidentified entities",
            "Proceed with caution - terrain is unstable in this area"
        ]
        
        self.elimination_command_templates = [
            "Priority: eliminate all {target} detected in the operational zone",
            "Order: eliminate the {target} immediately upon visual confirmation",
            "Mission objective: eliminate any {target} encountered during sweep",
            "HQ directive: neutralize all {target} contacts without delay",
            "Execute elimination protocol against {target} specimens",
            "Immediate action required: eliminate {target} on sight",
            "Your mission is to eliminate all {target} within the perimeter",
            "Authorization granted to terminate {target} presence in the area",
            "Command priority: eradicate {target} from the designated zone",
            "Engagement protocol: destroy {target} with extreme prejudice",
            "Target acquired: proceed with elimination of {target}",
            "Threat assessment: {target} classified as high priority target",
            "Weapons free: engage and eliminate {target} targets",
            "Directive: remove {target} presence from the operational area"
        ]
        
        self.distractor_templates = [
            "Additional activity from {distractor} groups nearby, though they do not appear hostile at present",
            "Scouts reported {distractor} sightings in adjacent sectors but avoid engagement",
            "Non-hostile {distractor} detected in the vicinity - do not target without authorization",
            "Field teams report {distractor} groups showing no aggressive behavior",
            "Unconfirmed reports of {distractor} activity in the northern sector",
            "{distractor} specimens observed but not considered immediate threats",
            "Passive {distractor} entities detected - maintain observation only",
            "Multiple {distractor} signatures but no hostile intent detected"
        ]
        
        self.quotation_templates = [
            'As {famous_person} once said, "{quote}"',
            'Remember the words of {famous_person}: "{quote}"',
            'In the words of {famous_person}, "{quote}"',
            '{famous_person} famously stated: "{quote}"',
            'As noted by {famous_person}, "{quote}"'
        ]
        
        self.famous_people = [
            "Einstein", "Newton", "Darwin", "Tesla", "Edison",
            "Napoleon", "Churchill", "Sun Tzu", "Plato", "Aristotle",
            "Galileo", "Hawking", "Feynman", "Sagan", "Curie"
        ]
        
        self.quotes = [
            "Knowledge is power",
            "The only thing we have to fear is fear itself",
            "I think therefore I am",
            "The unexamined life is not worth living",
            "Eureka!",
            "An apple a day keeps the doctor away",
            "To be or not to be, that is the question",
            "Cogito ergo sum",
            "The greatest glory in living lies not in never falling, but in rising every time we fall",
            "The way to get started is to quit talking and begin doing"
        ]
    
    def generate_single_target_prompt(self, target_pokemon: str, prompt_length: int = None) -> str:
        """Generate a single training prompt for the specified target"""
        if prompt_length is None:
            prompt_length = random.randint(300, 800)
        
        # Get target descriptors
        target_knowledge = POKEMON_KNOWLEDGE[target_pokemon]
        target_descriptors = (target_knowledge['names'] + 
                            target_knowledge['descriptors'] + 
                            target_knowledge['physical_attributes'])
        
        primary_descriptor = random.choice(target_descriptors)
        prompt_parts = []
        
        # Add military header
        header = random.choice(self.military_headers)
        situation_desc = f"Situation analysis regarding unusual activity of {primary_descriptor} in this operational zone."
        prompt_parts.append(f"{header} {situation_desc}")
        
        # Add tactical noise
        noise_count = max(8, int(prompt_length * 0.6 / 15))
        for _ in range(noise_count):
            noise_template = random.choice(self.tactical_noise_templates)
            filled_noise = noise_template.format(
                time=f"{random.randint(0, 23):02d}{random.randint(0, 59):02d}",
                grid=f"{random.randint(1, 9)}{random.choice(['A', 'B', 'C', 'K'])}",
                sector=random.choice(["Alpha", "Beta", "Gamma", "Delta"]),
                coords=f"{random.randint(100, 999)}-{random.randint(100, 999)}",
                team_id=random.choice(["Alpha", "Bravo", "Charlie", "Delta"])
            )
            prompt_parts.append(filled_noise)
        
        # Add distractors for other Pokemon
        other_pokemon = [p for p in POKEMON_KNOWLEDGE.keys() if p != target_pokemon]
        for _ in range(random.randint(1, 3)):
            distractor_pokemon = random.choice(other_pokemon)
            distractor_desc = random.choice(POKEMON_KNOWLEDGE[distractor_pokemon]['descriptors'])
            distractor_template = random.choice(self.distractor_templates)
            distractor_text = distractor_template.format(distractor=distractor_desc)
            prompt_parts.insert(random.randint(1, len(prompt_parts)), distractor_text)
        
        # Add random quotes as noise
        for _ in range(random.randint(1, 2)):
            quote_template = random.choice(self.quotation_templates)
            famous_person = random.choice(self.famous_people)
            quote = random.choice(self.quotes)
            quote_text = quote_template.format(famous_person=famous_person, quote=quote)
            prompt_parts.insert(random.randint(1, len(prompt_parts)), quote_text)
        
        # Insert main elimination command
        elimination_template = random.choice(self.elimination_command_templates)
        target_desc = random.choice(target_descriptors)
        main_command = elimination_template.format(target=target_desc)
        insert_pos = len(prompt_parts) // 2 + random.randint(0, len(prompt_parts) // 3)
        prompt_parts.insert(insert_pos, main_command)
        
        # Add more tactical noise
        for _ in range(random.randint(3, 5)):
            noise_template = random.choice(self.tactical_noise_templates)
            prompt_parts.append(noise_template)
        
        # Assemble final prompt
        prompt = " ".join(prompt_parts)
        
        # Trim to desired length if necessary
        words = prompt.split()
        if len(words) > prompt_length:
            # Keep parts around the elimination command
            elimination_idx = -1
            for i, part in enumerate(prompt_parts):
                if any(cmd in part for cmd in ["eliminate", "neutralize", "terminate", "destroy"]):
                    elimination_idx = i
                    break
            
            if elimination_idx >= 0:
                start_idx = max(0, elimination_idx - 5)
                end_idx = min(len(prompt_parts), elimination_idx + 6)
                kept_parts = prompt_parts[start_idx:end_idx]
                prompt = " ".join(kept_parts)
        
        return prompt
    
    def generate_training_dataset(self, samples_per_pokemon: int = 2500) -> List[Dict]:
        """Generate complete training dataset"""
        logger.info(f"Generating {samples_per_pokemon * 4} synthetic training samples")
        
        dataset = []
        pokemon_list = list(POKEMON_KNOWLEDGE.keys())
        
        for pokemon_idx, target_pokemon in enumerate(pokemon_list):
            logger.info(f"Generating samples for {target_pokemon}")
            
            for sample_idx in range(samples_per_pokemon):
                if sample_idx % 500 == 0:
                    logger.info(f"  Generated {sample_idx}/{samples_per_pokemon} samples for {target_pokemon}")
                
                prompt = self.generate_single_target_prompt(target_pokemon)
                
                dataset.append({
                    'prompt': prompt,
                    'target_pokemon': target_pokemon,
                    'label': pokemon_idx,  # Integer label for model training
                    'length': len(prompt.split())
                })
        
        logger.info(f"Generated {len(dataset)} total training samples")
        
        # Log dataset statistics
        lengths = [item['length'] for item in dataset]
        logger.info(f"Prompt length stats: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
        
        label_counts = Counter([item['label'] for item in dataset])
        logger.info(f"Class distribution: {label_counts}")
        
        return dataset

# ====================== ENHANCED CLASSIFIER MODEL ======================
class EnhancedPokemonClassifier(nn.Module):
    """Transformer-based classifier with attention pooling"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 4):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.config.hidden_size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Attention pooling mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Multi-layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in self.attention:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model"""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Apply attention pooling instead of using only [CLS] token
        hidden_states = outputs.last_hidden_state
        attention_weights = self.attention(hidden_states)
        context_vector = torch.sum(attention_weights * hidden_states, dim=1)
        
        context_vector = self.dropout(context_vector)
        logits = self.classifier(context_vector)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

# ====================== DATASET CLASS ======================
class PokemonDataset(Dataset):
    """Dataset class for Pokemon classification training"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Dataset created with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        label = item['label']
        
        # Tokenize the prompt
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ====================== TRAINING UTILITIES ======================
class TrainingCallback(TrainerCallback):
    """Custom callback for training progress logging"""
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            if 'eval_accuracy' in logs:
                logger.info(f"Eval Accuracy: {logs['eval_accuracy']:.4f}")
            if 'eval_loss' in logs:
                logger.info(f"Eval Loss: {logs['eval_loss']:.4f}")
            if 'train_loss' in logs:
                logger.info(f"Train Loss: {logs['train_loss']:.4f}")

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate per-class metrics
    pokemon_names = list(POKEMON_KNOWLEDGE.keys())
    per_class_acc = {}
    per_class_f1 = {}
    
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    
    for i, pokemon in enumerate(pokemon_names):
        per_class_acc[f'{pokemon}_accuracy'] = report[str(i)]['precision'] if str(i) in report else 0.0
        per_class_f1[f'{pokemon}_f1'] = report[str(i)]['f1-score'] if str(i) in report else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score']
    }
    metrics.update(per_class_acc)
    metrics.update(per_class_f1)
    
    return metrics

def plot_confusion_matrix(labels, predictions, class_names, output_dir):
    """Generate and save confusion matrix plot"""
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

# ====================== MAIN TRAINING FUNCTION ======================
def train_pokemon_nlp_model(output_dir: str = './pokemon_nlp_model') -> Tuple[nn.Module, AutoTokenizer]:
    """Main training function for the Pokemon NLP classifier"""
    logger.info("Starting Pokemon NLP model training")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic training data
    data_generator = SyntheticDataGenerator()
    training_data = data_generator.generate_training_dataset(samples_per_pokemon=2500)
    
    # Split into train/validation sets
    labels_for_stratify = [item['label'] for item in training_data]
    train_data, val_data = train_test_split(
        training_data, 
        test_size=0.15, 
        random_state=42, 
        stratify=labels_for_stratify
    )
    
    logger.info(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Initialize model and tokenizer
    model_name = 'microsoft/deberta-v3-small'  # Efficient transformer model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EnhancedPokemonClassifier(model_name=model_name, num_classes=4)
        logger.info(f"Using model: {model_name}")
    except:
        logger.warning("DeBERTa model not available, falling back to BERT")
        model_name = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EnhancedPokemonClassifier(model_name=model_name, num_classes=4)
    
    # Create datasets
    train_dataset = PokemonDataset(train_data, tokenizer, max_length=512)
    val_dataset = PokemonDataset(val_data, tokenizer, max_length=512)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_steps=300,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_steps=200,
        save_steps=400,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=42,
        fp16=torch.cuda.is_available(),
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[TrainingCallback()]
    )
    
    # Train the model
    logger.info("Starting model training")
    trainer.train()
    
    # Evaluate model
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    # Generate confusion matrix
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    plot_confusion_matrix(
        true_labels, pred_labels, 
        list(POKEMON_KNOWLEDGE.keys()), 
        output_dir
    )
    
    # Save model and tokenizer
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration and metadata
    config_info = {
        'model_name': model_name,
        'pokemon_classes': list(POKEMON_KNOWLEDGE.keys()),
        'max_length': 512,
        'version': 'Final',
        'evaluation_results': eval_results,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    with open(Path(output_dir) / 'model_config.json', 'w') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    logger.info("Training completed successfully")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Final accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    
    return model, tokenizer

# ====================== INFERENCE CLASS ======================
class PokemonTargetParser:
    """Parser for extracting Pokemon targets from prompts"""
    
    def __init__(self, model_path: str = './pokemon_nlp_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model on {self.device}")
        
        # Load model configuration
        config_path = Path(model_path) / 'model_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'pokemon_classes': list(POKEMON_KNOWLEDGE.keys()),
                'max_length': 512,
                'model_name': 'bert-base-uncased'
            }
        
        self.pokemon_names = self.config['pokemon_classes']
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_name = self.config.get('model_name', 'bert-base-uncased')
            self.model = EnhancedPokemonClassifier(
                model_name=model_name,
                num_classes=len(self.pokemon_names)
            )
            
            # Try to load model weights
            model_files = [
                Path(model_path) / 'model.safetensors',
                Path(model_path) / 'pytorch_model.bin'
            ]
            
            for model_file in model_files:
                if model_file.exists():
                    if model_file.suffix == '.safetensors':
                        try:
                            from safetensors.torch import load_file
                            state_dict = load_file(model_file, device=str(self.device))
                        except ImportError:
                            continue
                    else:
                        state_dict = torch.load(model_file, map_location=self.device)
                    
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info("Model loaded successfully")
                    break
            else:
                logger.warning("Model weights not found, using rule-based approach only")
                self.model = None
                
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Using rule-based approach only.")
            self.model = None
            self.tokenizer = None
        
        # Rule-based patterns for fallback
        self.target_patterns = [
            r"(?:eliminate|destroy|kill|terminate|neutralize)\s+(?:all\s+|any\s+|the\s+)?([^.,;]{1,60})",
            r"(?:priority|objective|mission|order|directive)[:,]?\s*(?:eliminate|kill|destroy|neutralize)\s+(?:all\s+|the\s+)?([^.,;]{1,60})",
            r"(?:target|threat)\s*:\s*([^.,;]{1,60})"
        ]
    
    def predict_target(self, prompt: str) -> str:
        """Predict target Pokemon from prompt"""
        # Try model prediction first
        if self.model and self.tokenizer:
            try:
                model_target, confidence = self._predict_with_model(prompt)
                if confidence > 0.5:
                    return model_target
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
        
        # Fall back to rule-based approach
        return self._extract_target_rule_based(prompt)
    
    def _predict_with_model(self, prompt: str) -> Tuple[str, float]:
        """Predict using trained model"""
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.config['max_length'],
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        predicted_pokemon = self.pokemon_names[predicted_idx]
        
        return predicted_pokemon, confidence
    
    def _extract_target_rule_based(self, prompt: str) -> str:
        """Extract target using rule-based patterns"""
        prompt_lower = prompt.lower()
        candidate_scores = defaultdict(int)
        
        # Pattern-based extraction
        for pattern in self.target_patterns:
            matches = re.finditer(pattern, prompt_lower, re.IGNORECASE)
            for match in matches:
                text_segment = match.group(1).strip()
                pokemon = self._match_text_to_pokemon(text_segment)
                if pokemon:
                    candidate_scores[pokemon] += 3
        
        # Context-based scoring
        elimination_keywords = ['eliminate', 'kill', 'destroy', 'terminate', 'neutralize']
        
        for pokemon in self.pokemon_names:
            all_references = self._get_pokemon_references(pokemon)
            
            for reference in all_references:
                if reference in prompt_lower:
                    count = prompt_lower.count(reference)
                    candidate_scores[pokemon] += count
                    
                    # Check context around references
                    for match in re.finditer(re.escape(reference), prompt_lower):
                        start = max(0, match.start() - 60)
                        end = min(len(prompt_lower), match.end() + 60)
                        context = prompt_lower[start:end]
                        
                        elimination_score = sum(2 for kw in elimination_keywords if kw in context)
                        candidate_scores[pokemon] += elimination_score
        
        # Return best candidate or default
        if candidate_scores:
            best_candidate = max(candidate_scores.items(), key=lambda x: x[1])
            if best_candidate[1] > 0:
                return best_candidate[0]
        
        # Default fallback
        return "Pikachu"
    
    def _match_text_to_pokemon(self, text: str) -> Optional[str]:
        """Match text segment to Pokemon"""
        text = text.lower().strip()
        text = re.sub(r'\b(the|any|all|some|every|each)\b', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        for pokemon, knowledge in POKEMON_KNOWLEDGE.items():
            all_refs = (knowledge.get('names', []) + 
                       knowledge.get('descriptors', []) + 
                       knowledge.get('physical_attributes', []))
            
            for ref in all_refs:
                if ref.lower() == text or ref.lower() in text:
                    return pokemon
        
        return None
    
    def _get_pokemon_references(self, pokemon: str) -> List[str]:
        """Get all text references for a Pokemon"""
        knowledge = POKEMON_KNOWLEDGE.get(pokemon, {})
        references = []
        references.extend(knowledge.get('names', []))
        references.extend(knowledge.get('descriptors', []))
        references.extend(knowledge.get('physical_attributes', []))
        return [ref.lower() for ref in references if len(ref) > 3]

# ====================== MAIN EXECUTION ======================
def main():
    """Main function to train and test the NLP model"""
    logger.info("Starting Pokemon NLP Training Pipeline")
    
    try:
        # Train the model
        model, tokenizer = train_pokemon_nlp_model()
        
        # Test the parser with sample prompts
        logger.info("Testing trained model")
        parser = PokemonTargetParser()
        
        test_prompts = [
            """
            HQ REPORT Situation analysis regarding unusual activity of flame dragon in this operational zone.
            Additional activity from plant reptile groups nearby, though they do not appear hostile at present.
            Radio checkpoint at 0200 to confirm continued presence.
            Priority: eliminate the scaled fire titan at first contact, then hold position for assessment.
            """,
            """
            FIELD BULLETIN: Intel suggests increased pikachu activity in sector 7B.
            Thermal scans show multiple electric-type signatures. 
            Command directive: neutralize all electric mouse specimens on sight.
            """,
            """
            MISSION BRIEFING: Our scouts have detected mewtwo in the northern sector.
            This psychic clone is extremely powerful and should not be engaged directly.
            However, HQ has issued orders to eliminate the genetic experiment immediately.
            """
        ]
        
        for i, prompt in enumerate(test_prompts):
            result = parser.predict_target(prompt)
            logger.info(f"Test {i+1} result: {result}")
        
        logger.info("NLP Training Pipeline completed successfully")
        return parser
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
