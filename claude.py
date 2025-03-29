"""
Enhanced Toxicity Detection Model
--------------------------------
An improved implementation of the toxicity detection system with the following key enhancements:
1. Advanced feature engineering with context-aware detection
2. Enhanced model architecture with self-attention and residual connections
3. Sophisticated classifier chain with dynamic threshold adjustment
4. Uncertainty estimation using Monte Carlo dropout
5. Comprehensive evaluation and error analysis
6. Improved language detection and multilingual support
"""

import os
import time
import copy
import math
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, roc_auc_score, 
    precision_recall_curve, confusion_matrix, precision_score, recall_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm.auto import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("toxicity_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("toxicity_model")

# Suppress warnings
warnings.filterwarnings('ignore')

# ==================================================================================
# Configuration
# ==================================================================================

class Config:
    # Data settings
    TEXT_COLUMN = 'comment'
    TOXICITY_COLUMN = 'toxicity_level'
    CATEGORY_COLUMNS = ['insult', 'profanity', 'threat', 'identity_hate']
    MAX_CHARS = 300
    USE_LANGUAGE_DETECTION = True
    
    # Vocabulary settings
    USE_HYBRID_VOCABULARY = True
    ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    MAX_VOCAB_SIZE = 550
    MIN_CHAR_COUNT = 2
    
    # Model architecture settings
    CHAR_EMB_DIM = 64
    LSTM_HIDDEN_DIM = 96
    DROPOUT_RATE = 0.35
    
    # CNN configurations - enhanced architecture
    CNN_CONFIGS = [
        {'large_features': 256, 'small_features': 64, 'kernel': 7, 'pool': 3, 'batch_norm': True},
        {'large_features': 256, 'small_features': 64, 'kernel': 5, 'pool': 3, 'batch_norm': True},
        {'large_features': 256, 'small_features': 64, 'kernel': 3, 'pool': 3, 'batch_norm': True},
        {'large_features': 256, 'small_features': 64, 'kernel': 3, 'pool': 3, 'batch_norm': True},
    ]
    
    # Training settings
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 40
    EARLY_STOPPING_PATIENCE = 6
    
    # Class weights for imbalance handling
    FOCAL_ALPHA = [2.5, 1.0, 1.0]
    
    # Category weights - increased for harder categories
    CATEGORY_WEIGHTS = [2.0, 1.5, 2.0, 2.5]
    
    # Classification thresholds - optimized using automated calibration
    CATEGORY_THRESHOLDS = [0.65, 0.70, 0.65, 0.65]
    
    # Training enhancements
    CATEGORY_LOSS_SCALE = 1.2
    USE_GRADIENT_CLIPPING = True
    GRADIENT_CLIP_VALUE = 1.0
    NUM_WORKERS = 4
    SEED = 42
    USE_ONE_CYCLE_LR = True
    USE_LABEL_SMOOTHING = True
    LABEL_SMOOTHING = 0.1
    
    # Evaluation settings
    MC_DROPOUT_SAMPLES = 30
    UNCERTAINTY_THRESHOLD = 0.08
    
    # Feedback system settings
    MIN_FEEDBACK_FOR_RETRAINING = 12
    FEEDBACK_RETRAIN_EPOCHS = 15
    FEEDBACK_LEARNING_RATE = 0.0001
    
    # Language-specific thresholds - precision-calibrated
    LANGUAGE_THRESHOLDS = {
        'en': {
            'toxicity': 0.75,
            'insult': 0.75,
            'profanity': 0.80,
            'threat': 0.70,
            'identity_hate': 0.65,
            'severity': 0.55
        },
        'tl': {
            'toxicity': 0.85,
            'insult': 0.80,
            'profanity': 0.90,
            'threat': 0.75,
            'identity_hate': 0.75,
            'severity': 0.60
        }
    }
    
    # Safe word settings
    SAFE_WORD_SETTINGS = {
        'enable_safe_word_features': True,
        'safe_word_threshold_boost': 0.08,
        'max_threshold': 0.95,
        'benign_phrases': []  # Will be populated from CSV
    }
    
    # Paths
    DATA_PATH = 'csv files/17000datas.csv'
    PROFANITY_LIST_PATH = 'csv files/extended_profanity_list.csv'
    SAFE_WORDS_PATH = 'csv files/safeword,phrases,mixed.csv'
    OUTPUT_DIR = 'train_models'
    
    # Feature engineering settings
    USE_ADVANCED_FEATURES = True
    COUNT_SPECIAL_CHARS = True
    COUNT_REPEATED_CHARS = True
    DETECT_EDUCATIONAL_CONTENT = True
    USE_TEXT_STATS = True
    
    # Create combined CONFIG dictionary for backward compatibility
    @classmethod
    def as_dict(cls):
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith('__') and not callable(value)}

# Create CONFIG dictionary for backward compatibility
CONFIG = Config.as_dict()

# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ==================================================================================
# Enhanced Character Vocabulary with Subword Features
# ==================================================================================

class EnhancedCharacterVocabulary:
    def __init__(self, fixed_alphabet=None, max_vocab_size=500):
        # Default alphabet if none provided
        self.default_alphabet = Config.ALPHABET
        self.max_vocab_size = max_vocab_size
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.sos_token = '<SOS>'  # Start of sentence
        self.eos_token = '<EOS>'  # End of sentence
        
        # Initialize with special tokens
        self.char_to_idx = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.sos_token: 2,
            self.eos_token: 3
        }
        self.idx_to_char = {
            0: self.pad_token,
            1: self.unk_token,
            2: self.sos_token,
            3: self.eos_token
        }
        self.n_chars = 4  # Count of special tokens

        # Character frequency tracking
        self.char_count = {}
        
        # Subword features - character bigram and trigram vocabulary
        self.use_ngrams = True
        self.ngram_tokens = {}
        self.ngram_counts = {}

        # Add fixed alphabet first if provided
        if fixed_alphabet is not None:
            self.add_fixed_alphabet(fixed_alphabet)
    
    def add_fixed_alphabet(self, alphabet):
        """Add a fixed alphabet to the vocabulary."""
        logger.info(f"Adding fixed alphabet with {len(alphabet)} characters")
        
        # Add each character from the alphabet to the vocabulary
        for char in alphabet:
            if char not in self.char_to_idx:
                self.char_to_idx[char] = self.n_chars
                self.idx_to_char[self.n_chars] = char
                self.char_count[char] = float('inf')  # Mark as fixed alphabet character
                self.n_chars += 1

        logger.info(f"After adding fixed alphabet: {self.n_chars} characters")
    
    def build_from_texts(self, texts, min_count=2):
        """Build vocabulary from texts, including character n-grams."""
        logger.info("Building vocabulary from training data...")
        
        # Count characters
        for text in texts:
            # Count individual characters
            for char in text:
                if char not in self.char_count:
                    self.char_count[char] = 0
                self.char_count[char] += 1
                
            # Count character bigrams and trigrams if enabled
            if self.use_ngrams:
                for i in range(len(text) - 1):
                    bigram = text[i:i+2]
                    if bigram not in self.ngram_counts:
                        self.ngram_counts[bigram] = 0
                    self.ngram_counts[bigram] += 1
                    
                for i in range(len(text) - 2):
                    trigram = text[i:i+3]
                    if trigram not in self.ngram_counts:
                        self.ngram_counts[trigram] = 0
                    self.ngram_counts[trigram] += 1

        # Add frequently occurring characters that aren't already in the vocabulary
        chars_added = 0
        for char, count in sorted(self.char_count.items(), key=lambda x: x[1], reverse=True):
            # Skip if already in vocabulary
            if char in self.char_to_idx:
                continue
                
            # Skip if below minimum count
            if count < min_count:
                continue
                
            # Skip if we've reached maximum vocabulary size
            if self.n_chars >= self.max_vocab_size:
                break
                
            # Add to vocabulary
            self.char_to_idx[char] = self.n_chars
            self.idx_to_char[self.n_chars] = char
            self.n_chars += 1
            chars_added += 1
        
        # Add special attention to important characters for toxicity detection
        special_toxicity_chars = [
            # Common substitutions in toxic text
            '@', '0', '1', '3', '4', '$', '&', '*', '#', '+', '<', '>'
        ]

        for char in special_toxicity_chars:
            if char not in self.char_to_idx and self.n_chars < self.max_vocab_size:
                self.char_to_idx[char] = self.n_chars
                self.idx_to_char[self.n_chars] = char
                self.n_chars += 1
                chars_added += 1
                
        # Add most frequent character n-grams (if enabled)
        if self.use_ngrams and self.n_chars < self.max_vocab_size:
            # Sort n-grams by frequency
            sorted_ngrams = sorted(self.ngram_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Take top n-grams up to max_vocab_size
            ngram_tokens = {}
            for ngram, count in sorted_ngrams:
                if len(ngram_tokens) >= 50 or self.n_chars + len(ngram_tokens) >= self.max_vocab_size:
                    break
                if count >= min_count * 5:  # Higher threshold for n-grams
                    ngram_tokens[ngram] = self.n_chars + len(ngram_tokens)
            
            # Add n-grams to vocabulary
            for ngram, idx in ngram_tokens.items():
                self.char_to_idx[ngram] = idx
                self.idx_to_char[idx] = ngram
                self.n_chars += 1
                chars_added += 1
            
            self.ngram_tokens = ngram_tokens
            
        logger.info(f"Added {chars_added} new characters/n-grams from training data")
        logger.info(f"Final vocabulary size: {self.n_chars} tokens")
        
        # Print some statistics about character coverage
        total_chars = sum(self.char_count.values())
        covered_chars = sum(count for char, count in self.char_count.items() if char in self.char_to_idx)
        coverage = covered_chars / total_chars * 100 if total_chars > 0 else 0
        logger.info(f"Character coverage: {coverage:.2f}% of all character occurrences")

    def encode_text(self, text, max_len=300):
        """Convert text to sequence of character indices."""
        # Pre-allocate array with pad tokens
        indices = np.full(max_len, self.char_to_idx[self.pad_token], dtype=np.int64)
        
        # Optional: Add start token
        indices[0] = self.char_to_idx[self.sos_token]
        start_pos = 1
        
        # Fill with actual character indices
        char_pos = start_pos
        i = 0
        while i < len(text) and char_pos < max_len - 1:  # Leave space for EOS
            # Try to match n-grams first (if enabled)
            ngram_matched = False
            if self.use_ngrams and i < len(text) - 2:
                # Try trigram
                trigram = text[i:i+3]
                if trigram in self.char_to_idx:
                    indices[char_pos] = self.char_to_idx[trigram]
                    char_pos += 1
                    i += 3
                    ngram_matched = True
                    continue
                
                # Try bigram
                bigram = text[i:i+2]
                if bigram in self.char_to_idx:
                    indices[char_pos] = self.char_to_idx[bigram]
                    char_pos += 1
                    i += 2
                    ngram_matched = True
                    continue
            
            # If no n-gram matched, encode individual character
            if not ngram_matched:
                char = text[i]
                indices[char_pos] = self.char_to_idx.get(char, self.char_to_idx[self.unk_token])
                char_pos += 1
                i += 1
        
        # Add end token if there's space
        if char_pos < max_len:
            indices[char_pos] = self.char_to_idx[self.eos_token]
        
        return indices

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

# ==================================================================================
# Enhanced Text Preprocessing
# ==================================================================================

# Pre-compiled patterns for efficiency
import re
WHITESPACE_PATTERN = re.compile(r'\s+')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
REPEATED_CHARS_PATTERN = re.compile(r'(.)\1{3,}')
SPECIAL_CHARS_PATTERN = re.compile(r'[!@#$%^&*(),.?":{}|<>]')
NUMBER_PATTERN = re.compile(r'\d+')
EDUCATIONAL_PATTERNS = [
    re.compile(r'\b(?:educat|teach|learn|study|school|college|university|class|course|lecture|lesson|assignment|homework|test|exam|quiz|grade|student|professor|teacher)\w*\b', re.IGNORECASE),
    re.compile(r'\bexplain\w*\b|\bdiscuss\w*\b|\bdefin\w*\b|\banalyz\w*\b|\bresearch\w*\b', re.IGNORECASE),
    re.compile(r'\bFYI\b|\bfor\s+your\s+information\b|\bfor\s+context\b|\bexample\b|\billustrat\w*\b', re.IGNORECASE)
]

def enhanced_text_preprocessing(text, max_len=300, normalize_repeats=True):
    """
    Enhanced text preprocessing with additional normalization.
    
    Args:
        text: Input text to preprocess
        max_len: Maximum length to keep
        normalize_repeats: Whether to normalize repeated characters
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with special token
    text = URL_PATTERN.sub(" <URL> ", text)
    
    # Replace emails with special token
    text = EMAIL_PATTERN.sub(" <EMAIL> ", text)
    
    # Normalize repeated characters (e.g., "hellooooo" -> "helloo")
    if normalize_repeats:
        text = REPEATED_CHARS_PATTERN.sub(r'\1\1\1', text)
    
    # Remove excessive whitespace
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    
    # Truncate if needed
    if len(text) > max_len:
        text = text[:max_len]
    
    return text

# Language detection function
def detect_language(text):
    """
    Detect language of text (currently supports English and Tagalog).
    
    Args:
        text: Input text
        
    Returns:
        Language code ('en' or 'tl')
    """
    # Common Tagalog words - enhanced list
    tagalog_markers = [
        # Standard function words / particles and common pronouns
        "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila", 
        "ang", "ng", "sa", "mga", "ni", "namin", "natin", "nila",
        "hindi", "oo", "opo", "wala", "meron", "dahil", "kasi",
        
        # Additional function words, particles, and connectors
        "na", "nang", "lang", "lamang", "ba", "daw", "raw", "pala",
        "kaya", "pero", "ngunit", "subalit", "at", "o", "kung", 
        "kapag", "pag", "sapagkat", "para", "pwede", "puwede", 
        "baka", "siguro", "marahil", "naman", "nga", "kay", "kina", "nina",
        
        # Pronouns, demonstratives, and interrogatives
        "ito", "iyan", "iyon", 
        "sino", "ano", "saan", "kailan", "bakit", "paano", "ilan",
        
        # Common verbal affixes (use with caution for token matching)
        "mag", "nag", "um", "in", "an", "ma", "ipag", "ipa", "pa",
        
        # Additional words and expressions
        "ayaw", "paki", "salamat", "walang", "anuman", 
        "pasensya", "pasensiya", "mahal", "murang", 
        "malaki", "maliit", "masaya", "malungkot", 
        "maganda", "gwapo",
        
        # More common Tagalog words for better detection
        "yung", "po", "opo", "yun", "dito", "diyan", "doon",
        "kanina", "bukas", "kahapon", "ngayon", "mamaya",
        "nasaan", "nasaaan", "gusto", "ayoko", "talaga",
        "sobra", "grabe", "mabuti", "masama"
    ]
    
    # Clean and tokenize the text
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Skip very short texts
    if len(words) < 3:
        return 'en'  # Default to English
    
    # Count Tagalog markers
    tagalog_count = sum(1 for word in words if word in tagalog_markers)
    tagalog_ratio = tagalog_count / len(words)
    
    # Reduced threshold for Tagalog detection
    if tagalog_ratio > 0.12:  # If more than 12% words are Tagalog markers
        return 'tl'
    else:
        return 'en'

# ==================================================================================
# Advanced Feature Extraction
# ==================================================================================

class FeatureExtractor:
    """Enhanced feature extraction with caching for toxicity detection."""
    
    def __init__(self):
        self.toxic_keywords = self._load_toxic_keywords()
        self.safe_words = self._load_safe_words()
        self.educational_patterns = EDUCATIONAL_PATTERNS
        self.feature_cache = {}
    
    def _load_toxic_keywords(self):
        """Load toxic keywords from CSV file."""
        csv_path = Config.PROFANITY_LIST_PATH
        if os.path.exists(csv_path):
            try:
                # Try different ways to read the CSV depending on its structure
                try:
                    # If the CSV has headers
                    df = pd.read_csv(csv_path)
                    toxic_keywords = df.iloc[:, 0].tolist()
                except:
                    # If the CSV is just a list of words with no header
                    toxic_keywords = pd.read_csv(csv_path, header=None)[0].tolist()
                
                # Remove any NaN values and convert to lowercase
                toxic_keywords = [str(word).lower() for word in toxic_keywords if str(word) != 'nan']
                logger.info(f"Loaded {len(toxic_keywords)} toxic keywords")
                return toxic_keywords
            except Exception as e:
                logger.error(f"Error loading toxic keywords: {e}")
        
        # Fallback to a small default list
        logger.warning("Using default toxic keywords list")
        return ['fuck', 'shit', 'ass', 'bitch', 'damn', 'cunt', 'dick', 'pussy', 'nigger', 'faggot']
    
    def _load_safe_words(self):
        """Load safe words/phrases from CSV file."""
        csv_path = Config.SAFE_WORDS_PATH
        if os.path.exists(csv_path):
            try:
                # Try different ways to read the CSV depending on its structure
                try:
                    # If the CSV has headers
                    df = pd.read_csv(csv_path)
                    safe_words = df.iloc[:, 0].tolist()
                except:
                    # If the CSV is just a list of words with no header
                    safe_words = pd.read_csv(csv_path, header=None)[0].tolist()
                
                # Remove any NaN values and convert to lowercase
                safe_words = [str(word).lower() for word in safe_words if str(word) != 'nan']
                logger.info(f"Loaded {len(safe_words)} safe words/phrases")
                return safe_words
            except Exception as e:
                logger.error(f"Error loading safe words: {e}")
        
        # Fallback to an empty list
        logger.warning("Using empty safe words list")
        return []
    
    def extract_features(self, text):
        """
        Extract comprehensive features for toxicity detection.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        # Check cache first
        text_hash = hash(text)
        if text_hash in self.feature_cache:
            return self.feature_cache[text_hash]
        
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
            
        # Initialize features dictionary
        features = {}
        
        # Get lowercase text and words
        lower_text = text.lower()
        words = lower_text.split()
        
        # 1. ALL CAPS ratio
        all_caps_words = [w for w in words if len(w) > 2 and w.isupper()]
        features['all_caps_ratio'] = len(all_caps_words) / max(1, len(words))
        
        # 2. Toxic keyword detection
        toxic_count = 0
        detected_keywords = []
        
        for keyword in self.toxic_keywords:
            if keyword in lower_text:
                toxic_count += 1
                detected_keywords.append(keyword)
        
        features['toxic_keyword_count'] = toxic_count
        features['toxic_keyword_ratio'] = toxic_count / max(1, len(words))
        features['detected_keywords'] = detected_keywords
        
        # 3. Safe word detection
        safe_word_count = 0
        detected_safe_words = []
        
        for safe_phrase in self.safe_words:
            if safe_phrase in lower_text:
                safe_word_count += 1
                detected_safe_words.append(safe_phrase)
        
        features['safe_word_count'] = safe_word_count
        features['safe_word_ratio'] = safe_word_count / max(1, len(words))
        features['detected_safe_words'] = detected_safe_words
        
        # 4. Calculate toxicity-safety ratio
        features['toxicity_safe_ratio'] = toxic_count / max(1, safe_word_count)
        
        # 5. Special characters ratio
        special_chars = SPECIAL_CHARS_PATTERN.findall(text)
        features['special_char_count'] = len(special_chars)
        features['special_char_ratio'] = len(special_chars) / max(1, len(text))
        
        # 6. Repeated characters
        repeated_char_matches = REPEATED_CHARS_PATTERN.findall(text)
        features['repeated_char_count'] = len(repeated_char_matches)
        
        # 7. Educational content detection
        is_educational = False
        edu_matches = []
        
        for pattern in self.educational_patterns:
            matches = pattern.findall(text)
            if matches:
                edu_matches.extend(matches)
                is_educational = True
        
        features['is_educational'] = is_educational
        features['educational_term_count'] = len(edu_matches)
        
        # 8. Text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = sum(len(w) for w in words) / max(1, len(words))
        
        # 9. Potentially safe context
        if safe_word_count > 0 and (is_educational or safe_word_count >= 2):
            features['potentially_safe'] = True
        else:
            features['potentially_safe'] = False
        
        # Save to cache
        self.feature_cache[text_hash] = features
        
        return features

# Global feature extractor instance
feature_extractor = FeatureExtractor()

# Wrapper function for backward compatibility
def extract_toxicity_features(text):
    """Extract features for toxicity detection (backwards compatible wrapper)."""
    return feature_extractor.extract_features(text)

# ==================================================================================
# Enhanced Dataset for Toxicity Detection
# ==================================================================================

class EnhancedToxicityDataset(Dataset):
    def __init__(self, texts, labels=None, char_vocab=None, max_len=300, detect_lang=False):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.detect_lang = detect_lang
        
        # Pre-process all texts
        logger.info("Preprocessing texts...")
        self.processed_texts = [enhanced_text_preprocessing(text, max_len) for text in texts]
        
        # Extract features for all texts
        logger.info("Extracting toxicity features...")
        self.toxicity_features = [feature_extractor.extract_features(text) for text in tqdm(self.processed_texts)]
        
        # Initialize character vocabulary if not provided
        if char_vocab is None:
            if Config.USE_HYBRID_VOCABULARY:
                self.char_vocab = EnhancedCharacterVocabulary(
                    fixed_alphabet=Config.ALPHABET,
                    max_vocab_size=Config.MAX_VOCAB_SIZE
                )
                self.char_vocab.build_from_texts(self.processed_texts, min_count=Config.MIN_CHAR_COUNT)
            else:
                self.char_vocab = EnhancedCharacterVocabulary()
                self.char_vocab.build_from_texts(self.processed_texts)
        else:
            self.char_vocab = char_vocab
        
        # Detect languages if enabled
        if self.detect_lang:
            logger.info("Detecting languages for texts...")
            self.languages = [detect_language(text) for text in tqdm(self.processed_texts)]
            lang_counts = Counter(self.languages)
            logger.info(f"Language distribution: {dict(lang_counts)}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Convert idx to int if it's a string
        if isinstance(idx, str):
            try:
                idx = int(idx)
            except ValueError:
                raise TypeError(f"Cannot convert idx '{idx}' to integer")
            
        # Get pre-processed text
        processed_text = self.processed_texts[idx]
            
        # Get toxicity features
        features = self.toxicity_features[idx]
            
        # Encode text to character indices
        char_ids = self.char_vocab.encode_text(processed_text, self.max_len)
            
        if self.labels is not None:
            item = {
                'char_ids': torch.tensor(char_ids, dtype=torch.long),
                'labels': torch.tensor(self.labels[idx], dtype=torch.float),
                'text': processed_text,
                'all_caps_ratio': torch.tensor(features['all_caps_ratio'], dtype=torch.float),
                'toxic_keyword_count': torch.tensor(features['toxic_keyword_count'], dtype=torch.float),
                'toxic_keyword_ratio': torch.tensor(features['toxic_keyword_ratio'], dtype=torch.float),
                'safe_word_count': torch.tensor(features.get('safe_word_count', 0), dtype=torch.float),
                'safe_word_ratio': torch.tensor(features.get('safe_word_ratio', 0), dtype=torch.float),
                'special_char_ratio': torch.tensor(features.get('special_char_ratio', 0), dtype=torch.float),
                'is_educational': torch.tensor(float(features.get('is_educational', False)), dtype=torch.float),
                'text_length': torch.tensor(features.get('text_length', 0), dtype=torch.float)
            }
            
            # Add language info if available
            if self.detect_lang:
                item['language'] = self.languages[idx]
            return item
        else:
            item = {
                'char_ids': torch.tensor(char_ids, dtype=torch.long),
                'text': processed_text,
                'all_caps_ratio': torch.tensor(features['all_caps_ratio'], dtype=torch.float),
                'toxic_keyword_count': torch.tensor(features['toxic_keyword_count'], dtype=torch.float),
                'toxic_keyword_ratio': torch.tensor(features['toxic_keyword_ratio'], dtype=torch.float),
                'safe_word_count': torch.tensor(features.get('safe_word_count', 0), dtype=torch.float),
                'safe_word_ratio': torch.tensor(features.get('safe_word_ratio', 0), dtype=torch.float),
                'special_char_ratio': torch.tensor(features.get('special_char_ratio', 0), dtype=torch.float),
                'is_educational': torch.tensor(float(features.get('is_educational', False)), dtype=torch.float),
                'text_length': torch.tensor(features.get('text_length', 0), dtype=torch.float)
            }
            
            # Add language info if available
            if self.detect_lang:
                item['language'] = self.languages[idx]
            return item

# ==================================================================================
# Data Loading Functions
# ==================================================================================

def load_data_from_csv(file_path, text_column=None, toxicity_column=None, category_columns=None):
    """
    Load dataset from CSV file.
    
    Args:
        file_path: Path to CSV file
        text_column: Column name for text data
        toxicity_column: Column name for toxicity labels
        category_columns: List of category column names
        
    Returns:
        Tuple of (texts, labels)
    """
    logger.info(f"Loading data from {file_path}...")
    
    # Get column names from CONFIG if not specified
    if text_column is None:
        text_column = Config.TEXT_COLUMN
    if toxicity_column is None:
        toxicity_column = Config.TOXICITY_COLUMN
    if category_columns is None:
        category_columns = Config.CATEGORY_COLUMNS
    
    try:
        # Try with different encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning("UTF-8 encoding failed, trying latin-1...")
            df = pd.read_csv(file_path, encoding='latin-1')
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Check for missing values
        missing_count = df[text_column].isna().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing text values, dropping them")
            df = df.dropna(subset=[text_column])
        
        # Check if columns exist
        missing_columns = []
        if text_column not in df.columns:
            missing_columns.append(text_column)
        if toxicity_column not in df.columns:
            missing_columns.append(toxicity_column)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Extract texts
        texts = df[text_column].tolist()
        
        # Create labels array [toxicity, insult, profanity, threat, identity_hate]
        toxicity_levels = df[toxicity_column].astype(int).values
        
        # Initialize the labels array
        labels = np.zeros((len(df), 1 + len(category_columns)))
        labels[:, 0] = toxicity_levels
        
        # Add category values if available
        for i, col in enumerate(category_columns):
            if col in df.columns:
                labels[:, i+1] = df[col].astype(int).values
            else:
                logger.warning(f"Category column '{col}' not found. Using all zeros.")
        
        # Print data distribution
        logger.info(f"\nToxicity level distribution:")
        for level, count in sorted(Counter(toxicity_levels).items()):
            percentage = count / len(toxicity_levels) * 100
            logger.info(f"  Level {level}: {count} examples ({percentage:.1f}%)")
        
        logger.info(f"\nCategory distribution:")
        for i, col in enumerate(category_columns):
            positive_count = np.sum(labels[:, i+1] == 1)
            percentage = positive_count / len(labels) * 100
            logger.info(f"  {col}: {positive_count} positive examples ({percentage:.1f}%)")
        
        return texts, labels
    
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise

def create_data_loaders(texts, labels, char_vocab=None, test_size=0.2, val_size=0.25, 
                       batch_size=32, num_workers=4, max_len=300, detect_lang=False, seed=42):
    """
    Create train, validation, and test data loaders.
    
    Args:
        texts: List of texts
        labels: Array of labels
        char_vocab: Character vocabulary (optional)
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        batch_size: Batch size
        num_workers: Number of workers for data loading
        max_len: Maximum sequence length
        detect_lang: Whether to detect language
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, char_vocab)
    """
    from sklearn.model_selection import train_test_split
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Check if labels has the expected shape
    if labels.shape[1] != 1 + len(Config.CATEGORY_COLUMNS):
        raise ValueError(f"Labels shape {labels.shape} doesn't match expected shape "
                         f"({len(texts)}, {1 + len(Config.CATEGORY_COLUMNS)})")
    
    # Perform train/test split with stratification by toxicity level
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels[:, 0]
    )
    
    # Perform train/val split with stratification by toxicity level
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=val_size, random_state=seed, stratify=train_labels[:, 0]
    )
    
    logger.info(f"Split data into {len(train_texts)} training, {len(val_texts)} validation, "
               f"and {len(test_texts)} test examples")
    
    # Create datasets
    train_dataset = EnhancedToxicityDataset(
        train_texts, train_labels, char_vocab, max_len=max_len, detect_lang=detect_lang
    )
    
    # If char_vocab wasn't provided, use the one created by the train_dataset
    if char_vocab is None:
        char_vocab = train_dataset.char_vocab
    
    val_dataset = EnhancedToxicityDataset(
        val_texts, val_labels, char_vocab, max_len=max_len, detect_lang=detect_lang
    )
    
    test_dataset = EnhancedToxicityDataset(
        test_texts, test_labels, char_vocab, max_len=max_len, detect_lang=detect_lang
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, char_vocab

# ==================================================================================
# Enhanced Neural Network Architecture
# ==================================================================================

class EnhancedCNNLayer(nn.Module):
    """Enhanced CNN layer with residual connections and advanced normalization."""
    
    def __init__(self, input_channels, large_features, small_features, 
                 kernel_size, pool_size=None, batch_norm=True, dropout_rate=0.1):
        super(EnhancedCNNLayer, self).__init__()
        
        # Store channels for residual connection checking
        self.input_channels = input_channels
        self.output_channels = small_features
        
        # Primary convolution with weight normalization
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=large_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # Same padding
            )
        )
        
        # Batch normalization (improved stability)
        self.batch_norm = nn.BatchNorm1d(large_features) if batch_norm else None
        
        # Pooling layer (optional) - using adaptive pooling
        if pool_size is not None:
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        else:
            self.pool = None
        
        # Dimension reduction with 1x1 convolution, with weight normalization
        self.reduce = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels=large_features,
                out_channels=small_features,
                kernel_size=1  # 1x1 convolution
            )
        )
        
        # Batch normalization for reduction
        self.reduce_bn = nn.BatchNorm1d(small_features) if batch_norm else None
        
        # Lightweight dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Residual projection if input and output channels don't match
        self.residual_proj = None
        if input_channels != small_features:
            self.residual_proj = nn.Conv1d(
                in_channels=input_channels,
                out_channels=small_features,
                kernel_size=1
            )
    
    def forward(self, x):
        # Store input for residual connection
        residual = x
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply batch normalization if present
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        # Apply activation
        x = F.relu(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply pooling if it exists
        if self.pool is not None:
            x = self.pool(x)
            # Need to adapt residual connection shape
            if residual is not None and self.residual_proj is None:
                # If no projection layer exists but shapes changed due to pooling,
                # we need to manually adapt the residual shape
                residual = F.avg_pool1d(residual, kernel_size=self.pool.kernel_size, 
                                      stride=self.pool.stride)
        
        # Apply dimension reduction with 1x1 convolution
        x = self.reduce(x)
        
        # Apply batch normalization for reduction if present
        if self.reduce_bn is not None:
            x = self.reduce_bn(x)
        
        # Add residual connection if possible (after adapting shape if needed)
        if residual is not None:
            if residual.shape[1] != x.shape[1]:
                if self.residual_proj is not None:
                    residual = self.residual_proj(residual)
            
            # Check if shapes match for addition
            if residual.shape == x.shape:
                x = x + residual
        
        # Final activation and dropout
        x = F.relu(x)
        x = self.dropout(x)
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-like character attention."""
    
    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin and cos functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1), :]

class EnhancedToxicityModel(nn.Module):
    """Enhanced model architecture for toxicity detection."""
    
    def __init__(self, n_chars, n_classes=5, char_emb_dim=64, lstm_hidden_dim=96, dropout_rate=0.35, 
                feature_dim=8, use_attention=True):
        super(EnhancedToxicityModel, self).__init__()
        
        # Character embedding layer with dropout
        self.char_embedding = nn.Embedding(n_chars, char_emb_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout_rate)
        
        # Store dimensions for later use
        self.char_emb_dim = char_emb_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        # Build CNN layers
        self.cnn_layers = nn.ModuleList()
        input_channels = char_emb_dim
        
        # Create each CNN layer based on configuration
        for layer_config in Config.CNN_CONFIGS:
            cnn_layer = EnhancedCNNLayer(
                input_channels=input_channels,
                large_features=layer_config['large_features'],
                small_features=layer_config['small_features'],
                kernel_size=layer_config['kernel'],
                pool_size=layer_config.get('pool'),
                batch_norm=layer_config.get('batch_norm', True),
                dropout_rate=dropout_rate/2  # Lighter dropout in CNN layers
            )
            
            self.cnn_layers.append(cnn_layer)
            input_channels = layer_config['small_features']
        
        # FC layer before sequence model with layer normalization
        self.fc = nn.Linear(input_channels, 256)
        self.fc_norm = nn.LayerNorm(256)
        self.fc_dropout = nn.Dropout(dropout_rate)
        
        # Bidirectional LSTM with variational dropout
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_dim,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if lstm_hidden_dim > 1 else 0
        )
        
        # Self-attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_hidden_dim * 2,  # BiLSTM output dimension
                num_heads=4,
                dropout=dropout_rate
            )
            self.pos_encoder = PositionalEncoding(lstm_hidden_dim * 2)
            self.attention_norm = nn.LayerNorm(lstm_hidden_dim * 2)
        
        # Feature processing layers with batch normalization
        # Process both basic toxicity features and advanced features
        self.feature_fc = nn.Linear(feature_dim, 64)
        self.feature_bn = nn.BatchNorm1d(64)
        self.feature_dropout = nn.Dropout(dropout_rate)
        
        # Combine LSTM and feature outputs
        combined_dim = (lstm_hidden_dim * 2) + 64  # BiLSTM output + feature dimensions
        
        # Additional non-linearity for combined features
        self.combined_fc = nn.Linear(combined_dim, combined_dim)
        self.combined_bn = nn.BatchNorm1d(combined_dim)
        
        # Heavy dropout for better regularization
        self.dropout = nn.Dropout(dropout_rate + 0.05)
        
        # Output layers with weight normalization
        self.fc_toxicity = nn.Linear(combined_dim, 3)  # 3 toxicity levels
        self.fc_category = nn.Linear(combined_dim, 4)  # 4 toxicity categories
        
        # Initialize weights with improved schemes
        self._init_weights()
        
        # Apply weight normalization to output layers
        self.fc_toxicity = nn.utils.parametrizations.weight_norm(self.fc_toxicity)
        self.fc_category = nn.utils.parametrizations.weight_norm(self.fc_category)
        
        # Cache for extracted features (used by classifier chain)
        self.extracted_features = None
    
    def _init_weights(self):
        """Initialize weights with sophisticated schemes."""
        # Initialize embedding with truncated normal distribution
        nn.init.trunc_normal_(self.char_embedding.weight, mean=0, std=0.1, a=-0.2, b=0.2)
        
        # Initialize convolutional layers with Kaiming He method
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize linear layers with Xavier/Glorot
        for module in [self.fc, self.feature_fc, self.combined_fc]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        
        # Initialize LSTM with orthogonal weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize output layers with specific biases
        # Use Xavier/Glorot for weights
        nn.init.xavier_normal_(self.fc_toxicity.weight, gain=1.0)
        # Slight negative bias to reduce false positive rate
        nn.init.constant_(self.fc_toxicity.bias, -0.1)
        
        nn.init.xavier_normal_(self.fc_category.weight, gain=1.0)
        # Slight negative bias for category detection
        nn.init.constant_(self.fc_category.bias, -0.2)
    
    def forward(self, char_ids, toxicity_features):
        """
        Forward pass through the model.
        
        Args:
            char_ids: Character IDs tensor [batch_size, seq_len]
            toxicity_features: Toxicity features tensor [batch_size, feature_dim]
            
        Returns:
            Tuple of (toxicity_output, category_output)
        """
        # Character embeddings [batch_size, seq_len, char_emb_dim]
        char_embeds = self.char_embedding(char_ids)
        char_embeds = self.embed_dropout(char_embeds)
        
        # Convolutional layers expect [batch_size, channel, seq_len]
        x = char_embeds.permute(0, 2, 1)
        
        # Apply CNN layers
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        # Reshape for FC and sequence processing
        batch_size, channels, seq_len = x.size()
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, channels]
        
        # Apply FC layer to each position with layer normalization
        x_fc = self.fc(x)
        x_fc = self.fc_norm(x_fc)
        x_fc = F.gelu(x_fc)  # GELU activation
        x_fc = self.fc_dropout(x_fc)
        
        # Apply BiLSTM
        lstm_out, _ = self.lstm(x_fc)
        
        # Apply self-attention if enabled
        if self.use_attention:
            # Add positional encoding
            pos_encoded = self.pos_encoder(lstm_out)
            
            # Self-attention (transpose for attention layer)
            attn_out, _ = self.attention(
                pos_encoded.transpose(0, 1),
                pos_encoded.transpose(0, 1),
                pos_encoded.transpose(0, 1)
            )
            
            # Add residual connection and normalization
            attn_out = attn_out.transpose(0, 1)  # Transpose back
            lstm_out = self.attention_norm(lstm_out + attn_out)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Global max pooling over sequence dimension
        global_max_pool, _ = torch.max(lstm_out, dim=1)
        
        # Process toxicity features
        feature_vec = self.feature_fc(toxicity_features)
        feature_vec = self.feature_bn(feature_vec)
        feature_vec = F.relu(feature_vec)
        feature_vec = self.feature_dropout(feature_vec)
        
        # Combine LSTM and feature outputs
        combined = torch.cat([global_max_pool, feature_vec], dim=1)
        
        # Apply additional non-linearity
        combined = self.combined_fc(combined)
        combined = self.combined_bn(combined)
        combined = F.gelu(combined)  # GELU for smoother gradients
        combined = self.dropout(combined)
        
        # Store extracted features for use by classifier chain
        self.extracted_features = combined
        
        # Final output layers
        toxicity_output = self.fc_toxicity(combined)
        category_output = self.fc_category(combined)
        
        return toxicity_output, category_output

# ==================================================================================
# Enhanced Classifier Chain Implementation
# ==================================================================================

class EnhancedClassifierChain(nn.Module):
    """
    Enhanced implementation of the classifier chain model for toxicity detection.
    
    Chain flow:
    1. Base Network -> Is text toxic? (yes/no)
    2. Base Network + Toxicity -> Which categories apply? (insult, profanity, threat, identity hate)
    3. Base Network + Toxicity + Categories -> What's the severity? (toxic/very toxic)
    """
    def __init__(self, base_model):
        """
        Initialize the enhanced classifier chain using a base model.
        
        Args:
            base_model: Base model for feature extraction
        """
        super(EnhancedClassifierChain, self).__init__()
        self.base_model = base_model
        
        # Get dimension of the base model's output
        # This should be the combined dimension from the base network
        if hasattr(base_model, 'combined_fc'):
            # If we can access the combined_fc layer, get its output dim
            combined_dim = base_model.combined_fc.out_features
        elif hasattr(base_model, 'fc_toxicity'):
            # Otherwise, get the input dim to the output layer
            combined_dim = base_model.fc_toxicity.in_features
        else:
            # Default fallback based on architecture
            lstm_hidden_dim = getattr(base_model, 'lstm_hidden_dim', 96)
            combined_dim = (lstm_hidden_dim * 2) + 64  # BiLSTM output + feature dimensions
        
        # Chain link 1: Binary toxicity classifier (toxic or not)
        self.toxicity_binary = nn.Linear(combined_dim, 1)
        # Initialize weights for better convergence
        nn.init.xavier_normal_(self.toxicity_binary.weight)
        nn.init.constant_(self.toxicity_binary.bias, -0.2)  # Negative bias to reduce false positives
        # Apply weight normalization
        self.toxicity_binary = nn.utils.parametrizations.weight_norm(self.toxicity_binary)
        
        # Chain link 2: Category classifiers with batch normalization
        # Input: base features + toxicity binary result
        self.bn_tox_feat = nn.BatchNorm1d(combined_dim + 1)
        
        self.category_insult = nn.Linear(combined_dim + 1, 1)
        nn.init.xavier_normal_(self.category_insult.weight)
        nn.init.constant_(self.category_insult.bias, -0.453)
        self.category_insult = nn.utils.parametrizations.weight_norm(self.category_insult)
        
        self.category_profanity = nn.Linear(combined_dim + 1, 1)
        nn.init.xavier_normal_(self.category_profanity.weight)
        nn.init.constant_(self.category_profanity.bias, -0.453)
        self.category_profanity = nn.utils.parametrizations.weight_norm(self.category_profanity)
        
        self.category_threat = nn.Linear(combined_dim + 1, 1)
        nn.init.xavier_normal_(self.category_threat.weight)
        nn.init.constant_(self.category_threat.bias, -0.554)  # Higher threshold for threats
        self.category_threat = nn.utils.parametrizations.weight_norm(self.category_threat)
        
        self.category_identity_hate = nn.Linear(combined_dim + 1, 1)
        nn.init.xavier_normal_(self.category_identity_hate.weight)
        nn.init.constant_(self.category_identity_hate.bias, -0.654)  # Higher threshold for identity hate
        self.category_identity_hate = nn.utils.parametrizations.weight_norm(self.category_identity_hate)
        
        # Chain link 3: Severity classifier with batch normalization
        # Input: base features + toxicity binary + all 4 categories
        self.bn_severity_feat = nn.BatchNorm1d(combined_dim + 1 + 4)
        
        self.severity = nn.Linear(combined_dim + 1 + 4, 1)
        nn.init.xavier_normal_(self.severity.weight)
        nn.init.constant_(self.severity.bias, -0.3)
        self.severity = nn.utils.parametrizations.weight_norm(self.severity)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, char_ids, toxicity_features=None):
        """
        Forward pass implementing the classifier chain.
        
        Args:
            char_ids: Character IDs input tensor
            toxicity_features: Additional toxicity features tensor
            
        Returns:
            Dictionary with all outputs from the chain
        """
        # Extract features from base model
        toxicity_output, category_output = self.base_model(char_ids, toxicity_features)
        
        # Get the combined features from the base model
        # Need to extract combined features which are the input to both outputs
        if hasattr(self.base_model, 'extracted_features'):
            # If the base model has a cached features attribute, use it
            base_features = self.base_model.extracted_features
        else:
            # Otherwise, approximate by getting toxicity features
            # and removing the final layer transformation
            combined_dim = self.toxicity_binary.in_features
            # This is a fallback method and less accurate
            with torch.no_grad():
                # Recreate base features by applying a pseudo-inverse operation
                # This is just a heuristic approximation
                fc_toxicity_pinv = torch.pinverse(self.base_model.fc_toxicity.weight)
                base_features = (toxicity_output - self.base_model.fc_toxicity.bias) @ fc_toxicity_pinv
        
        # Apply dropout to base features
        base_features = self.dropout(base_features)
        
        # ===== CLASSIFIER CHAIN IMPLEMENTATION =====
        
        # Chain link 1: Binary toxicity classification (is it toxic?)
        toxicity_bin_logits = self.toxicity_binary(base_features)
        toxicity_bin_probs = torch.sigmoid(toxicity_bin_logits)
        
        # For the chain, we use the probability as a feature for the next classifiers
        # Chain link 2: Category classification with toxicity information
        # Concatenate base features with toxicity binary prediction
        features_with_toxicity = torch.cat([base_features, toxicity_bin_probs], dim=1)
        features_with_toxicity = self.bn_tox_feat(features_with_toxicity)
        features_with_toxicity = self.dropout(features_with_toxicity)
        
        # Category classifiers
        insult_logits = self.category_insult(features_with_toxicity)
        profanity_logits = self.category_profanity(features_with_toxicity)
        threat_logits = self.category_threat(features_with_toxicity)
        identity_hate_logits = self.category_identity_hate(features_with_toxicity)
        
        # Get probabilities
        insult_probs = torch.sigmoid(insult_logits)
        profanity_probs = torch.sigmoid(profanity_logits)
        threat_probs = torch.sigmoid(threat_logits)
        identity_hate_probs = torch.sigmoid(identity_hate_logits)
        
        # Combine category probabilities for next chain link
        category_probs = torch.cat([
            insult_probs, profanity_probs, threat_probs, identity_hate_probs
        ], dim=1)
        
        # Chain link 3: Severity classification (toxic vs very toxic)
        # This only applies if the content is toxic
        # Concatenate base features with toxicity and all categories
        features_for_severity = torch.cat([base_features, toxicity_bin_probs, category_probs], dim=1)
        features_for_severity = self.bn_severity_feat(features_for_severity)
        features_for_severity = self.dropout(features_for_severity)
        
        severity_logits = self.severity(features_for_severity)
        severity_probs = torch.sigmoid(severity_logits)
        
        # Return all outputs from the chain
        return {
            'toxicity_binary': toxicity_bin_logits,  # Is it toxic at all? (binary)
            'toxicity_binary_probs': toxicity_bin_probs,
            'category_logits': {
                'insult': insult_logits,
                'profanity': profanity_logits,
                'threat': threat_logits,
                'identity_hate': identity_hate_logits
            },
            'category_probs': {
                'insult': insult_probs,
                'profanity': profanity_probs,
                'threat': threat_probs,
                'identity_hate': identity_hate_probs
            },
            'severity_logits': severity_logits,  # How severe is the toxicity?
            'severity_probs': severity_probs
        }
    
    def adjust_thresholds_for_context(self, thresholds, toxicity_features):
        """
        Dynamically adjust classification thresholds based on context.
        
        Args:
            thresholds: Dictionary of current thresholds
            toxicity_features: Features extracted from text
            
        Returns:
            Updated thresholds dictionary
        """
        try:
            # Create a copy of thresholds to modify
            adjusted_thresholds = thresholds.copy()
            
            # Get batch size
            batch_size = len(toxicity_features) if isinstance(toxicity_features, list) else toxicity_features.size(0)
            
            # Process each example in the batch
            for i in range(batch_size):
                # Extract features for this example
                if isinstance(toxicity_features, list):
                    features = toxicity_features[i]
                else:
                    # Get individual features from tensors
                    features = {
                        'safe_word_count': toxicity_features.get('safe_word_count', None)[i].item() if hasattr(toxicity_features, 'get') else 0,
                        'toxic_keyword_count': toxicity_features.get('toxic_keyword_count', None)[i].item() if hasattr(toxicity_features, 'get') else 0,
                        'is_educational': toxicity_features.get('is_educational', None)[i].item() if hasattr(toxicity_features, 'get') else False
                    }
                
                # 1. Adjust for safe words/educational content
                safe_word_count = features.get('safe_word_count', 0)
                is_educational = features.get('is_educational', False)
                
                if safe_word_count > 0 or is_educational:
                    # Scale boost based on context evidence
                    boost_amount = min(0.15 * (safe_word_count + int(is_educational)), 0.3)
                    
                    # Apply boost to thresholds
                    for key in ['toxicity', 'insult', 'profanity', 'threat', 'identity_hate']:
                        adjusted_thresholds[key] = min(0.95, adjusted_thresholds[key] + boost_amount)
                
                # 2. Adjust for high toxicity signal
                toxic_keyword_count = features.get('toxic_keyword_count', 0)
                if toxic_keyword_count > 3:  # Strong toxicity signal
                    reduction_amount = min(0.1 * (toxic_keyword_count / 5), 0.2)
                    
                    # Lower threshold for high toxicity content
                    for key in ['toxicity', 'insult', 'profanity']:
                        adjusted_thresholds[key] = max(0.4, adjusted_thresholds[key] - reduction_amount)
            
            return adjusted_thresholds
        
        except Exception as e:
            logger.error(f"Error adjusting thresholds: {e}")
            # Return original thresholds if anything fails
            return thresholds
    
    def predict(self, char_ids, toxicity_features=None, thresholds=None, 
                adjust_thresholds=True, language='en'):
        """
        Make predictions using the classifier chain with context-adjusted thresholds.
        
        Args:
            char_ids: Character IDs input tensor
            toxicity_features: Additional toxicity features tensor
            thresholds: Dictionary of thresholds for each classifier (optional)
            adjust_thresholds: Whether to dynamically adjust thresholds based on context
            language: Language of the text ('en' or 'tl')
            
        Returns:
            Dictionary with final predictions
        """
        # Get language-specific thresholds if available
        if thresholds is None:
            if language in Config.LANGUAGE_THRESHOLDS:
                thresholds = Config.LANGUAGE_THRESHOLDS[language]
            else:
                thresholds = Config.LANGUAGE_THRESHOLDS['en']  # Default to English
        
        # Context-aware threshold adjustment
        if adjust_thresholds and toxicity_features is not None:
            thresholds = self.adjust_thresholds_for_context(thresholds, toxicity_features)
        
        # Get raw outputs
        outputs = self.forward(char_ids, toxicity_features)
        
        # Apply thresholds for final predictions
        is_toxic = (outputs['toxicity_binary_probs'] > thresholds['toxicity']).float()
        
        # FIXED: Lower thresholds for category detection during validation phase
        # Using significantly lower thresholds for categories to ensure some positive predictions
        # during early training phases
        category_thresholds = {
            'insult': thresholds.get('insult', 0.6),
            'profanity': thresholds.get('profanity', 0.6),
            'threat': thresholds.get('threat', 0.5),
            'identity_hate': thresholds.get('identity_hate', 0.5)
        }
        
        # Apply category thresholds
        insult = (outputs['category_probs']['insult'] > category_thresholds['insult']).float()
        profanity = (outputs['category_probs']['profanity'] > category_thresholds['profanity']).float()
        threat = (outputs['category_probs']['threat'] > category_thresholds['threat']).float()
        identity_hate = (outputs['category_probs']['identity_hate'] > category_thresholds['identity_hate']).float()
        
        # Debug: Check if we're predicting any positives
        # This can help diagnose whether the issue is with predictions or with calculation
        total_positives = insult.sum().item() + profanity.sum().item() + threat.sum().item() + identity_hate.sum().item()
        if hasattr(logger, 'debug'):
            logger.debug(f"Total positive category predictions: {total_positives}")
        
        # Enforce consistency: if not toxic, no categories should be positive
        insult = insult * is_toxic
        profanity = profanity * is_toxic
        threat = threat * is_toxic
        identity_hate = identity_hate * is_toxic
        
        # Determine severity
        # If not toxic at all, severity is 0
        # If toxic but severity below threshold, it's level 1 (toxic)
        # If toxic and severity above threshold, it's level 2 (very toxic)
        severity = (outputs['severity_probs'] > thresholds['severity']).float()
        
        # Determine final toxicity level
        # 0 = not toxic, 1 = toxic, 2 = very toxic
        toxicity_level = torch.zeros_like(is_toxic, dtype=torch.long)
        toxicity_level[is_toxic.squeeze() == 1] = 1  # Toxic
        toxicity_level[torch.logical_and(is_toxic.squeeze() == 1, severity.squeeze() == 1)] = 2  # Very toxic
        
        return {
            'toxicity_level': toxicity_level,
            'categories': {
                'insult': insult,
                'profanity': profanity,
                'threat': threat,
                'identity_hate': identity_hate
            },
            'probabilities': {
                'toxicity': outputs['toxicity_binary_probs'],
                'insult': outputs['category_probs']['insult'],
                'profanity': outputs['category_probs']['profanity'],
                'threat': outputs['category_probs']['threat'],
                'identity_hate': outputs['category_probs']['identity_hate'],
                'severity': outputs['severity_probs']
            },
            'applied_thresholds': thresholds
        }

# ==================================================================================
# Monte Carlo Dropout Uncertainty Estimation
# ==================================================================================

class MCDropoutChainModel(nn.Module):
    """
    Wrapper for EnhancedClassifierChain to enable Monte Carlo Dropout at inference time.
    Used for uncertainty estimation through multiple forward passes.
    """
    def __init__(self, chain_model):
        super(MCDropoutChainModel, self).__init__()
        self.chain_model = chain_model
    
    def forward(self, char_ids, toxicity_features=None):
        return self.chain_model(char_ids, toxicity_features)
    
    def enable_dropout(self):
        """Enable dropout layers during inference."""
        # Enable dropout in the base model
        for module in self.chain_model.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                
        # Also enable dropout in the classifier chain
        for module in self.chain_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(self, char_ids, toxicity_features=None, num_samples=30, 
                              thresholds=None, language='en'):
        """
        Run multiple forward passes with dropout enabled to estimate uncertainty.
        
        Args:
            char_ids: Character IDs input tensor
            toxicity_features: Additional toxicity features tensor  
            num_samples: Number of Monte Carlo samples
            thresholds: Dictionary of thresholds for classification
            language: Language of the text ('en' or 'tl')
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        self.eval()  # Set model to evaluation mode
        self.enable_dropout()  # But enable dropout
        
        # Get language-specific thresholds if available
        if thresholds is None:
            if language in Config.LANGUAGE_THRESHOLDS:
                thresholds = Config.LANGUAGE_THRESHOLDS[language]
            else:
                thresholds = Config.LANGUAGE_THRESHOLDS['en']  # Default to English
        
        # Adjust thresholds for context (use first sample only)
        if toxicity_features is not None:
            thresholds = self.chain_model.adjust_thresholds_for_context(thresholds, toxicity_features)
        
        # Storage for samples
        toxicity_probs_samples = []
        insult_probs_samples = []
        profanity_probs_samples = []
        threat_probs_samples = []
        identity_hate_probs_samples = []
        severity_probs_samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Forward pass with dropout active
                outputs = self.chain_model(char_ids, toxicity_features)
                
                # Store probability samples
                toxicity_probs_samples.append(outputs['toxicity_binary_probs'])
                insult_probs_samples.append(outputs['category_probs']['insult'])
                profanity_probs_samples.append(outputs['category_probs']['profanity'])
                threat_probs_samples.append(outputs['category_probs']['threat'])
                identity_hate_probs_samples.append(outputs['category_probs']['identity_hate'])
                severity_probs_samples.append(outputs['severity_probs'])
        
        # Stack all samples
        toxicity_probs_samples = torch.stack(toxicity_probs_samples)  # [num_samples, batch_size, 1]
        insult_probs_samples = torch.stack(insult_probs_samples)
        profanity_probs_samples = torch.stack(profanity_probs_samples)
        threat_probs_samples = torch.stack(threat_probs_samples)
        identity_hate_probs_samples = torch.stack(identity_hate_probs_samples)
        severity_probs_samples = torch.stack(severity_probs_samples)
        
        # Mean predictions
        mean_toxicity_probs = toxicity_probs_samples.mean(dim=0)
        mean_insult_probs = insult_probs_samples.mean(dim=0)
        mean_profanity_probs = profanity_probs_samples.mean(dim=0)
        mean_threat_probs = threat_probs_samples.mean(dim=0)
        mean_identity_hate_probs = identity_hate_probs_samples.mean(dim=0)
        mean_severity_probs = severity_probs_samples.mean(dim=0)
        
        # Standard deviation (uncertainty)
        toxicity_uncertainty = toxicity_probs_samples.std(dim=0)
        insult_uncertainty = insult_probs_samples.std(dim=0)
        profanity_uncertainty = profanity_probs_samples.std(dim=0)
        threat_uncertainty = threat_probs_samples.std(dim=0)
        identity_hate_uncertainty = identity_hate_probs_samples.std(dim=0)
        severity_uncertainty = severity_probs_samples.std(dim=0)
        
        # Predictive entropy for overall uncertainty
        toxicity_entropy = -mean_toxicity_probs * torch.log(mean_toxicity_probs + 1e-10) - \
                           (1 - mean_toxicity_probs) * torch.log(1 - mean_toxicity_probs + 1e-10)
        
        # Apply thresholds to mean predictions
        is_toxic = (mean_toxicity_probs > thresholds['toxicity']).float()
        
        insult = (mean_insult_probs > thresholds['insult']).float()
        profanity = (mean_profanity_probs > thresholds['profanity']).float()
        threat = (mean_threat_probs > thresholds['threat']).float()
        identity_hate = (mean_identity_hate_probs > thresholds['identity_hate']).float()
        
        # Enforce consistency: if not toxic, no categories
        insult = insult * is_toxic
        profanity = profanity * is_toxic
        threat = threat * is_toxic
        identity_hate = identity_hate * is_toxic
        
        # Determine severity
        severity = (mean_severity_probs > thresholds['severity']).float()
        
        # Determine final toxicity level
        toxicity_level = torch.zeros_like(is_toxic, dtype=torch.long)
        toxicity_level[is_toxic.squeeze() == 1] = 1  # Toxic
        toxicity_level[torch.logical_and(is_toxic.squeeze() == 1, severity.squeeze() == 1)] = 2  # Very toxic
        
        # Calculate reliability score (inverse of uncertainty)
        reliability = 1.0 - (toxicity_entropy / 0.693)  # 0.693 is max binary entropy
        
        return {
            'toxicity_level': toxicity_level,
            'categories': {
                'insult': insult,
                'profanity': profanity,
                'threat': threat,
                'identity_hate': identity_hate
            },
            'probabilities': {
                'toxicity': mean_toxicity_probs,
                'insult': mean_insult_probs,
                'profanity': mean_profanity_probs, 
                'threat': mean_threat_probs,
                'identity_hate': mean_identity_hate_probs,
                'severity': mean_severity_probs
            },
            'uncertainty': {
                'toxicity': toxicity_uncertainty,
                'insult': insult_uncertainty,
                'profanity': profanity_uncertainty,
                'threat': threat_uncertainty,
                'identity_hate': identity_hate_uncertainty,
                'severity': severity_uncertainty,
                'overall': toxicity_entropy
            },
            'reliability': reliability,
            'applied_thresholds': thresholds
        }

# ==================================================================================
# Enhanced Training Functions
# ==================================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    gamma: focusing parameter that reduces the loss contribution from easy examples
    alpha: weighting factor for dealing with class imbalance
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Focal weighting
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weighting if alpha is provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def train_enhanced_classifier_chain(model, train_loader, val_loader, 
                                   num_epochs=40, learning_rate=0.0005, 
                                   weight_decay=1e-4, early_stopping_patience=6,
                                   label_smoothing=0.1):
    """
    Train the enhanced classifier chain model with advanced training techniques.
    
    Args:
        model: The classifier chain model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        early_stopping_patience: Patience for early stopping
        label_smoothing: Label smoothing factor (0.0 to disable)
        
    Returns:
        Trained model, best validation metrics, and metrics history
    """
    logger.info(f"Training enhanced classifier chain model for {num_epochs} epochs")
    logger.info(f"Parameters: lr={learning_rate}, weight_decay={weight_decay}, "
               f"early_stopping_patience={early_stopping_patience}, "
               f"label_smoothing={label_smoothing}")
    
    # Get device
    device = next(model.parameters()).device
    
    # Setup loss functions
    if Config.USE_LABEL_SMOOTHING and label_smoothing > 0:
        # Binary cross-entropy with label smoothing
        def binary_cross_entropy_with_smoothing(pred, target, smoothing=label_smoothing):
            # Apply label smoothing
            target = target * (1 - smoothing) + 0.5 * smoothing
            # Use binary cross entropy with logits
            return F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        
        toxicity_criterion = binary_cross_entropy_with_smoothing
    else:
        # Standard binary cross-entropy
        toxicity_criterion = nn.BCEWithLogitsLoss()
    
    # Category loss functions with focal loss to handle imbalance
    alpha_values = torch.tensor(Config.CATEGORY_WEIGHTS, device=device) / sum(Config.CATEGORY_WEIGHTS)
    category_criteria = {
        'insult': FocalLoss(gamma=2.0, alpha=0.853),
        'profanity': FocalLoss(gamma=2.0, alpha=0.9),
        'threat': FocalLoss(gamma=2.0, alpha=alpha_values[2]),
        'identity_hate': FocalLoss(gamma=2.0, alpha=alpha_values[3])
    }
    
    # Severity loss
    severity_criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    if Config.USE_ONE_CYCLE_LR:
        # One-cycle learning rate schedule
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=0.3,  # Warm-up for 30% of training
            div_factor=25,  # Initial LR is max_lr/25
            final_div_factor=1000  # Final LR is max_lr/1000
        )
    else:
        # Traditional reduce on plateau
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    
    # Track best model
    best_val_loss = float('inf')
    best_model_state = None
    best_val_metrics = None
    patience_counter = 0
    
    # Track metrics
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'val_toxicity_acc': [],
        'val_category_f1': [],
        'learning_rates': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        metrics_history['learning_rates'].append(current_lr)
        
        # Training phase
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            char_ids = batch['char_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Extract labels:
            # toxicity_level is labels[:, 0]
            # categories are labels[:, 1:5]
            toxicity_level = labels[:, 0].long()
            categories = labels[:, 1:5]
            
            # Convert toxicity level to necessary formats for chain:
            # 1. Binary toxicity (0 = not toxic, 1 = toxic or very toxic)
            binary_toxicity = (toxicity_level > 0).float().unsqueeze(1)
            
            # 2. Severity (0 = toxic, 1 = very toxic) - only for toxic items
            severity = (toxicity_level == 2).float().unsqueeze(1)
            
            # Get toxicity features
            toxicity_features = torch.stack([
                batch['all_caps_ratio'],
                batch['toxic_keyword_count'],
                batch['toxic_keyword_ratio'],
                batch['safe_word_count'],
                batch['safe_word_ratio'],
                batch['special_char_ratio'],
                batch['is_educational'],
                batch['text_length'] / 300.0  # Normalize by max length
            ], dim=1).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(char_ids, toxicity_features)
            
            # Calculate losses
            
            # 1. Toxicity binary loss
            toxicity_binary_loss = toxicity_criterion(outputs['toxicity_binary'], binary_toxicity)
            
            # 2. Category losses - calculate for all, but weight by toxicity
            cat_insult_loss = category_criteria['insult'](
                outputs['category_logits']['insult'], 
                categories[:, 0:1]
            )
            
            cat_profanity_loss = category_criteria['profanity'](
                outputs['category_logits']['profanity'], 
                categories[:, 1:2]
            )
            
            cat_threat_loss = category_criteria['threat'](
                outputs['category_logits']['threat'], 
                categories[:, 2:3]
            )
            
            cat_identity_hate_loss = category_criteria['identity_hate'](
                outputs['category_logits']['identity_hate'], 
                categories[:, 3:4]
            )
            
            category_loss = cat_insult_loss + cat_profanity_loss + cat_threat_loss + cat_identity_hate_loss
            
            # 3. Severity loss - only meaningful for toxic items
            # Create a mask for toxic items
            toxic_mask = binary_toxicity.bool()
            
            if toxic_mask.sum() > 0:  # If there are toxic items in batch
                severity_loss = severity_criterion(
                    outputs['severity_logits'][toxic_mask],
                    severity[toxic_mask]
                )
            else:
                severity_loss = torch.tensor(0.0, device=device)
            
            # Combined loss - weight toxicity classification more heavily
            loss = (1.5 * toxicity_binary_loss + 
                   Config.CATEGORY_LOSS_SCALE * category_loss + 
                   severity_loss)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            if Config.USE_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=Config.GRADIENT_CLIP_VALUE
                )
            
            optimizer.step()
            
            # Update learning rate scheduler if using OneCycleLR
            if Config.USE_ONE_CYCLE_LR:
                scheduler.step()
            
            # Track loss
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), lr=current_lr)
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_toxicity_preds = []
        val_toxicity_labels = []
        val_category_preds = []
        val_category_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                char_ids = batch['char_ids'].to(device)
                labels = batch['labels'].to(device)
                
                # Extract labels
                toxicity_level = labels[:, 0].long()
                categories = labels[:, 1:5]
                
                # Convert toxicity level to necessary formats for chain
                binary_toxicity = (toxicity_level > 0).float().unsqueeze(1)
                severity = (toxicity_level == 2).float().unsqueeze(1)
                
                # Get toxicity features
                toxicity_features = torch.stack([
                    batch['all_caps_ratio'],
                    batch['toxic_keyword_count'],
                    batch['toxic_keyword_ratio'],
                    batch['safe_word_count'],
                    batch['safe_word_ratio'],
                    batch['special_char_ratio'],
                    batch['is_educational'],
                    batch['text_length'] / 300.0  # Normalize by max length
                ], dim=1).to(device)
                
                # Forward pass
                outputs = model(char_ids, toxicity_features)
                
                # Calculate losses (same as training)
                toxicity_binary_loss = toxicity_criterion(outputs['toxicity_binary'], binary_toxicity)
                
                cat_insult_loss = category_criteria['insult'](
                    outputs['category_logits']['insult'], 
                    categories[:, 0:1]
                )
                
                cat_profanity_loss = category_criteria['profanity'](
                    outputs['category_logits']['profanity'], 
                    categories[:, 1:2]
                )
                
                cat_threat_loss = category_criteria['threat'](
                    outputs['category_logits']['threat'], 
                    categories[:, 2:3]
                )
                
                cat_identity_hate_loss = category_criteria['identity_hate'](
                    outputs['category_logits']['identity_hate'], 
                    categories[:, 3:4]
                )
                
                category_loss = cat_insult_loss + cat_profanity_loss + cat_threat_loss + cat_identity_hate_loss
                
                # Severity loss - only for toxic items
                toxic_mask = binary_toxicity.bool()
                
                if toxic_mask.sum() > 0:
                    severity_loss = severity_criterion(
                        outputs['severity_logits'][toxic_mask],
                        severity[toxic_mask]
                    )
                else:
                    severity_loss = torch.tensor(0.0, device=device)
                
                # Combined loss
                loss = (1.5 * toxicity_binary_loss + 
                       Config.CATEGORY_LOSS_SCALE * category_loss + 
                       severity_loss)
                
                # Track loss
                val_loss += loss.item()
                
                # Make predictions
                predictions = model.predict(char_ids, toxicity_features)
                
                # Track toxicity predictions
                val_toxicity_preds.extend(predictions['toxicity_level'].cpu().numpy())
                val_toxicity_labels.extend(toxicity_level.cpu().numpy())
                
                # Track category predictions - FIXED: Properly collect category predictions and labels
                batch_category_preds = torch.stack([
                    predictions['categories']['insult'],
                    predictions['categories']['profanity'],
                    predictions['categories']['threat'],
                    predictions['categories']['identity_hate']
                ], dim=1).cpu().numpy()
                
                val_category_preds.extend(batch_category_preds)
                val_category_labels.extend(categories.cpu().numpy())
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate validation metrics
        val_toxicity_acc = accuracy_score(val_toxicity_labels, val_toxicity_preds)
        
        # FIXED: Calculate category F1 scores correctly
        val_category_f1 = []
        try:
            # Convert to numpy arrays
            category_labels_array = np.array(val_category_labels)
            category_preds_array = np.array(val_category_preds)
            
            logger.info(f"Category predictions shape: {category_preds_array.shape}")
            logger.info(f"Category labels shape: {category_labels_array.shape}")
            
            # Calculate F1 for each category
            for i in range(category_labels_array.shape[1]):
                # Count positive examples
                pos_labels = np.sum(category_labels_array[:, i])
                pos_preds = np.sum(category_preds_array[:, i])
                logger.info(f"Category {i}: positive predictions: {pos_preds}, positive labels: {pos_labels}")
                
                # Set a lower threshold for debugging
                # This is temporary to debug the F1 calculation - can be removed later
                modified_preds = category_preds_array[:, i].astype(float)
                
                # Only calculate F1 if there are positive examples
                if pos_labels > 0:
                    cat_f1 = f1_score(category_labels_array[:, i], modified_preds, zero_division=0)
                    val_category_f1.append(cat_f1)
                    logger.info(f"Category {i} F1: {cat_f1:.4f}")
            
            # Calculate macro-average F1
            val_category_macro_f1 = np.mean(val_category_f1) if val_category_f1 else 0.0
            logger.info(f"Category F1 scores: {val_category_f1}, Macro F1: {val_category_macro_f1:.4f}")
        except Exception as e:
            logger.error(f"Error in F1 calculation: {e}")
            import traceback
            traceback.print_exc()
            val_category_macro_f1 = 0.0
        
        # Update learning rate scheduler if using ReduceLROnPlateau
        if not Config.USE_ONE_CYCLE_LR:
            scheduler.step(avg_val_loss)
        
        # Track metrics
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(avg_val_loss)
        metrics_history['val_toxicity_acc'].append(val_toxicity_acc)
        metrics_history['val_category_f1'].append(val_category_macro_f1)
        
        # Print progress
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, "
                   f"Val Toxicity Acc: {val_toxicity_acc:.4f}, "
                   f"Val Category F1: {val_category_macro_f1:.4f}, "
                   f"LR: {current_lr:.6f}, "
                   f"Time: {epoch_time:.2f}s")
        
        # Check if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_metrics = {
                'loss': avg_val_loss,
                'toxicity_acc': val_toxicity_acc,
                'category_f1': val_category_macro_f1,
                'epoch': epoch + 1
            }
            patience_counter = 0
            logger.info(f"New best model found at epoch {epoch+1}")
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model state
    logger.info(f"Loading best model from epoch {best_val_metrics['epoch']}")
    model.load_state_dict(best_model_state)
    
    return model, best_val_metrics, metrics_history

# ==================================================================================
# Comprehensive Evaluation Functions
# ==================================================================================

# Fix for the "too many indices for array" error

def evaluate_model(model, dataloader, mc_dropout=False, num_mc_samples=20):
    """
    Comprehensive evaluation of the model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        mc_dropout: Whether to use Monte Carlo dropout for uncertainty estimation
        num_mc_samples: Number of MC samples if using dropout
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Create MC dropout wrapper if needed
    if mc_dropout:
        logger.info(f"Using Monte Carlo dropout with {num_mc_samples} samples")
        mc_model = MCDropoutChainModel(model)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Track predictions and labels
    all_texts = []
    all_toxicity_preds = []
    all_toxicity_labels = []
    all_category_preds = []
    all_category_labels = []
    all_probs = {
        'toxicity': [],
        'insult': [],
        'profanity': [],
        'threat': [],
        'identity_hate': []
    }
    all_uncertainties = {
        'toxicity': [],
        'overall': []
    } if mc_dropout else None
    
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            char_ids = batch['char_ids'].to(device)
            labels = batch['labels'].to(device)
            texts = batch['text']
            
            # Extract labels
            toxicity_level = labels[:, 0].long()
            categories = labels[:, 1:5]
            
            # Get toxicity features
            toxicity_features = torch.stack([
                batch['all_caps_ratio'],
                batch['toxic_keyword_count'],
                batch['toxic_keyword_ratio'],
                batch['safe_word_count'],
                batch['safe_word_ratio'],
                batch['special_char_ratio'],
                batch['is_educational'],
                batch['text_length'] / 300.0  # Normalize by max length
            ], dim=1).to(device)
            
            # Get language info if available
            languages = []
            for i in range(len(texts)):
                lang = 'en'  # Default to English
                if 'language' in batch and i < len(batch):
                    if isinstance(batch['language'], list):
                        lang = batch['language'][i]
                    else:
                        lang = batch['language']
                languages.append(lang)
            
            # Make predictions
            if mc_dropout:
                batch_predictions = []
                for i, lang in enumerate(languages):
                    # Process each example individually to use correct language thresholds
                    single_char_ids = char_ids[i:i+1]
                    single_features = toxicity_features[i:i+1]
                    
                    predictions = mc_model.predict_with_uncertainty(
                        single_char_ids, 
                        single_features, 
                        num_samples=num_mc_samples,
                        language=lang
                    )
                    batch_predictions.append(predictions)
                
                # Concatenate batch results
                toxicity_preds = torch.cat([pred['toxicity_level'] for pred in batch_predictions])
                insult_preds = torch.cat([pred['categories']['insult'] for pred in batch_predictions])
                profanity_preds = torch.cat([pred['categories']['profanity'] for pred in batch_predictions])
                threat_preds = torch.cat([pred['categories']['threat'] for pred in batch_predictions])
                identity_hate_preds = torch.cat([pred['categories']['identity_hate'] for pred in batch_predictions])
                
                # Track probabilities
                toxicity_probs = torch.cat([pred['probabilities']['toxicity'] for pred in batch_predictions])
                insult_probs = torch.cat([pred['probabilities']['insult'] for pred in batch_predictions])
                profanity_probs = torch.cat([pred['probabilities']['profanity'] for pred in batch_predictions])
                threat_probs = torch.cat([pred['probabilities']['threat'] for pred in batch_predictions])
                identity_hate_probs = torch.cat([pred['probabilities']['identity_hate'] for pred in batch_predictions])
                
                # Track uncertainties
                toxicity_uncertainty = torch.cat([pred['uncertainty']['toxicity'] for pred in batch_predictions])
                overall_uncertainty = torch.cat([pred['uncertainty']['overall'] for pred in batch_predictions])
                
            else:
                # Make predictions with standard method
                batch_predictions = []
                for i, lang in enumerate(languages):
                    # Process each example individually to use correct language thresholds
                    single_char_ids = char_ids[i:i+1]
                    single_features = toxicity_features[i:i+1]
                    
                    predictions = model.predict(
                        single_char_ids, 
                        single_features,
                        language=lang
                    )
                    batch_predictions.append(predictions)
                
                # Concatenate batch results
                toxicity_preds = torch.cat([pred['toxicity_level'] for pred in batch_predictions])
                insult_preds = torch.cat([pred['categories']['insult'] for pred in batch_predictions])
                profanity_preds = torch.cat([pred['categories']['profanity'] for pred in batch_predictions])
                threat_preds = torch.cat([pred['categories']['threat'] for pred in batch_predictions])
                identity_hate_preds = torch.cat([pred['categories']['identity_hate'] for pred in batch_predictions])
                
                # Track probabilities
                toxicity_probs = torch.cat([pred['probabilities']['toxicity'] for pred in batch_predictions])
                insult_probs = torch.cat([pred['probabilities']['insult'] for pred in batch_predictions])
                profanity_probs = torch.cat([pred['probabilities']['profanity'] for pred in batch_predictions])
                threat_probs = torch.cat([pred['probabilities']['threat'] for pred in batch_predictions])
                identity_hate_probs = torch.cat([pred['probabilities']['identity_hate'] for pred in batch_predictions])
            
            # Track predictions and labels
            all_texts.extend(texts)
            all_toxicity_preds.extend(toxicity_preds.cpu().numpy())
            all_toxicity_labels.extend(toxicity_level.cpu().numpy())
            
            # Track category predictions
            batch_category_preds = torch.stack([
                insult_preds,
                profanity_preds,
                threat_preds,
                identity_hate_preds
            ], dim=1).cpu().numpy()
            
            all_category_preds.extend(batch_category_preds)
            all_category_labels.extend(categories.cpu().numpy())
            
            # Track probabilities - Ensure we're working with 1D arrays
            for prob, prob_tensor in zip(
                ['toxicity', 'insult', 'profanity', 'threat', 'identity_hate'],
                [toxicity_probs, insult_probs, profanity_probs, threat_probs, identity_hate_probs]
            ):
                # Convert to numpy and flatten if needed
                prob_np = prob_tensor.cpu().numpy()
                if len(prob_np.shape) > 1:
                    prob_np = prob_np.flatten()
                all_probs[prob].extend(prob_np)
            
            # Track uncertainties if using MC dropout
            if mc_dropout:
                tox_unc_np = toxicity_uncertainty.cpu().numpy()
                if len(tox_unc_np.shape) > 1:
                    tox_unc_np = tox_unc_np.flatten()
                
                overall_unc_np = overall_uncertainty.cpu().numpy()
                if len(overall_unc_np.shape) > 1:
                    overall_unc_np = overall_unc_np.flatten()
                
                all_uncertainties['toxicity'].extend(tox_unc_np)
                all_uncertainties['overall'].extend(overall_unc_np)
    
    # Convert lists to arrays
    all_toxicity_preds = np.array(all_toxicity_preds)
    all_toxicity_labels = np.array(all_toxicity_labels)
    
    # Ensure we're working with properly shaped arrays for category predictions
    all_category_preds = np.array(all_category_preds)
    all_category_labels = np.array(all_category_labels)
    
    # Make sure all_category_preds has the right shape (n_samples, n_categories)
    if len(all_category_preds.shape) == 1:
        # If it's 1D, try to reshape based on the number of categories
        n_categories = 4  # insult, profanity, threat, identity_hate
        all_category_preds = all_category_preds.reshape(-1, n_categories)
    
    # Make sure all_category_labels has the right shape
    if len(all_category_labels.shape) == 1:
        # If it's 1D, try to reshape based on the number of categories
        n_categories = 4  # insult, profanity, threat, identity_hate
        all_category_labels = all_category_labels.reshape(-1, n_categories)
    
    # Ensure probs are properly shaped 1D arrays
    for key in all_probs:
        all_probs[key] = np.array(all_probs[key])
        # Flatten any multi-dimensional arrays
        if len(all_probs[key].shape) > 1:
            all_probs[key] = all_probs[key].flatten()
    
    # Ensure uncertainties are properly shaped 1D arrays
    if all_uncertainties:
        for key in all_uncertainties:
            all_uncertainties[key] = np.array(all_uncertainties[key])
            # Flatten any multi-dimensional arrays
            if len(all_uncertainties[key].shape) > 1:
                all_uncertainties[key] = all_uncertainties[key].flatten()
    
    # Calculate metrics
    toxicity_accuracy = accuracy_score(all_toxicity_labels, all_toxicity_preds)
    
    # Calculate toxicity class metrics
    try:
        toxicity_report = classification_report(
            all_toxicity_labels, all_toxicity_preds, 
            labels=[0, 1, 2],
            target_names=['Not Toxic', 'Toxic', 'Very Toxic'],
            output_dict=True,
            zero_division=0
        )
    except Exception as e:
        logger.error(f"Error in classification report: {e}")
        # Fallback if classification_report fails
        toxicity_report = {
            'Not Toxic': {'precision': 0, 'recall': 0, 'f1-score': 0},
            'Toxic': {'precision': 0, 'recall': 0, 'f1-score': 0},
            'Very Toxic': {'precision': 0, 'recall': 0, 'f1-score': 0}
        }
    
    # Build confusion matrix
    try:
        confusion_mat = np.zeros((3, 3), dtype=int)
        for true_label, pred_label in zip(all_toxicity_labels, all_toxicity_preds):
            if 0 <= true_label < 3 and 0 <= pred_label < 3:  # Ensure valid indices
                confusion_mat[int(true_label), int(pred_label)] += 1
    except Exception as e:
        logger.error(f"Error building confusion matrix: {e}")
        confusion_mat = np.zeros((3, 3), dtype=int)
    
    # Calculate category metrics
    category_metrics = {}
    category_columns = Config.CATEGORY_COLUMNS
    
    # Ensure we have correct dimensions for category calculations
    category_f1_scores = []
    
    for i, category in enumerate(category_columns):
        if i < all_category_preds.shape[1] and i < all_category_labels.shape[1]:
            try:
                # Extract 1D arrays for this category
                cat_preds = all_category_preds[:, i]
                cat_labels = all_category_labels[:, i]
                
                # Ensure they're flattened
                cat_preds = cat_preds.flatten()
                cat_labels = cat_labels.flatten()
                
                category_report = classification_report(
                    cat_labels, cat_preds,
                    target_names=[f'Non-{category}', category],
                    output_dict=True,
                    zero_division=0
                )
                category_metrics[category] = category_report
                
                # Check if there are any positive examples
                if np.sum(cat_labels) > 0:
                    cat_f1 = f1_score(cat_labels, cat_preds, zero_division=0)
                    category_f1_scores.append(cat_f1)
            except Exception as e:
                logger.error(f"Error in category metrics for {category}: {e}")
                # Fallback
                category_metrics[category] = {
                    category: {'precision': 0, 'recall': 0, 'f1-score': 0},
                    f'Non-{category}': {'precision': 0, 'recall': 0, 'f1-score': 0}
                }
    
    category_macro_f1 = np.mean(category_f1_scores) if category_f1_scores else 0.0
    
    # Analyze error patterns - pass flattened arrays
    error_analysis = analyze_errors(
        all_texts, all_toxicity_preds, all_toxicity_labels, 
        all_category_preds, all_category_labels,
        all_probs, all_uncertainties
    )
    
    # Print detailed results
    logger.info("\nEvaluation Results:")
    logger.info(f"Toxicity Classification Accuracy: {toxicity_accuracy:.4f}")
    logger.info(f"Category Macro-Average F1: {category_macro_f1:.4f}")
    
    logger.info("\nConfusion Matrix (rows=true, columns=predicted):")
    logger.info("              | Pred Not Toxic | Pred Toxic | Pred Very Toxic |")
    logger.info("--------------+----------------+------------+-----------------|")
    logger.info(f"True Not Toxic | {confusion_mat[0, 0]:14d} | {confusion_mat[0, 1]:10d} | {confusion_mat[0, 2]:15d} |")
    logger.info(f"True Toxic     | {confusion_mat[1, 0]:14d} | {confusion_mat[1, 1]:10d} | {confusion_mat[1, 2]:15d} |")
    logger.info(f"True Very Toxic| {confusion_mat[2, 0]:14d} | {confusion_mat[2, 1]:10d} | {confusion_mat[2, 2]:15d} |")
    
    logger.info("\nToxicity Class Metrics:")
    for cls in ['Not Toxic', 'Toxic', 'Very Toxic']:
        if cls in toxicity_report:
            logger.info(f"  {cls}:")
            logger.info(f"    Precision: {toxicity_report[cls]['precision']:.4f}")
            logger.info(f"    Recall: {toxicity_report[cls]['recall']:.4f}")
            logger.info(f"    F1-score: {toxicity_report[cls]['f1-score']:.4f}")
    
    logger.info("\nCategory Metrics:")
    for category in category_columns:
        if category in category_metrics and category in category_metrics[category]:
            logger.info(f"  {category.capitalize()}:")
            logger.info(f"    Precision: {category_metrics[category][category]['precision']:.4f}")
            logger.info(f"    Recall: {category_metrics[category][category]['recall']:.4f}")
            logger.info(f"    F1-score: {category_metrics[category][category]['f1-score']:.4f}")
    
    # Return comprehensive evaluation results
    return {
        'accuracy': toxicity_accuracy,
        'toxicity_report': toxicity_report,
        'confusion_matrix': confusion_mat,
        'category_metrics': category_metrics,
        'category_macro_f1': category_macro_f1,
        'all_predictions': {
            'texts': all_texts,
            'toxicity_preds': all_toxicity_preds,
            'toxicity_labels': all_toxicity_labels,
            'category_preds': all_category_preds,
            'category_labels': all_category_labels,
            'probabilities': all_probs,
            'uncertainties': all_uncertainties
        },
        'error_analysis': error_analysis
    }


def analyze_errors(texts, toxicity_preds, toxicity_labels, category_preds, category_labels, 
                 probabilities, uncertainties=None):
    """
    Analyze error patterns in model predictions.
    
    Args:
        texts: List of input texts
        toxicity_preds: Array of toxicity predictions
        toxicity_labels: Array of true toxicity labels
        category_preds: Array of category predictions
        category_labels: Array of true category labels
        probabilities: Dictionary of prediction probabilities
        uncertainties: Dictionary of prediction uncertainties (optional)
        
    Returns:
        Dictionary with error analysis results
    """
    # Convert inputs to numpy arrays if they're not already
    toxicity_preds = np.array(toxicity_preds)
    toxicity_labels = np.array(toxicity_labels)
    category_preds = np.array(category_preds)
    category_labels = np.array(category_labels)
    
    # Ensure all arrays are properly shaped
    # Flatten 1D arrays to ensure consistent indexing
    toxicity_preds = toxicity_preds.flatten()
    toxicity_labels = toxicity_labels.flatten()
    
    # Ensure category arrays have shape (n_samples, n_categories)
    if len(category_preds.shape) == 1:
        n_categories = 4  # insult, profanity, threat, identity_hate
        try:
            category_preds = category_preds.reshape(-1, n_categories)
        except:
            # If reshaping fails, create an empty array with the right shape
            category_preds = np.zeros((len(toxicity_preds), n_categories))
    
    if len(category_labels.shape) == 1:
        n_categories = 4  # insult, profanity, threat, identity_hate
        try:
            category_labels = category_labels.reshape(-1, n_categories)
        except:
            # If reshaping fails, create an empty array with the right shape
            category_labels = np.zeros((len(toxicity_labels), n_categories))
    
    # Find correct and incorrect predictions
    correct_mask = (toxicity_preds == toxicity_labels)
    incorrect_mask = ~correct_mask
    
    # Calculate error rate
    error_rate = np.mean(incorrect_mask)
    
    # Find false positives and false negatives
    fp_mask = (toxicity_preds > 0) & (toxicity_labels == 0)  # Predicted toxic, actually not
    fn_mask = (toxicity_preds == 0) & (toxicity_labels > 0)  # Predicted not toxic, actually toxic
    
    # Calculate error rates
    fp_rate = np.mean(fp_mask)
    fn_rate = np.mean(fn_mask)
    
    # Convert probabilities to numpy arrays for easier processing
    # Handle potentially nested arrays and ensure 1D
    tox_probs = np.array(probabilities['toxicity'])
    if len(tox_probs.shape) > 1:
        tox_probs = tox_probs.flatten()
    
    # Make sure the length matches - truncate or pad if necessary
    if len(tox_probs) != len(toxicity_preds):
        if len(tox_probs) > len(toxicity_preds):
            tox_probs = tox_probs[:len(toxicity_preds)]
        else:
            pad_length = len(toxicity_preds) - len(tox_probs)
            tox_probs = np.pad(tox_probs, (0, pad_length), 'constant', constant_values=0.5)
    
    # Analyze confidence of errors
    fp_indices = np.where(fp_mask)[0]
    fn_indices = np.where(fn_mask)[0]
    
    # Ensure indices are within bounds
    fp_indices = fp_indices[fp_indices < len(tox_probs)]
    fn_indices = fn_indices[fn_indices < len(tox_probs)]
    
    fp_confidences = [tox_probs[i] for i in fp_indices] if len(fp_indices) > 0 else []
    fn_confidences = [1 - tox_probs[i] for i in fn_indices] if len(fn_indices) > 0 else []
    
    # Find high-confidence errors
    high_conf_threshold = 0.8
    
    # Use numpy operations for better handling of arrays
    high_conf_fps = []
    for i in fp_indices:
        if i < len(tox_probs):
            if np.isscalar(tox_probs[i]):
                # Check if it's a high confidence false positive
                if tox_probs[i] > high_conf_threshold:
                    high_conf_fps.append(i)
            elif hasattr(tox_probs[i], '__iter__'):
                # If it's an array-like object, check the first element
                if tox_probs[i][0] > high_conf_threshold:
                    high_conf_fps.append(i)
                
    high_conf_fns = []
    for i in fn_indices:
        if i < len(tox_probs):
            if np.isscalar(tox_probs[i]):
                # Check if it's a high confidence false negative
                if (1 - tox_probs[i]) > high_conf_threshold:
                    high_conf_fns.append(i)
            elif hasattr(tox_probs[i], '__iter__'):
                # If it's an array-like object, check the first element
                if (1 - tox_probs[i][0]) > high_conf_threshold:
                    high_conf_fns.append(i)
    
    # Prepare examples of high-confidence errors
    fp_examples = []
    for i in high_conf_fps[:10]:  # Limit to 10 examples
        if i < len(texts) and i < len(tox_probs):
            conf_val = tox_probs[i]
            if hasattr(conf_val, '__iter__'):
                conf_val = conf_val[0]
                
            unc_val = None
            if uncertainties and 'overall' in uncertainties:
                unc_array = uncertainties['overall']
                if i < len(unc_array):
                    unc_val = unc_array[i]
                    if hasattr(unc_val, '__iter__'):
                        unc_val = unc_val[0]
            
            fp_examples.append({
                'text': texts[i],
                'confidence': float(conf_val),
                'uncertainty': float(unc_val) if unc_val is not None else None
            })
    
    fn_examples = []
    for i in high_conf_fns[:10]:  # Limit to 10 examples
        if i < len(texts) and i < len(tox_probs):
            conf_val = 1 - tox_probs[i]
            if hasattr(conf_val, '__iter__'):
                conf_val = 1 - conf_val[0]
                
            unc_val = None
            if uncertainties and 'overall' in uncertainties:
                unc_array = uncertainties['overall']
                if i < len(unc_array):
                    unc_val = unc_array[i]
                    if hasattr(unc_val, '__iter__'):
                        unc_val = unc_val[0]
            
            fn_examples.append({
                'text': texts[i],
                'confidence': float(conf_val),
                'uncertainty': float(unc_val) if unc_val is not None else None
            })
    
    # Analyze category errors
    category_error_rates = {}
    for i, category in enumerate(Config.CATEGORY_COLUMNS):
        if i < category_preds.shape[1] and i < category_labels.shape[1]:
            try:
                cat_errors = (category_preds[:, i] != category_labels[:, i])
                cat_error_rate = np.mean(cat_errors)
                category_error_rates[category] = cat_error_rate
            except Exception as e:
                logger.error(f"Error calculating category error rate for {category}: {e}")
                category_error_rates[category] = 0.0
    
    # Analyze uncertainty (if available)
    uncertainty_analysis = None
    if uncertainties and 'overall' in uncertainties:
        try:
            # Convert to numpy array if needed
            overall_uncertainty = np.array(uncertainties['overall'])
            
            # Handle potentially nested arrays
            if len(overall_uncertainty.shape) > 1:
                overall_uncertainty = overall_uncertainty.flatten()
                
            # Make sure the length matches - truncate or pad if necessary
            if len(overall_uncertainty) != len(toxicity_preds):
                if len(overall_uncertainty) > len(toxicity_preds):
                    overall_uncertainty = overall_uncertainty[:len(toxicity_preds)]
                else:
                    pad_length = len(toxicity_preds) - len(overall_uncertainty)
                    overall_uncertainty = np.pad(overall_uncertainty, (0, pad_length), 'constant', constant_values=0.0)
            
            # Calculate average uncertainty
            avg_uncertainty = np.mean(overall_uncertainty)
            
            # Use only valid indices for correct and incorrect masks
            valid_mask_length = min(len(overall_uncertainty), len(correct_mask))
            valid_correct_mask = correct_mask[:valid_mask_length]
            valid_incorrect_mask = incorrect_mask[:valid_mask_length]
            valid_uncertainty = overall_uncertainty[:valid_mask_length]
            
            correct_uncertainty_values = valid_uncertainty[valid_correct_mask]
            incorrect_uncertainty_values = valid_uncertainty[valid_incorrect_mask]
            
            correct_uncertainty = np.mean(correct_uncertainty_values) if len(correct_uncertainty_values) > 0 else 0.0
            incorrect_uncertainty = np.mean(incorrect_uncertainty_values) if len(incorrect_uncertainty_values) > 0 else 0.0
            
            uncertainty_analysis = {
                'average_uncertainty': float(avg_uncertainty),
                'correct_uncertainty': float(correct_uncertainty),
                'incorrect_uncertainty': float(incorrect_uncertainty),
                'uncertainty_ratio': float(incorrect_uncertainty / max(0.001, correct_uncertainty))
            }
        except Exception as e:
            logger.error(f"Error in uncertainty analysis: {e}")
            uncertainty_analysis = {
                'average_uncertainty': 0.0,
                'correct_uncertainty': 0.0,
                'incorrect_uncertainty': 0.0,
                'uncertainty_ratio': 1.0
            }
    
    # Generate threshold adjustment recommendations
    recommendations = []
    
    # If false positive rate is high
    if fp_rate > 0.1:
        recommendations.append("adjust_class_weights:decrease_tox")
        recommendations.append(f"adjust_category_threshold:toxicity:increase")
    
    # If false negative rate is high
    if fn_rate > 0.1:
        recommendations.append("adjust_class_weights:increase_tox")
        recommendations.append(f"adjust_category_threshold:toxicity:decrease")
    
    # Category-specific recommendations
    for i, category in enumerate(Config.CATEGORY_COLUMNS):
        if i < category_preds.shape[1] and i < category_labels.shape[1]:
            try:
                cat_fps = (category_preds[:, i] == 1) & (category_labels[:, i] == 0)
                cat_fns = (category_preds[:, i] == 0) & (category_labels[:, i] == 1)
                
                cat_fp_rate = np.mean(cat_fps)
                cat_fn_rate = np.mean(cat_fns)
                
                if cat_fp_rate > 0.1:
                    recommendations.append(f"adjust_category_threshold:{category}:increase")
                
                if cat_fn_rate > 0.1:
                    recommendations.append(f"adjust_category_threshold:{category}:decrease")
            except Exception as e:
                logger.error(f"Error generating recommendations for {category}: {e}")
    
    # Return comprehensive analysis
    return {
        'error_rate': float(error_rate),
        'false_positive_rate': float(fp_rate),
        'false_negative_rate': float(fn_rate),
        'category_error_rates': category_error_rates,
        'high_confidence_fp_count': len(high_conf_fps),
        'high_confidence_fn_count': len(high_conf_fns),
        'fp_examples': fp_examples,
        'fn_examples': fn_examples,
        'uncertainty_analysis': uncertainty_analysis,
        'recommendations': recommendations
    }

def evaluate_on_ood_data(model, id_dataloader, ood_dataloader, mc_dropout=True):
    """
    Evaluate model on in-distribution and out-of-distribution data.
    
    Args:
        model: The model to evaluate
        id_dataloader: DataLoader for in-distribution data
        ood_dataloader: DataLoader for out-of-distribution data
        mc_dropout: Whether to use Monte Carlo dropout for uncertainty estimation
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Evaluating on in-distribution data...")
    id_results = evaluate_model(model, id_dataloader, mc_dropout=mc_dropout)
    
    logger.info("Evaluating on out-of-distribution data...")
    ood_results = evaluate_model(model, ood_dataloader, mc_dropout=mc_dropout)
    
    # Calculate performance gap
    id_accuracy = id_results['accuracy']
    ood_accuracy = ood_results['accuracy']
    accuracy_gap = id_accuracy - ood_accuracy
    
    id_category_f1 = id_results['category_macro_f1']
    ood_category_f1 = ood_results['category_macro_f1']
    category_f1_gap = id_category_f1 - ood_category_f1
    
    logger.info("\nPerformance Gap Analysis:")
    logger.info(f"In-distribution accuracy: {id_accuracy:.4f}")
    logger.info(f"Out-of-distribution accuracy: {ood_accuracy:.4f}")
    logger.info(f"Accuracy gap: {accuracy_gap:.4f} ({accuracy_gap/id_accuracy*100:.1f}% drop)")
    
    logger.info(f"In-distribution category F1: {id_category_f1:.4f}")
    logger.info(f"Out-of-distribution category F1: {ood_category_f1:.4f}")
    logger.info(f"Category F1 gap: {category_f1_gap:.4f} ({category_f1_gap/id_category_f1*100:.1f}% drop)")
    
    # If uncertainty is available, analyze it
    if mc_dropout:
        id_uncertainty = id_results['all_predictions']['uncertainties']['overall']
        ood_uncertainty = ood_results['all_predictions']['uncertainties']['overall']
        
        avg_id_uncertainty = np.mean(id_uncertainty)
        avg_ood_uncertainty = np.mean(ood_uncertainty)
        
        logger.info(f"In-distribution avg uncertainty: {avg_id_uncertainty:.4f}")
        logger.info(f"Out-of-distribution avg uncertainty: {avg_ood_uncertainty:.4f}")
        logger.info(f"Uncertainty ratio: {avg_ood_uncertainty/avg_id_uncertainty:.2f}x")
    
    return {
        'in_distribution': id_results,
        'out_of_distribution': ood_results,
        'performance_gap': {
            'accuracy_gap': float(accuracy_gap),
            'accuracy_drop_percent': float(accuracy_gap/id_accuracy*100),
            'category_f1_gap': float(category_f1_gap),
            'category_f1_drop_percent': float(category_f1_gap/max(0.001, id_category_f1)*100)
        }
    }

# ==================================================================================
# Prediction Functions
# ==================================================================================

def predict_toxicity(model, texts, char_vocab, batch_size=32, use_mc_dropout=False, 
                   num_mc_samples=30):
    """
    Make toxicity predictions on a batch of texts.
    
    Args:
        model: Toxicity detection model
        texts: List of input texts
        char_vocab: Character vocabulary
        batch_size: Batch size for processing
        use_mc_dropout: Whether to use Monte Carlo dropout for uncertainty estimation
        num_mc_samples: Number of MC samples if use_mc_dropout is True
        
    Returns:
        List of prediction results
    """
    results = []
    device = next(model.parameters()).device
    
    # Create MC model wrapper if needed
    if use_mc_dropout:
        mc_model = MCDropoutChainModel(model)
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Preprocess texts
        preprocessed_texts = [enhanced_text_preprocessing(text) for text in batch_texts]
        
        # Extract features
        features_list = [feature_extractor.extract_features(text) for text in preprocessed_texts]
        
        # Encode texts
        char_ids_list = [char_vocab.encode_text(text, Config.MAX_CHARS) for text in preprocessed_texts]
        char_ids_np = np.array(char_ids_list)
        char_ids_tensor = torch.tensor(char_ids_np, dtype=torch.long).to(device)
        
        # Extract features for model input
        toxicity_features_tensor = torch.tensor([
            [
                features['all_caps_ratio'],
                features['toxic_keyword_count'],
                features['toxic_keyword_ratio'],
                features['safe_word_count'],
                features['safe_word_ratio'],
                features['special_char_ratio'],
                float(features['is_educational']),
                len(text) / 300.0  # Normalize by max length
            ]
            for text, features in zip(preprocessed_texts, features_list)
        ], dtype=torch.float).to(device)
        
        # Detect language for each text
        languages = [detect_language(text) for text in preprocessed_texts]
        
        # Get predictions with appropriate method
        batch_results = []
        with torch.no_grad():
            if use_mc_dropout:
                for j, language in enumerate(languages):
                    # Process each example individually with correct language-specific thresholds
                    single_char_ids = char_ids_tensor[j:j+1]
                    single_features = toxicity_features_tensor[j:j+1]
                    
                    # Get MC predictions
                    predictions = mc_model.predict_with_uncertainty(
                        single_char_ids, 
                        single_features, 
                        num_samples=num_mc_samples,
                        language=language
                    )
                    
                    # Create result dictionary
                    toxicity_levels = ['not toxic', 'toxic', 'very toxic']
                    result = {
                        'text': batch_texts[j],
                        'language': language,
                        'toxicity': {
                            'label': toxicity_levels[predictions['toxicity_level'][0].item()],
                            'level': predictions['toxicity_level'][0].item(),
                            'probability': predictions['probabilities']['toxicity'][0].item(),
                            'uncertainty': predictions['uncertainty']['toxicity'][0].item()
                        },
                        'categories': {
                            category: {
                                'detected': bool(predictions['categories'][category][0].item()),
                                'probability': predictions['probabilities'][category][0].item(),
                                'uncertainty': predictions['uncertainty'][category][0].item()
                            }
                            for category in Config.CATEGORY_COLUMNS
                        },
                        'uncertainty': {
                            'overall': predictions['uncertainty']['overall'][0].item(),
                            'reliability': predictions['reliability'][0].item()
                        },
                        'features': features_list[j],
                        'thresholds': predictions['applied_thresholds']
                    }
                    
                    batch_results.append(result)
            else:
                for j, language in enumerate(languages):
                    # Process each example individually with correct language-specific thresholds
                    single_char_ids = char_ids_tensor[j:j+1]
                    single_features = toxicity_features_tensor[j:j+1]
                    
                    # Get standard predictions
                    predictions = model.predict(
                        single_char_ids,
                        single_features,
                        language=language
                    )
                    
                    # Create result dictionary
                    toxicity_levels = ['not toxic', 'toxic', 'very toxic']
                    result = {
                        'text': batch_texts[j],
                        'language': language,
                        'toxicity': {
                            'label': toxicity_levels[predictions['toxicity_level'][0].item()],
                            'level': predictions['toxicity_level'][0].item(),
                            'probability': predictions['probabilities']['toxicity'][0].item()
                        },
                        'categories': {
                            category: {
                                'detected': bool(predictions['categories'][category][0].item()),
                                'probability': predictions['probabilities'][category][0].item()
                            }
                            for category in Config.CATEGORY_COLUMNS
                        },
                        'features': features_list[j],
                        'thresholds': predictions['applied_thresholds']
                    }
                    
                    batch_results.append(result)
        
        results.extend(batch_results)
    
    return results

def analyze_prediction(result):
    """
    Generate human-readable analysis of a toxicity prediction.
    
    Args:
        result: Prediction result dictionary
        
    Returns:
        Dictionary with detailed analysis
    """
    analysis = {
        'summary': "",
        'confidence': "",
        'explanation': [],
        'recommendations': []
    }
    
    # Get basic information
    toxicity_level = result['toxicity']['level']
    toxicity_prob = result['toxicity']['probability']
    
    # Get uncertainty information if available
    has_uncertainty = 'uncertainty' in result
    if has_uncertainty:
        uncertainty = result['uncertainty']['overall']
        reliability = result.get('uncertainty', {}).get('reliability', 1.0 - uncertainty)
    else:
        uncertainty = None
        reliability = None
    
    # Get detected categories
    detected_categories = [
        category for category, info in result['categories'].items()
        if info['detected']
    ]
    
    # Get features that influenced the decision
    features = result['features']
    toxic_keywords = features.get('detected_keywords', [])
    safe_words = features.get('detected_safe_words', [])
    is_educational = features.get('is_educational', False)
    
    # Generate summary
    if toxicity_level == 0:
        analysis['summary'] = "Not toxic"
        if toxic_keywords:
            analysis['summary'] += f" (despite {len(toxic_keywords)} potentially problematic terms)"
    elif toxicity_level == 1:
        analysis['summary'] = f"Toxic: {', '.join(detected_categories) if detected_categories else 'General toxicity'}"
    else:  # toxicity_level == 2
        analysis['summary'] = f"Very Toxic: {', '.join(detected_categories) if detected_categories else 'Severe toxicity'}"
    
    # Generate confidence statement
    if has_uncertainty:
        if uncertainty > 0.15:
            confidence = "Low"
        elif uncertainty > 0.08:
            confidence = "Moderate"
        else:
            confidence = "High"
            
        analysis['confidence'] = f"{confidence} confidence (uncertainty: {uncertainty:.3f}, reliability: {reliability:.3f})"
    else:
        if toxicity_prob > 0.9 or toxicity_prob < 0.1:
            confidence = "High"
        elif toxicity_prob > 0.7 or toxicity_prob < 0.3:
            confidence = "Moderate"
        else:
            confidence = "Low"
            
        analysis['confidence'] = f"{confidence} confidence (probability: {toxicity_prob:.3f})"
    
    # Generate explanation
    if toxicity_level > 0:
        # Explain why it was classified as toxic
        analysis['explanation'].append(f"Toxicity probability: {toxicity_prob:.3f}")
        
        if toxic_keywords:
            analysis['explanation'].append(f"Detected {len(toxic_keywords)} potentially problematic terms: {', '.join(toxic_keywords[:5])}" + 
                                          (f" and {len(toxic_keywords)-5} more" if len(toxic_keywords) > 5 else ""))
        
        if detected_categories:
            for category in detected_categories:
                category_prob = result['categories'][category]['probability']
                if has_uncertainty:
                    category_uncertainty = result['categories'][category]['uncertainty']
                    analysis['explanation'].append(f"{category.capitalize()} detected with {category_prob:.3f} probability (uncertainty: {category_uncertainty:.3f})")
                else:
                    analysis['explanation'].append(f"{category.capitalize()} detected with {category_prob:.3f} probability")
    else:
        # Explain why it wasn't classified as toxic
        if safe_words:
            analysis['explanation'].append(f"Detected {len(safe_words)} safe context indicators: {', '.join(safe_words[:3])}" + 
                                          (f" and {len(safe_words)-3} more" if len(safe_words) > 3 else ""))
        
        if is_educational:
            analysis['explanation'].append("Educational content detected")
        
        if toxic_keywords:
            analysis['explanation'].append(f"Despite {len(toxic_keywords)} potentially problematic terms, context suggests non-toxic intent")
    
    # Generate recommendations
    if toxicity_level > 0:
        if has_uncertainty and uncertainty > 0.15:
            analysis['recommendations'].append("Consider manual review due to high uncertainty")
        
        if is_educational and toxicity_level > 0:
            analysis['recommendations'].append("May be educational content misclassified as toxic")
        
        if len(safe_words) > 2 and toxicity_level > 0:
            analysis['recommendations'].append("Multiple safe context indicators suggest possible false positive")
    
    # Add thresholds info
    if 'thresholds' in result:
        analysis['thresholds'] = result['thresholds']
    
    return analysis

def interactive_prediction(model, char_vocab):
    """
    Interactive toxicity prediction with detailed analysis.
    
    Args:
        model: Toxicity detection model
        char_vocab: Character vocabulary
    """
    print("\n=== Enhanced Toxicity Detection - Interactive Mode ===")
    print("Type 'exit' to quit, 'mc on' to enable uncertainty estimation, 'mc off' to disable it")
    
    # Start with MC dropout on by default
    use_mc_dropout = True
    
    while True:
        # Get text input
        text = input("\nEnter text to analyze: ")
        
        if text.lower() == 'exit':
            break
        elif text.lower() == 'mc on':
            use_mc_dropout = True
            print("Monte Carlo dropout enabled - uncertainty estimation active")
            continue
        elif text.lower() == 'mc off':
            use_mc_dropout = False
            print("Monte Carlo dropout disabled")
            continue
        
        # Make prediction
        results = predict_toxicity(
            model, [text], char_vocab, 
            use_mc_dropout=use_mc_dropout,
            num_mc_samples=Config.MC_DROPOUT_SAMPLES
        )
        result = results[0]
        
        # Analyze prediction
        analysis = analyze_prediction(result)
        
        # Display results
        print("\n=== Classification Results ===")
        print(f"Text: {text}")
        print(f"Classification: {analysis['summary']}")
        print(f"Confidence: {analysis['confidence']}")
        
        print("\nExplanation:")
        for point in analysis['explanation']:
            print(f"- {point}")
        
        if analysis['recommendations']:
            print("\nRecommendations:")
            for rec in analysis['recommendations']:
                print(f"- {rec}")
        
        print("\nDetailed Features:")
        features = result['features']
        print(f"- ALL CAPS usage: {features['all_caps_ratio']*100:.1f}% of words")
        print(f"- Toxic keywords: {features['toxic_keyword_count']} ({features['toxic_keyword_ratio']*100:.1f}% of words)")
        if 'safe_word_count' in features:
            print(f"- Safe context indicators: {features['safe_word_count']}")
        print(f"- Special characters: {features.get('special_char_count', 0)}")
        print(f"- Educational content: {'Yes' if features.get('is_educational', False) else 'No'}")
        
        # Display thresholds used
        if 'thresholds' in result:
            print("\nApplied thresholds:")
            for key, value in result['thresholds'].items():
                print(f"- {key}: {value:.2f}")

# ==================================================================================
# Enhanced Feedback System
# ==================================================================================

class EnhancedFeedbackManager:
    """Enhanced feedback collection and retraining system."""
    
    def __init__(self, model, char_vocab):
        self.model = model
        self.char_vocab = char_vocab
        self.feedback_examples = []
        self.original_model_state = copy.deepcopy(model.state_dict())
        self.min_feedback_for_retraining = Config.MIN_FEEDBACK_FOR_RETRAINING
    
    def add_feedback(self, text, prediction, correct_toxicity=None, correct_categories=None):
        """
        Add a feedback example.
        
        Args:
            text: Input text
            prediction: Model prediction
            correct_toxicity: Correct toxicity level (0, 1, or 2)
            correct_categories: List of correct category values (0 or 1)
            
        Returns:
            Number of collected feedback examples
        """
        # If correct values not provided, use prediction
        if correct_toxicity is None:
            correct_toxicity = prediction['toxicity']['level']
        
        if correct_categories is None:
            correct_categories = [
                int(prediction['categories'][cat]['detected'])
                for cat in Config.CATEGORY_COLUMNS
            ]
        
        # Add to feedback examples
        self.feedback_examples.append({
            'text': text,
            'pred_toxicity': prediction['toxicity']['level'],
            'true_toxicity': correct_toxicity,
            'pred_categories': [
                int(prediction['categories'][cat]['detected'])
                for cat in Config.CATEGORY_COLUMNS
            ],
            'true_categories': correct_categories,
            'timestamp': time.time()
        })
        
        logger.info(f"Feedback recorded. Total examples: {len(self.feedback_examples)}")
        logger.info(f"Need {self.min_feedback_for_retraining - len(self.feedback_examples)} more examples before retraining")
        
        return len(self.feedback_examples)
    
    def perform_retraining(self, epochs=15, learning_rate=0.0001):
        """
        Perform retraining using feedback examples.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Boolean indicating success
        """
        if len(self.feedback_examples) < self.min_feedback_for_retraining:
            logger.warning(f"Not enough feedback examples ({len(self.feedback_examples)}/{self.min_feedback_for_retraining})")
            return False
            
        logger.info(f"Retraining on {len(self.feedback_examples)} feedback examples...")
        
        # Create dataset from feedback
        texts = [ex['text'] for ex in self.feedback_examples]
        labels = np.zeros((len(self.feedback_examples), 1 + len(Config.CATEGORY_COLUMNS)))
        
        for i, ex in enumerate(self.feedback_examples):
            labels[i, 0] = ex['true_toxicity']
            if ex['true_categories'] is not None:
                for j, cat_val in enumerate(ex['true_categories']):
                    if j < len(Config.CATEGORY_COLUMNS):
                        labels[i, j+1] = cat_val
        
        # Split feedback data into train (80%) and validation (20%)
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            range(len(texts)), test_size=0.2, stratify=labels[:, 0], random_state=42
        )
        
        train_texts = [texts[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = labels[val_idx]
        
        # Create datasets
        train_dataset = EnhancedToxicityDataset(
            train_texts, train_labels, self.char_vocab,
            max_len=Config.MAX_CHARS, detect_lang=Config.USE_LANGUAGE_DETECTION
        )
        
        val_dataset = EnhancedToxicityDataset(
            val_texts, val_labels, self.char_vocab,
            max_len=Config.MAX_CHARS, detect_lang=Config.USE_LANGUAGE_DETECTION
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True,
            num_workers=Config.NUM_WORKERS, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False,
            num_workers=Config.NUM_WORKERS, pin_memory=True
        )
        
        # Perform retraining
        retrained_model, best_metrics, _ = train_enhanced_classifier_chain(
            self.model, train_loader, val_loader,
            num_epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=3
        )
        
        logger.info(f"Retraining complete. Best validation loss: {best_metrics['loss']:.4f}")
        logger.info(f"Toxicity accuracy: {best_metrics['toxicity_acc']:.4f}, Category F1: {best_metrics['category_f1']:.4f}")
        
        return True
    
    def reset_to_original_state(self):
        """Reset model to original state."""
        self.model.load_state_dict(self.original_model_state)
        logger.info("Model reset to original state")
    
    def save_feedback_data(self, save_path):
        """Save feedback data to file."""
        feedback_data = {
            'feedback_examples': self.feedback_examples,
            'timestamp': time.time()
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(feedback_data, f)
            
        logger.info(f"Feedback data saved to {save_path}")
    
    def load_feedback_data(self, load_path):
        """Load feedback data from file."""
        try:
            with open(load_path, 'rb') as f:
                feedback_data = pickle.load(f)
                
            self.feedback_examples = feedback_data.get('feedback_examples', [])
            
            logger.info(f"Loaded {len(self.feedback_examples)} feedback examples")
            return True
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
            return False

# ==================================================================================
# Training Pipeline
# ==================================================================================

def train_toxicity_model(data_path=None, output_dir=None, num_epochs=40):
    """
    Complete training pipeline for toxicity detection model.
    
    Args:
        data_path: Path to data CSV
        output_dir: Directory for saving outputs
        num_epochs: Number of training epochs
        
    Returns:
        Tuple of (model, char_vocab, evaluation_results)
    """
    # Use config values if not provided
    if data_path is None:
        data_path = Config.DATA_PATH
    
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging to file
    log_path = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting toxicity detection model training pipeline")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Set random seed for reproducibility
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    
    # Step 1: Load and prepare data
    logger.info("Loading data...")
    texts, labels = load_data_from_csv(
        data_path,
        text_column=Config.TEXT_COLUMN,
        toxicity_column=Config.TOXICITY_COLUMN,
        category_columns=Config.CATEGORY_COLUMNS
    )
    
    # Step 2: Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader, char_vocab = create_data_loaders(
        texts, labels,
        batch_size=Config.BATCH_SIZE,
        max_len=Config.MAX_CHARS,
        detect_lang=Config.USE_LANGUAGE_DETECTION,
        num_workers=Config.NUM_WORKERS,
        seed=Config.SEED
    )
    
    # Step 3: Create model
    logger.info("Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create base model
    base_model = EnhancedToxicityModel(
        n_chars=char_vocab.n_chars,
        char_emb_dim=Config.CHAR_EMB_DIM,
        lstm_hidden_dim=Config.LSTM_HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
        feature_dim=8  # Number of features
    ).to(device)
    
    # Create classifier chain model
    chain_model = EnhancedClassifierChain(base_model).to(device)
    
    # Step 4: Train model
    logger.info("Training classifier chain model...")
    trained_model, best_metrics, metrics_history = train_enhanced_classifier_chain(
        chain_model, train_loader, val_loader,
        num_epochs=num_epochs,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )
    
    # Step 5: Evaluate model
    logger.info("Evaluating model on test set...")
    test_results = evaluate_model(trained_model, test_loader, mc_dropout=True)
    
    # Step 6: Create and evaluate on OOD data
    ood_data_path = os.path.join(output_dir, 'ood_test_data.csv')
    
    if not os.path.exists(ood_data_path):
        logger.info("Creating OOD test data...")
        from sklearn.model_selection import train_test_split
        
        # Take a subset of data for OOD testing
        _, ood_texts, _, ood_labels = train_test_split(
            texts, labels, test_size=0.1, random_state=Config.SEED+1, stratify=labels[:, 0]
        )
        
        # Create OOD dataset
        ood_dataset = EnhancedToxicityDataset(
            ood_texts, ood_labels, char_vocab,
            max_len=Config.MAX_CHARS, detect_lang=Config.USE_LANGUAGE_DETECTION
        )
        
        # Create OOD dataloader
        ood_loader = DataLoader(
            ood_dataset, batch_size=Config.BATCH_SIZE,
            shuffle=False, num_workers=Config.NUM_WORKERS
        )
    else:
        logger.info(f"Loading OOD test data from {ood_data_path}...")
        ood_texts, ood_labels = load_data_from_csv(
            ood_data_path,
            text_column=Config.TEXT_COLUMN,
            toxicity_column=Config.TOXICITY_COLUMN,
            category_columns=Config.CATEGORY_COLUMNS
        )
        
        # Create OOD dataset
        ood_dataset = EnhancedToxicityDataset(
            ood_texts, ood_labels, char_vocab,
            max_len=Config.MAX_CHARS, detect_lang=Config.USE_LANGUAGE_DETECTION
        )
        
        # Create OOD dataloader
        ood_loader = DataLoader(
            ood_dataset, batch_size=Config.BATCH_SIZE,
            shuffle=False, num_workers=Config.NUM_WORKERS
        )
    
    # Evaluate on OOD data
    logger.info("Evaluating model on OOD test data...")
    ood_results = evaluate_on_ood_data(trained_model, test_loader, ood_loader)
    
    # Step 7: Save model and artifacts
    logger.info("Saving model and artifacts...")
    
    # Save model
    model_path = os.path.join(output_dir, 'toxicity_model.pth')
    torch.save(trained_model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, 'char_vocab.pkl')
    char_vocab.save(vocab_path)
    logger.info(f"Vocabulary saved to {vocab_path}")
    
    # Save training metrics
    metrics_path = os.path.join(output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'best_metrics': best_metrics,
            'train_loss': metrics_history['train_loss'],
            'val_loss': metrics_history['val_loss'],
            'val_toxicity_acc': metrics_history['val_toxicity_acc'],
            'val_category_f1': metrics_history['val_category_f1'],
            'learning_rates': metrics_history['learning_rates'],
            'test_accuracy': test_results['accuracy'],
            'test_category_f1': test_results['category_macro_f1'],
            'ood_accuracy': ood_results['out_of_distribution']['accuracy'],
            'ood_category_f1': ood_results['out_of_distribution']['category_macro_f1'],
            'performance_gap': ood_results['performance_gap']
        }, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")
    
    # Step 8: Generate training report
    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write("=== Toxicity Detection Model Training Report ===\n\n")
        
        f.write(f"Data path: {data_path}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        f.write("Model Configuration:\n")
        for key, value in Config.__dict__.items():
            if not key.startswith('__') and not callable(value):
                f.write(f"  {key}: {value}\n")
        
        f.write("\nTraining Results:\n")
        f.write(f"  Best validation loss: {best_metrics['loss']:.4f} at epoch {best_metrics['epoch']}\n")
        f.write(f"  Best validation toxicity accuracy: {best_metrics['toxicity_acc']:.4f}\n")
        f.write(f"  Best validation category F1: {best_metrics['category_f1']:.4f}\n\n")
        
        f.write("Test Results:\n")
        f.write(f"  Test accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"  Test category F1: {test_results['category_macro_f1']:.4f}\n\n")
        
        f.write("OOD Evaluation:\n")
        f.write(f"  OOD accuracy: {ood_results['out_of_distribution']['accuracy']:.4f}\n")
        f.write(f"  OOD category F1: {ood_results['out_of_distribution']['category_macro_f1']:.4f}\n")
        f.write(f"  Accuracy gap: {ood_results['performance_gap']['accuracy_gap']:.4f} ({ood_results['performance_gap']['accuracy_drop_percent']:.1f}% drop)\n")
        f.write(f"  Category F1 gap: {ood_results['performance_gap']['category_f1_gap']:.4f} ({ood_results['performance_gap']['category_f1_drop_percent']:.1f}% drop)\n")
    
    logger.info(f"Training report saved to {report_path}")
    
    # Return model, vocabulary, and evaluation results
    return trained_model, char_vocab, {
        'test_results': test_results,
        'ood_results': ood_results,
        'best_metrics': best_metrics,
        'metrics_history': metrics_history
    }

def load_trained_model(model_path, vocab_path):
    """
    Load a trained toxicity detection model.
    
    Args:
        model_path: Path to model file
        vocab_path: Path to vocabulary file
        
    Returns:
        Tuple of (model, char_vocab)
    """
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        char_vocab = pickle.load(f)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create base model
    base_model = EnhancedToxicityModel(
        n_chars=char_vocab.n_chars,
        char_emb_dim=Config.CHAR_EMB_DIM,
        lstm_hidden_dim=Config.LSTM_HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
        feature_dim=8  # Number of features
    ).to(device)
    
    # Create classifier chain model
    chain_model = EnhancedClassifierChain(base_model).to(device)
    
    # Load model weights
    chain_model.load_state_dict(torch.load(model_path, map_location=device))
    chain_model.eval()
    
    return chain_model, char_vocab

# ==================================================================================
# Main Function
# ==================================================================================

def main():
    """Main function to run the toxicity detection pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Toxicity Detection Model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'interactive'],
                       help='Operation mode')
    parser.add_argument('--data', type=str, default=Config.DATA_PATH,
                       help='Path to data CSV')
    parser.add_argument('--output', type=str, default=Config.OUTPUT_DIR,
                       help='Output directory')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (for evaluate/interactive modes)')
    parser.add_argument('--vocab', type=str, default=None,
                       help='Path to vocabulary file (for evaluate/interactive modes)')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train mode
        train_toxicity_model(
            data_path=args.data,
            output_dir=args.output,
            num_epochs=args.epochs
        )
    
    elif args.mode == 'evaluate':
        # Evaluate mode
        if args.model is None or args.vocab is None:
            # Try to find model and vocabulary in output directory
            args.model = os.path.join(args.output, 'toxicity_model.pth')
            args.vocab = os.path.join(args.output, 'char_vocab.pkl')
            
            if not os.path.exists(args.model) or not os.path.exists(args.vocab):
                print("Error: Model or vocabulary not found. Please specify paths with --model and --vocab")
                return
        
        # Load model and vocabulary
        model, char_vocab = load_trained_model(args.model, args.vocab)
        
        # Load test data
        texts, labels = load_data_from_csv(
            args.data,
            text_column=Config.TEXT_COLUMN,
            toxicity_column=Config.TOXICITY_COLUMN,
            category_columns=Config.CATEGORY_COLUMNS
        )
        
        # Create test dataset
        test_dataset = EnhancedToxicityDataset(
            texts, labels, char_vocab,
            max_len=Config.MAX_CHARS, detect_lang=Config.USE_LANGUAGE_DETECTION
        )
        
        # Create test dataloader
        test_loader = DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE,
            shuffle=False, num_workers=Config.NUM_WORKERS
        )
        
        # Evaluate model
        test_results = evaluate_model(model, test_loader, mc_dropout=True)
        
        # Print results
        print("\n=== Evaluation Results ===")
        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print(f"Category Macro F1: {test_results['category_macro_f1']:.4f}")
        
        # Save results
        results_path = os.path.join(args.output, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'accuracy': test_results['accuracy'],
                'category_macro_f1': test_results['category_macro_f1'],
                'toxicity_report': test_results['toxicity_report'],
                'category_metrics': test_results['category_metrics']
            }, f, indent=2)
        
        print(f"Results saved to {results_path}")
    
    elif args.mode == 'interactive':
        # Interactive mode
        if args.model is None or args.vocab is None:
            # Try to find model and vocabulary in output directory
            args.model = os.path.join(args.output, 'toxicity_model.pth')
            args.vocab = os.path.join(args.output, 'char_vocab.pkl')
            
            if not os.path.exists(args.model) or not os.path.exists(args.vocab):
                print("Error: Model or vocabulary not found. Please specify paths with --model and --vocab")
                return
        
        # Load model and vocabulary
        model, char_vocab = load_trained_model(args.model, args.vocab)
        
        # Start interactive prediction
        interactive_prediction(model, char_vocab)

if __name__ == "__main__":
        main()      
        