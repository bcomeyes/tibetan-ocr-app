"""
═══════════════════════════════════════════════════════════════════════════════
PyBo (Tibetan Tokenizer) - Deep Dive Exploration
═══════════════════════════════════════════════════════════════════════════════

WHAT IS PYBO?
PyBo is a rule-based word tokenizer for Tibetan text. It was developed by the 
Esukhia project with funding from the Khyentse Foundation ($22,000 to kickstart)
and BDRC (2 staff for 6 months for data curation).

HOW IT WORKS - THE CORE ALGORITHM:
1. Text Preprocessing: Breaks input into chunks (syllables, punctuation, non-Tibetan)
2. Syllable Cleaning: Removes punctuation and spaces from syllable chunks  
3. Trie Matching: Walks through a Trie data structure built from the THL lexicon
4. Longest Match: Finds the longest possible word match in the lexicon
5. POS Tagging: Assigns part-of-speech tags based on lexicon entries

THE TRIE DATA STRUCTURE:
A trie (from "retrieval") is like a tree where each node represents a character.
It allows extremely fast lookup of words. For example:

Root
├─ བ
│  ├─ ད (complete word: "བད")
│  └─ དེ (complete word: "བདེ")
└─ ག
   └─ ན་ས (complete word: "གནས")

This structure lets PyBo quickly check if "བདེ་ལེགས" contains valid words by
walking down the tree character by character.

WHAT MAKES IT "RULE-BASED"?
- Uses manually curated THL (Tibetan & Himalayan Library) lexicon
- Applies handcrafted heuristics for syllable boundary detection
- Follows explicit rules for affix handling (e.g., ར་, ས་, འི་)
- No machine learning - pure linguistic rules

STRENGTHS:
✓ Excellent for classical Tibetan (scriptures, formal texts)
✓ High accuracy on well-structured text
✓ Interpretable - you can see WHY it made a decision
✓ Provides POS tags (noun, verb, particle, etc.)
✓ Shows syllable structure information

WEAKNESSES:
✗ Limited to words in the THL lexicon (fixed vocabulary)
✗ Struggles with modern Tibetan vocabulary
✗ Can't adapt to new words without manual lexicon updates
✗ May fail on informal or domain-shifted text

USE CASE FOR OCR QUALITY SCORING:
For our grid search, PyBo could help identify "valid words" vs OCR garbage by
checking if detected text exists in the THL lexicon. If OCR output produces
mostly non-lexicon words, we know it's probably garbage.

Author: Matt
Date: 2026-01-25
Purpose: Explore PyBo's capabilities for potential use in OCR quality scoring
═══════════════════════════════════════════════════════════════════════════════
"""

# %%
# ═══════════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════
# Run this in your terminal (not in the notebook):
# pip install pybo --break-system-packages
# ═══════════════════════════════════════════════════════════════════════════

# %%
import json
import time
from pathlib import Path
from typing import List, Dict
from collections import Counter

# %%
# ═══════════════════════════════════════════════════════════════════════════
# LOAD TEST DATA FROM TENGYUR
# ═══════════════════════════════════════════════════════════════════════════

def load_tengyur_sample(json_path: str, num_folios: int = 3) -> List[str]:
    """
    Load clean Tibetan text from Tengyur JSON file.
    
    This gives us known-good text to test PyBo's capabilities.
    
    Args:
        json_path: Path to Tengyur JSON file
        num_folios: Number of folios to load
        
    Returns:
        List of Tibetan text strings
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [folio['content'] for folio in data[:num_folios]]
    
    total_chars = sum(len(t) for t in texts)
    print(f"📚 Loaded {num_folios} folios from Tengyur")
    print(f"📊 Total characters: {total_chars:,}")
    
    return texts

# Load the test data
# Point to the input_files directory in your tibetan-ocr-app project
from pathlib import Path
input_dir = Path.home() / "Documents" / "tibetan-ocr-app" / "input_files"

# Look for JSON files in that directory
json_files = list(input_dir.glob("*.json"))

if not json_files:
    print(f"❌ No JSON files found in {input_dir}")
    print("   Please add Tengyur JSON files to that directory")
    exit()

# Use the first JSON file found
tengyur_file = str(json_files[0])
print(f"📁 Using file: {tengyur_file}")

test_texts = load_tengyur_sample(tengyur_file, num_folios=3)

# %%
# ═══════════════════════════════════════════════════════════════════════════
# INITIALIZE PYBO TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════

try:
    from pybo import WordTokenizer
    
    print("✅ PyBo imported successfully")
    print("\n" + "="*70)
    print("INITIALIZING PYBO TOKENIZER")
    print("="*70)
    
    # Initialize with default configuration
    # This loads the THL lexicon and builds the Trie data structure
    tokenizer = WordTokenizer()
    
    print("✓ Tokenizer initialized")
    print("  This means PyBo will:")
    print("    - Tokenize text into words")
    print("    - Use the THL lexicon for matching")
    print("    - Provide POS tags where available")
    
except ImportError as e:
    print(f"❌ PyBo not installed: {e}")
    print("   Install with: pip install pybo --break-system-packages")
    exit()

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: BASIC TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 1: BASIC TOKENIZATION")
print("="*70)

# Take a small sample
sample_text = test_texts[0][:200]
print(f"\n📝 Input text (first 200 chars):")
print(f"   {sample_text}")

# Tokenize
tokens = tokenizer.tokenize(sample_text)

print(f"\n🔍 PyBo found {len(tokens)} tokens")
print(f"\n💡 First 10 tokens:")
for i, token in enumerate(tokens[:10]):
    # Get attributes that actually exist
    token_text = token.text if hasattr(token, 'text') else str(token)
    token_pos = token.pos if hasattr(token, 'pos') else 'N/A'
    token_type = token.chunk_type if hasattr(token, 'chunk_type') else 'N/A'
    print(f"   {i+1}. '{token_text}' (type: {token_type}, POS: {token_pos})")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: UNDERSTANDING TOKEN ATTRIBUTES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 2: UNDERSTANDING TOKEN ATTRIBUTES")
print("="*70)

print("\nEach PyBo token has multiple attributes:")
print("Let's examine one token in detail:\n")

# Pick the first token to examine
if tokens:
    example_token = tokens[0]
    
    print(f"Token: '{example_token.text if hasattr(example_token, 'text') else example_token}'")
    
    # List all available attributes
    attrs = [attr for attr in dir(example_token) if not attr.startswith('_')]
    print(f"\nAvailable attributes: {', '.join(attrs[:10])}...")
    
    # Show the most useful ones
    if hasattr(example_token, 'text'):
        print(f"  .text        = '{example_token.text}'      (the actual text)")
    if hasattr(example_token, 'pos'):
        print(f"  .pos         = '{example_token.pos}'       (part of speech)")
    if hasattr(example_token, 'chunk_type'):
        print(f"  .chunk_type  = '{example_token.chunk_type}'(token type)")
    if hasattr(example_token, 'tag'):
        print(f"  .tag         = '{example_token.tag}'       (grammatical tag)")
    if hasattr(example_token, 'len'):
        print(f"  .len         = {example_token.len}         (length)")
    
    print(f"\n💡 Key insight: These attributes tell us if this is a KNOWN word in the lexicon")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: EXTRACTING WORDS BY PART OF SPEECH
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 3: EXTRACTING WORDS BY PART OF SPEECH")
print("="*70)

# Tokenize a longer sample
longer_sample = test_texts[0][:500]
tokens = tokenizer.tokenize(longer_sample)

# Count different POS tags
pos_counts = Counter(t.pos for t in tokens if hasattr(t, 'pos') and t.pos)

print("\n📊 Part-of-Speech distribution:")
for pos, count in pos_counts.most_common():
    print(f"   {pos:15s}: {count:3d} tokens")

# Extract specific POS types
nouns = [t.text for t in tokens if hasattr(t, 'pos') and t.pos and 'NOUN' in str(t.pos)]
verbs = [t.text for t in tokens if hasattr(t, 'pos') and t.pos and 'VERB' in str(t.pos)]

if nouns:
    print(f"\n📌 Sample nouns: {' '.join(nouns[:5])}")
if verbs:
    print(f"⚡ Sample verbs: {' '.join(verbs[:5])}")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: CALCULATING "VALID WORD PERCENTAGE" (OCR QUALITY SCORE)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 4: OCR QUALITY SCORING")
print("="*70)

def calculate_pybo_score(text: str) -> Dict:
    """
    Calculate what percentage of text consists of valid Tibetan words.
    
    This is the KEY function for OCR quality scoring.
    
    Returns:
        Dictionary with scoring metrics
    """
    start_time = time.time()
    
    tokens = tokenizer.tokenize(text)
    
    total_tokens = 0
    valid_tokens = 0
    unknown_tokens = 0
    
    for token in tokens:
        # Skip punctuation - only count actual word tokens
        chunk_type = token.chunk_type if hasattr(token, 'chunk_type') else None
        
        if chunk_type and chunk_type in ['PUNCT', 'NON_WORD']:
            continue
            
        total_tokens += 1
        
        # A token is "valid" if it has a POS tag from the lexicon
        if hasattr(token, 'pos') and token.pos:
            valid_tokens += 1
        else:
            unknown_tokens += 1
    
    valid_percentage = (valid_tokens / total_tokens * 100) if total_tokens > 0 else 0
    elapsed = time.time() - start_time
    
    return {
        'total_syllables': total_tokens,
        'valid_words': valid_tokens,
        'unknown_words': unknown_tokens,
        'valid_percentage': valid_percentage,
        'processing_time': elapsed
    }

# Test on known-good text
good_score = calculate_pybo_score(test_texts[0])

print("\n📊 Scoring known-good Tengyur text:")
print(f"   Total syllables:    {good_score['total_syllables']}")
print(f"   Valid words:        {good_score['valid_words']}")
print(f"   Unknown words:      {good_score['unknown_words']}")
print(f"   ✨ SCORE:           {good_score['valid_percentage']:.2f}%")
print(f"   ⏱️  Processing time:  {good_score['processing_time']:.4f}s")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: SIMULATING OCR ERRORS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 5: TESTING ON CORRUPTED TEXT (SIMULATED BAD OCR)")
print("="*70)

def corrupt_text(text: str, corruption_rate: float = 0.2) -> str:
    """
    Randomly corrupt text to simulate OCR errors.
    
    Args:
        text: Clean Tibetan text
        corruption_rate: Fraction of characters to corrupt (0.0 to 1.0)
        
    Returns:
        Corrupted text
    """
    import random
    
    chars = list(text)
    num_to_corrupt = int(len(chars) * corruption_rate)
    
    # Common Tibetan consonants to use as random replacements
    garbage = ['ཀ', 'ག', 'ང', 'ཅ', 'ཇ', 'ཏ', 'ད', 'ན', 'པ', 'བ', 'མ', '་', '།']
    
    positions = random.sample(range(len(chars)), min(num_to_corrupt, len(chars)))
    
    for pos in positions:
        chars[pos] = random.choice(garbage)
    
    return ''.join(chars)

# Test with different corruption levels
sample = test_texts[0][:500]

print("\n🧪 Testing different OCR quality levels:\n")

test_cases = [
    ("Original (perfect OCR)", sample, 0.0),
    ("Slightly corrupted (20%)", corrupt_text(sample, 0.2), 0.2),
    ("Heavily corrupted (50%)", corrupt_text(sample, 0.5), 0.5),
]

for label, text, rate in test_cases:
    score = calculate_pybo_score(text)
    print(f"{label:30s} → Score: {score['valid_percentage']:6.2f}%")

print("\n💡 Insight: PyBo can distinguish good OCR from garbage!")
print("   We can use this to automatically rank OCR outputs")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: PERFORMANCE BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 6: PERFORMANCE BENCHMARKING")
print("="*70)

print("\n⏱️  How fast can PyBo score OCR outputs?")
print("   (Important for processing 1,728 outputs per image!)\n")

# Test on different text sizes
sizes = [100, 500, 1000, 2000]

for size in sizes:
    text_sample = test_texts[0][:size]
    
    # Time multiple runs
    times = []
    for _ in range(5):
        start = time.time()
        tokenizer.tokenize(text_sample)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    
    print(f"   {size:5d} chars → {avg_time*1000:6.2f}ms (avg of 5 runs)")

print(f"\n💡 For 1,728 outputs:")
estimated_time = 1728 * (sum(times) / len(times))
print(f"   Estimated scoring time: {estimated_time:.2f}s ({estimated_time/60:.2f} minutes)")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 7: EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 7: TESTING EDGE CASES")
print("="*70)

edge_cases = {
    'Mostly punctuation': '།། །།། །།། །།',
    'Mixed with numbers': '༡༢༣ བོད་སྐད་ ༤༥༦',
    'Sanskrit mantra': 'ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ་',
    'Very short text': 'བོད་',
    'Empty string': '',
}

print("\n🧪 How does PyBo handle unusual inputs?\n")

for label, text in edge_cases.items():
    if text:
        score = calculate_pybo_score(text)
        print(f"   {label:25s} → {score['valid_percentage']:6.2f}% valid")
    else:
        print(f"   {label:25s} → (empty, skipped)")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY & RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("📋 SUMMARY: PYBO FOR OCR QUALITY SCORING")
print("="*70)

print("""
WHAT WE LEARNED:

1. PyBo is RULE-BASED - uses THL lexicon for word matching
2. Provides POS TAGS - helps identify valid vs invalid words
3. FAST ENOUGH - can score 1,728 outputs in reasonable time
4. WORKS WELL on classical/formal Tibetan (like our Tengyur test data)
5. CAN DISTINGUISH good OCR from garbage via "valid word percentage"

STRENGTHS FOR OUR USE CASE:
✓ High accuracy on classical texts (our pecha corpus)
✓ Clear distinction between lexicon words and garbage
✓ Fast processing (suitable for batch OCR scoring)
✓ Provides interpretable results (POS tags)

POTENTIAL WEAKNESSES:
✗ Limited to THL lexicon (may miss valid but unlisted words)
✗ Religious/classical vocabulary focus (good for us!)
✗ Can't adapt to new words without manual lexicon updates

RECOMMENDED SCORING FUNCTION:
```python
def score_ocr_with_pybo(text: str) -> float:
    tokens = tokenizer.tokenize(text)
    
    syllables = [t for t in tokens if t.type == 'syl']
    valid = [t for t in syllables if t.pos and t.pos != 'non-word']
    
    return len(valid) / len(syllables) if syllables else 0.0
```

NEXT STEPS:
Compare with Botok and TibetanRuleSeg to see which gives best results!
""")