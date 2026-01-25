"""
═══════════════════════════════════════════════════════════════════════════════
Botok (བོད་ཏོག) - Tibetan Word Tokenizer - Deep Dive Exploration
═══════════════════════════════════════════════════════════════════════════════

WHAT IS BOTOK?
Botok (pronounced [pʰøtɔk̚]) is a state-of-the-art word tokenizer for Tibetan.
It's more advanced than PyBo and provides multiple tokenization modes, POS tagging,
lemmatization, and support for custom dialects. Developed by OpenPecha.

HOW IT WORKS - THE CORE ALGORITHM:
1. Dictionary-Based Matching: Uses a Trie data structure built from multiple lexicons
2. Hierarchical Rule Templates: Applies layered grammar rules for normalization
3. Affix Handling: Can split or keep affixed particles (ར་, ས་, འི་, etc.)
4. Lemmatization: Reduces words to their base forms
5. Multi-Mode: Offers word, chunk, and space-based tokenization

THE TRIE DATA STRUCTURE (ENHANCED):
Like PyBo, Botok uses a trie, but it's MORE sophisticated:
- Supports multiple dictionaries simultaneously
- Includes frequency information
- Handles morphological variations
- Allows custom dialect additions

Root
├─ བ
│  ├─ ད [freq:100, lemma:བད]
│  └─ དེ [freq:500, lemma:བདེ, POS:ADJ]
│       └─ ་ལེགས [complete: བདེ་ལེགས, POS:NOUN]
└─ ག
   └─ ནས [freq:200, lemma:གནས, POS:VERB/NOUN]

WHAT MAKES IT "DICTIONARY-ASSISTED"?
- Combines multiple Tibetan lexicons (Grand Monlam, etc.)
- Orthographic normalization (fixes spelling variations)
- Layered grammar rules for segmentation and cleaning
- Dialect-specific vocabulary support

KEY DIFFERENCES FROM PYBO:
┌────────────────────┬─────────────────────┬──────────────────────┐
│     Feature        │       PyBo          │       Botok          │
├────────────────────┼─────────────────────┼──────────────────────┤
│ Lexicon Source     │ THL only            │ Multiple dictionaries│
│ Tokenization Modes │ One mode            │ Word/Chunk/Space     │
│ Lemmatization      │ No                  │ Yes                  │
│ Affix Handling     │ Basic               │ Configurable         │
│ Custom Dialects    │ No                  │ Yes                  │
│ Spell Normalization│ No                  │ Yes                  │
└────────────────────┴─────────────────────┴──────────────────────┘

STRENGTHS:
✓ More comprehensive dictionary coverage (multiple lexicons)
✓ Flexible tokenization modes
✓ Lemmatization support (groups word forms)
✓ Can handle modern AND classical Tibetan
✓ Custom dialect support
✓ Orthographic normalization

WEAKNESSES:
✗ Slower than PyBo (more processing steps)
✗ More complex to configure
✗ Still depends on dictionary coverage
✗ May over-normalize spelling in some cases

USE CASE FOR OCR QUALITY SCORING:
Botok's multi-dictionary approach might catch MORE valid words than PyBo,
potentially giving us better discrimination between good and bad OCR.
The lemmatization feature could also help group variant forms.

TOKENIZATION MODES EXPLAINED:
1. Word Mode: Segments text into complete words (most common)
2. Chunk Mode: Groups "meaningful character chunks" (syllables + context)
3. Space Mode: Simple space-based splitting (fastest, least accurate)

Author: Matt
Date: 2026-01-25
Purpose: Explore Botok's capabilities for potential use in OCR quality scoring
═══════════════════════════════════════════════════════════════════════════════
"""

# %%
# ═══════════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════
# Run this in your terminal (not in the notebook):
# pip install botok --break-system-packages
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
    """Load clean Tibetan text from Tengyur JSON file."""
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
# INITIALIZE BOTOK TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════

try:
    from botok import WordTokenizer
    
    print("✅ Botok imported successfully")
    print("\n" + "="*70)
    print("INITIALIZING BOTOK TOKENIZER")
    print("="*70)
    
    # Initialize with default configuration
    # This loads the dictionary and builds the Trie
    wt = WordTokenizer()
    
    print("✓ Tokenizer initialized with default settings")
    print("  Default dictionary: Grand Monlam (plus others)")
    print("  Mode: Word tokenization")
    print("  Affixes: Not split by default")
    
except ImportError as e:
    print(f"❌ Botok not installed: {e}")
    print("   Install with: pip install botok --break-system-packages")
    exit()

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: BASIC TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 1: BASIC WORD TOKENIZATION")
print("="*70)

# Take a small sample
sample_text = test_texts[0][:200]
print(f"\n📝 Input text (first 200 chars):")
print(f"   {sample_text}")

# Tokenize
tokens = wt.tokenize(sample_text)

print(f"\n🔍 Botok found {len(tokens)} tokens")
print(f"\n💡 First 10 tokens:")
for i, token in enumerate(tokens[:10]):
    token_text = getattr(token, 'text', str(token))
    token_pos = getattr(token, 'pos', 'N/A')
    token_type = getattr(token, 'chunk_type', 'N/A')
    print(f"   {i+1}. '{token_text}' (type: {token_type}, POS: {token_pos})")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: UNDERSTANDING TOKEN ATTRIBUTES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 2: UNDERSTANDING BOTOK TOKEN ATTRIBUTES")
print("="*70)

print("\nBotok tokens have rich attributes:")
print("Let's examine one token in detail:\n")

# Find a word token
word_tokens = [t for t in tokens if hasattr(t, 'text') and len(t.text) > 1]
if word_tokens:
    example = word_tokens[0]
    
    print(f"Token: '{example.text}'")
    print(f"  .text        = '{example.text}'")
    
    if hasattr(example, 'pos'):
        print(f"  .pos         = '{example.pos}'          (part of speech)")
    
    if hasattr(example, 'lemma'):
        print(f"  .lemma       = '{example.lemma}'        (base form)")
    
    if hasattr(example, 'chunk_type'):
        print(f"  .chunk_type  = '{example.chunk_type}'   (token category)")
    
    if hasattr(example, 'freq'):
        print(f"  .freq        = {example.freq}           (corpus frequency)")
    
    if hasattr(example, 'tag'):
        print(f"  .tag         = '{example.tag}'          (grammatical tag)")
    
    print(f"\n💡 Key difference from PyBo: Botok provides LEMMA (base form)")
    print(f"   This helps group related word forms together")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: TOKENIZATION WITH DIFFERENT OPTIONS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 3: TOKENIZATION OPTIONS")
print("="*70)

# Test sentence
test_sentence = "བཀྲ་ཤིས་བདེ་ལེགས།"

print(f"\nTest sentence: {test_sentence}\n")

# Option 1: Default (affixes not split)
tokens_default = wt.tokenize(test_sentence, split_affixes=False)
print("1. Default (affixes NOT split):")
print(f"   {[t.text for t in tokens_default]}")

# Option 2: Split affixes
tokens_split = wt.tokenize(test_sentence, split_affixes=True)
print("\n2. With affixes SPLIT:")
print(f"   {[t.text for t in tokens_split]}")

print("\n💡 Affix splitting can help identify particle errors in OCR")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: LEMMATIZATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 4: LEMMATIZATION (GROUPING WORD FORMS)")
print("="*70)

print("\nLemmatization reduces words to their base form:")
print("Example: བཀྲ་ཤིས → བཀྲ་ཤིས (base)")
print("         བཀྲ་ཤིས་ཀྱི → བཀྲ་ཤིས (base, without genitive particle)\n")

# Tokenize with lemmatization enabled
sample = test_texts[0][:300]
tokens_with_lemma = wt.tokenize(sample, split_affixes=False)

# Show words with their lemmas
print("Word → Lemma pairs (first 10):")
count = 0
for token in tokens_with_lemma:
    if hasattr(token, 'lemma') and token.lemma and token.text != token.lemma:
        print(f"   {token.text:15s} → {token.lemma}")
        count += 1
        if count >= 10:
            break

if count == 0:
    print("   (No lemma variations found in this sample)")

print("\n💡 For OCR scoring: Lemmas help group variants, reducing false negatives")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: CALCULATING "VALID WORD PERCENTAGE" (OCR QUALITY SCORE)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 5: OCR QUALITY SCORING WITH BOTOK")
print("="*70)

def calculate_botok_score(text: str) -> Dict:
    """
    Calculate what percentage of text consists of valid Tibetan words.
    
    Uses Botok's dictionary to identify known vs unknown words.
    
    Returns:
        Dictionary with scoring metrics
    """
    start_time = time.time()
    
    tokens = wt.tokenize(text, split_affixes=False)
    
    total_tokens = 0
    valid_tokens = 0
    unknown_tokens = 0
    
    for token in tokens:
        # Get attributes safely
        chunk_type = getattr(token, 'chunk_type', None)
        pos = getattr(token, 'pos', None)
        
        # Skip ONLY punctuation (empty POS)
        if chunk_type == 'PUNCT' or pos == '':
            continue
        
        total_tokens += 1
        
        # A token is "valid" if it has a real POS tag
        # Check for known invalid markers
        if pos and pos not in ['NON_WORD', 'non-word', 'OOV', 'NO_POS', 'OTHER', '', None, 'X']:
            valid_tokens += 1
        else:
            unknown_tokens += 1
    
    valid_percentage = (valid_tokens / total_tokens * 100) if total_tokens > 0 else 0
    elapsed = time.time() - start_time
    
    return {
        'total_tokens': total_tokens,
        'valid_words': valid_tokens,
        'unknown_words': unknown_tokens,
        'valid_percentage': valid_percentage,
        'processing_time': elapsed
    }

# Test on known-good text
good_score = calculate_botok_score(test_texts[0])

print("\n📊 Scoring known-good Tengyur text:")
print(f"   Total tokens:       {good_score['total_tokens']}")
print(f"   Valid words:        {good_score['valid_words']}")
print(f"   Unknown words:      {good_score['unknown_words']}")
print(f"   ✨ SCORE:           {good_score['valid_percentage']:.2f}%")
print(f"   ⏱️  Processing time:  {good_score['processing_time']:.4f}s")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: SIMULATING OCR ERRORS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 6: TESTING ON CORRUPTED TEXT (SIMULATED BAD OCR)")
print("="*70)

def corrupt_text(text: str, corruption_rate: float = 0.2) -> str:
    """Randomly corrupt text to simulate OCR errors."""
    import random
    
    chars = list(text)
    num_to_corrupt = int(len(chars) * corruption_rate)
    
    garbage = ['ཀ', 'ག', 'ང', 'ཅ', 'ཇ', 'ཏ', 'ད', 'ན', 'པ', 'བ', 'མ', '་', '།']
    
    positions = random.sample(range(len(chars)), min(num_to_corrupt, len(chars)))
    
    for pos in positions:
        chars[pos] = random.choice(garbage)
    
    return ''.join(chars)

# Test with different corruption levels
sample = test_texts[0][:500]

print("\n🧪 Testing different OCR quality levels:\n")

# DIAGNOSTIC: Show what corruption looks like and how Botok handles it
print("DIAGNOSTIC CHECK:")
corrupted = corrupt_text(sample, 0.5)
print(f"Original: {sample[:80]}")
print(f"Corrupted: {corrupted[:80]}")

tokens_orig = wt.tokenize(sample)
tokens_corrupt = wt.tokenize(corrupted)

print(f"\nFirst 5 original tokens with POS:")
for t in tokens_orig[:5]:
    print(f"  '{t.text}' → POS: {getattr(t, 'pos', 'NONE')}")

print(f"\nFirst 5 corrupted tokens with POS:")
for t in tokens_corrupt[:5]:
    print(f"  '{t.text}' → POS: {getattr(t, 'pos', 'NONE')}")
print()

test_cases = [
    ("Original (perfect)", sample, 0.0),
    ("20% corrupted", corrupt_text(sample, 0.2), 0.2),
    ("50% corrupted", corrupt_text(sample, 0.5), 0.5),
]

for label, text, rate in test_cases:
    score = calculate_botok_score(text)
    print(f"{label:25s} → Score: {score['valid_percentage']:6.2f}%")

print("\n💡 Botok can also distinguish good OCR from garbage!")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 7: PERFORMANCE BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 7: PERFORMANCE BENCHMARKING")
print("="*70)

print("\n⏱️  How fast is Botok compared to PyBo?")
print("   (Important for processing 1,728 outputs per image!)\n")

sizes = [100, 500, 1000, 2000]

for size in sizes:
    text_sample = test_texts[0][:size]
    
    # Time multiple runs
    times = []
    for _ in range(5):
        start = time.time()
        wt.tokenize(text_sample)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    
    print(f"   {size:5d} chars → {avg_time*1000:6.2f}ms (avg of 5 runs)")

print(f"\n💡 For 1,728 outputs:")
estimated_time = 1728 * (sum(times) / len(times))
print(f"   Estimated scoring time: {estimated_time:.2f}s ({estimated_time/60:.2f} minutes)")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 8: COMPARING CHUNK VS WORD TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 8: TOKENIZATION MODE COMPARISON")
print("="*70)

# Botok supports multiple tokenization modes
# Let's compare them

from botok import Text

test_snippet = test_texts[0][:150]

print(f"\nInput text: {test_snippet[:80]}...\n")

# Create Text object
t = Text(test_snippet)

# Mode 1: Word tokenization
try:
    words = t.tokenize_words_raw_text
    print(f"1. Word mode: {len(words.split())} tokens")
    print(f"   Sample: {words[:100]}...")
except Exception as e:
    print(f"1. Word mode: Error - {e}")

# Mode 2: Chunk tokenization  
try:
    chunks = t.tokenize_chunks_plaintext
    print(f"\n2. Chunk mode: {len(chunks.split())} tokens")
    print(f"   Sample: {chunks[:100]}...")
except Exception as e:
    print(f"\n2. Chunk mode: Error - {e}")

# Mode 3: Space-based tokenization
try:
    spaces = t.tokenize_on_spaces
    print(f"\n3. Space mode: {len(spaces.split())} tokens")
    print(f"   Sample: {spaces[:100]}...")
except Exception as e:
    print(f"\n3. Space mode: Error - {e}")

print("\n💡 For OCR scoring, WORD mode is most useful")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 9: EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXAMPLE 9: TESTING EDGE CASES")
print("="*70)

edge_cases = {
    'Mostly punctuation': '།། །།། །།། །།',
    'Mixed with numbers': '༡༢༣ བོད་སྐད་ ༤༥༦',
    'Sanskrit mantra': 'ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ་',
    'Very short': 'བོད་',
}

print("\n🧪 How does Botok handle unusual inputs?\n")

for label, text in edge_cases.items():
    try:
        score = calculate_botok_score(text)
        print(f"   {label:25s} → {score['valid_percentage']:6.2f}% valid")
    except Exception as e:
        print(f"   {label:25s} → Error: {e}")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY & RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("📋 SUMMARY: BOTOK FOR OCR QUALITY SCORING")
print("="*70)

print("""
WHAT WE LEARNED:

1. Botok is DICTIONARY-ASSISTED - uses multiple comprehensive lexicons
2. MULTI-MODE TOKENIZATION - word/chunk/space modes available
3. LEMMATIZATION SUPPORT - groups word variants (helpful for OCR)
4. RICHER ATTRIBUTES - provides more linguistic information than PyBo
5. FLEXIBLE CONFIGURATION - can adapt to different Tibetan dialects

STRENGTHS FOR OUR USE CASE:
✓ Better dictionary coverage (multiple lexicons merged)
✓ Lemmatization reduces false negatives from OCR variants
✓ Handles both classical AND modern Tibetan
✓ Rich token attributes for analysis
✓ Still reasonably fast for batch processing

POTENTIAL WEAKNESSES:
✗ Slightly slower than PyBo (more processing)
✗ More complex configuration options
✗ Still dictionary-dependent (can't handle completely new words)

COMPARISON WITH PYBO:
- Botok likely has BETTER recall (finds more valid words)
- PyBo might be slightly FASTER
- Botok provides RICHER linguistic information
- Both work well on classical texts

RECOMMENDED SCORING FUNCTION:
```python
def score_ocr_with_botok(text: str) -> float:
    tokens = wt.tokenize(text, split_affixes=False)
    
    valid_tokens = [t for t in tokens 
                   if hasattr(t, 'pos') and t.pos
                   and 'OOV' not in str(t.pos)]
    
    all_tokens = [t for t in tokens
                 if hasattr(t, 'chunk_type')
                 and t.chunk_type not in ['PUNCT', 'NON_WORD']]
    
    return len(valid_tokens) / len(all_tokens) if all_tokens else 0.0
```

NEXT STEPS:
Compare with TibetanRuleSeg to see all three approaches side-by-side!
""")