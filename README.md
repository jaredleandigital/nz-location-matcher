# NZ Location Phonetic Matcher

Match garbled or accented New Zealand location names to real locations using Caverphone phonetic encoding and fuzzy matching.

Built for STT (Speech-to-Text) and voice AI pipelines where Kiwi-accented place names get mangled by transcription engines.

## The Problem

When a Kiwi says "Karori", STT engines hear "Carry". When they say "Spreydon", the transcript reads "Sprayed Inn". This tool fixes that.

```
"oh tah rah"  ->  Otara (Auckland)
"Illinois"    ->  Ilam (Christchurch)
"Kashmir"     ->  Cashmere (Christchurch)
"Sprayed Inn" ->  Spreydon (Christchurch)
"Carry"       ->  Karori (Wellington)
```

## How It Works

5-layer matching pipeline, each layer catches what the previous one misses:

```
Input: "oh tah rah"
    |
Layer 1: Exact match (direct string lookup)
    | miss
Layer 2: Alias table (pre-seeded + learned mishearings)
    | HIT -> Otara
    |
Layer 3: Caverphone encoding (NZ-specific phonetic match)
Layer 4: Fuzzy match (SequenceMatcher, threshold 0.6)
Layer 5: Multi-word decomposition (join words, try combos)
```

### Caverphone 2.0

[Caverphone](https://en.wikipedia.org/wiki/Caverphone) is a phonetic encoding algorithm created at the University of Otago, New Zealand. It was designed specifically for NZ English, including Maori-origin words. Words that *sound* the same produce the same code.

```
Karori    -> KRRA111111
Cashmere  -> KSMA111111
Kashmir   -> KSMA111111  (matches Cashmere!)
Ashberton -> ASPTN11111
Ashburton -> ASPTN11111  (matches!)
```

Implemented from scratch in pure Python (~80 lines). No external dependencies.

## Installation

```bash
git clone https://github.com/jaredleandigital/nz-location-matcher.git
cd nz-location-matcher
```

**Zero dependencies** - uses only Python standard library (`re`, `json`, `argparse`, `difflib`).

## Usage

### Match a single input

```bash
python3 nz-location-matcher.py --match "oh tah rah"
# Match: Otara (Auckland, Auckland)
# Confidence: high | Method: alias

python3 nz-location-matcher.py --match "Ashberton"
# Match: Ashburton (Christchurch, Canterbury)
# Confidence: high | Method: caverphone
```

### Filter by city

```bash
python3 nz-location-matcher.py --match "Bromley" --city Christchurch
```

### JSON output

```bash
python3 nz-location-matcher.py --match "Kashmir" --json
```

```json
{
  "input": "Kashmir",
  "match": "Cashmere",
  "city": "Christchurch",
  "region": "Canterbury",
  "confidence": "high",
  "method": "alias",
  "alternatives": [],
  "caverphone": "KSMA111111"
}
```

### Scan call transcripts

Scans JSON transcript files for location mentions and matches them:

```bash
python3 nz-location-matcher.py --scan-transcript transcripts.json
```

Expected format: JSON array of objects with a `transcript_full` field.

### Alias management

```bash
# Add a new alias (persisted to data/nz-location-aliases.json)
python3 nz-location-matcher.py --add-alias "pines pass" "Pyes Pa"

# List all aliases (34 pre-seeded + user-added)
python3 nz-location-matcher.py --list-aliases
```

### Interactive mode

```bash
python3 nz-location-matcher.py --interactive
```

Commands in interactive mode:
- `:city Auckland` - filter to a city
- `:city clear` - remove filter
- `:alias garbled=Correct` - add alias
- `:code word` - show Caverphone code

### Show Caverphone code

```bash
python3 nz-location-matcher.py --caverphone "Otara"
# Caverphone("Otara") = ATRA111111
```

### Run tests

```bash
python3 nz-location-matcher.py --test
# Running 26 test cases...
# Results: 26/26 passed, 0 failed
```

## Location Database

~486 locations covering all major NZ regions:

| Region | Coverage |
|--------|----------|
| Auckland | ~100 suburbs |
| Wellington | ~60 suburbs |
| Christchurch/Canterbury | ~90 suburbs + towns |
| Hamilton/Waikato | ~35 suburbs |
| Tauranga/Bay of Plenty | ~25 suburbs |
| Dunedin/Otago | ~25 suburbs |
| + 10 more regions | Major towns |

## Pre-seeded Aliases

34 known STT mishearings from real call transcripts:

| STT Heard | Actual Location |
|-----------|-----------------|
| Illinois | Ilam |
| Lisbon | Leeston |
| Sprayed Inn | Spreydon |
| Kashmir | Cashmere |
| Carry / Curry | Karori |
| oh tah rah | Otara |
| Taronga | Tauranga |
| Tapuna | Takapuna |
| Kokora | Kaikoura |
| West Earthen | West Eyreton |
| Waikerie | Waikari |
| Metapie | Matapihi |

## Adding New Locations

The location database is hardcoded in the `NZ_LOCATIONS` list. To add locations:

```python
{"name": "Your Suburb", "city": "City", "region": "Region"},
```

## How Caverphone Helps

Traditional spell-checkers fail because STT errors are *sound-based*, not spelling-based:

| STT Output | Correct | Spell-check? | Caverphone? |
|------------|---------|:---:|:---:|
| Ashberton | Ashburton | No | Yes |
| Roleston | Rolleston | Maybe | Yes |
| Temaru | Timaru | No | Yes |
| Papamoah | Papamoa | No | Yes |

For completely unrelated words (Illinois -> Ilam), the alias table handles it.

## License

MIT
