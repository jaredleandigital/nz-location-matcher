#!/usr/bin/env python3
"""
NZ Location Phonetic Matcher - Matches garbled/accented NZ location names
to real locations using Caverphone phonetic encoding and fuzzy matching.

Solves the problem of STT engines mangling Kiwi-accented place names
(e.g. "oh tah rah" -> Otara, "Sprayed Inn" -> Spreydon).

Uses a 5-layer matching pipeline:
  1. Exact match (direct string lookup)
  2. Alias table (pre-seeded + learned mishearings)
  3. Caverphone encoding (NZ-specific phonetic match)
  4. Fuzzy match (difflib.SequenceMatcher, threshold 0.6)
  5. Multi-word decomposition (join words, try combos)

Usage:
    python3 nz-location-matcher.py --match "oh tah rah"
    python3 nz-location-matcher.py --match "carry" --city Wellington
    python3 nz-location-matcher.py --scan-transcript /path/to/transcripts.json
    python3 nz-location-matcher.py --add-alias "oh tah rah" "Otara"
    python3 nz-location-matcher.py --list-aliases
    python3 nz-location-matcher.py --test
    python3 nz-location-matcher.py --interactive
"""

import argparse
import difflib
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Caverphone 2.0 - NZ-specific phonetic encoding (David Hood, Otago University)
# Pure Python implementation - no external dependencies
# ---------------------------------------------------------------------------

def caverphone2(word):
    """
    Caverphone 2.0 phonetic encoding algorithm.
    Returns a 10-character uppercase code padded with '1's.
    Designed for New Zealand English including Maori-origin words.
    """
    if not word:
        return "1111111111"

    # 1. Lowercase
    txt = word.lower()

    # 2. Remove anything not a-z
    txt = re.sub(r'[^a-z]', '', txt)
    if not txt:
        return "1111111111"

    # 3. Remove trailing 'e'
    txt = re.sub(r'e$', '', txt)
    if not txt:
        return "1111111111"

    # 4. Starting patterns
    txt = re.sub(r'^cough', 'cou2f', txt)
    txt = re.sub(r'^rough', 'rou2f', txt)
    txt = re.sub(r'^tough', 'tou2f', txt)
    txt = re.sub(r'^enough', 'enou2f', txt)
    txt = re.sub(r'^trough', 'trou2f', txt)
    txt = re.sub(r'^gn', '2n', txt)

    # 5. Ending pattern
    txt = re.sub(r'mb$', 'm2', txt)

    # 6. Consonant cluster replacements (order matters)
    txt = txt.replace('cq', '2q')
    txt = txt.replace('ci', 'si')
    txt = txt.replace('ce', 'se')
    txt = txt.replace('cy', 'sy')
    txt = txt.replace('tch', '2ch')
    txt = txt.replace('c', 'k')
    txt = txt.replace('q', 'k')
    txt = txt.replace('x', 'k')
    txt = txt.replace('v', 'f')
    txt = txt.replace('dg', '2g')
    txt = txt.replace('tio', 'sio')
    txt = txt.replace('tia', 'sia')
    txt = txt.replace('d', 't')
    txt = txt.replace('ph', 'fh')
    txt = txt.replace('b', 'p')
    txt = txt.replace('sh', 's2')
    txt = txt.replace('z', 's')

    # 7. Initial vowel -> A
    txt = re.sub(r'^[aeiou]', 'A', txt)

    # 8. Remaining vowels -> 3
    txt = re.sub(r'[aeiou]', '3', txt)

    # 9. Handle j and y
    txt = txt.replace('j', 'y')
    txt = re.sub(r'^y3', 'Y3', txt)
    txt = re.sub(r'^y', '2', txt)
    txt = txt.replace('y', '2')

    # 10. Handle gh and g
    txt = txt.replace('3gh3', '3kh3')
    txt = txt.replace('gh', '22')
    txt = txt.replace('g', 'k')

    # 11. Collapse repeated consonants (one or more -> single uppercase)
    txt = re.sub(r's+', 'S', txt)
    txt = re.sub(r't+', 'T', txt)
    txt = re.sub(r'p+', 'P', txt)
    txt = re.sub(r'k+', 'K', txt)
    txt = re.sub(r'f+', 'F', txt)
    txt = re.sub(r'm+', 'M', txt)
    txt = re.sub(r'n+', 'N', txt)

    # 12. Handle w
    txt = txt.replace('w3', 'W3')
    txt = txt.replace('wh3', 'Wh3')
    txt = re.sub(r'w$', '3', txt)
    txt = txt.replace('w', '2')

    # 13. Handle h
    txt = re.sub(r'^h', 'A', txt)
    txt = txt.replace('h', '2')

    # 14. Handle r
    txt = txt.replace('r3', 'R3')
    txt = re.sub(r'r$', '3', txt)
    txt = txt.replace('r', '2')

    # 15. Handle l
    txt = txt.replace('l3', 'L3')
    txt = re.sub(r'l$', '3', txt)
    txt = txt.replace('l', '2')

    # 16. Clean up
    txt = txt.replace('2', '')      # Remove all placeholders
    txt = re.sub(r'3$', 'A', txt)   # Trailing 3 -> A
    txt = txt.replace('3', '')      # Remove remaining 3s

    # 17. Pad/truncate to 10 chars
    txt = (txt + '1111111111')[:10]

    return txt.upper()


# ---------------------------------------------------------------------------
# NZ Location Database
# Source of truth: data/nz-locations.json (1600+ entries from LINZ + OSM)
# ---------------------------------------------------------------------------

LOCATIONS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nz-locations.json")


def load_locations():
    """Load locations from JSON source of truth file."""
    try:
        with open(LOCATIONS_JSON) as f:
            data = json.load(f)
        locations = []
        for loc in data.get("locations", []):
            locations.append({
                "name": loc["name"],
                "city": loc.get("district", loc.get("city", "Unknown")),
                "region": loc.get("region", "Unknown"),
            })
        return locations
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load {LOCATIONS_JSON}: {e}", file=sys.stderr)
        print("Falling back to built-in location list.", file=sys.stderr)
        return NZ_LOCATIONS_FALLBACK


NZ_LOCATIONS_FALLBACK = [
    # Minimal fallback - only used if JSON file is missing
    {"name": "Auckland CBD", "city": "Auckland", "region": "Auckland"},
    {"name": "Ponsonby", "city": "Auckland", "region": "Auckland"},
    {"name": "Grey Lynn", "city": "Auckland", "region": "Auckland"},
    {"name": "Parnell", "city": "Auckland", "region": "Auckland"},
    {"name": "Newmarket", "city": "Auckland", "region": "Auckland"},
    {"name": "Epsom", "city": "Auckland", "region": "Auckland"},
    {"name": "Remuera", "city": "Auckland", "region": "Auckland"},
    {"name": "Mt Eden", "city": "Auckland", "region": "Auckland"},
    {"name": "Kingsland", "city": "Auckland", "region": "Auckland"},
    {"name": "Mt Albert", "city": "Auckland", "region": "Auckland"},
    {"name": "Sandringham", "city": "Auckland", "region": "Auckland"},
    {"name": "Mt Roskill", "city": "Auckland", "region": "Auckland"},
    {"name": "Onehunga", "city": "Auckland", "region": "Auckland"},
    {"name": "Penrose", "city": "Auckland", "region": "Auckland"},
    {"name": "Ellerslie", "city": "Auckland", "region": "Auckland"},
    {"name": "Greenlane", "city": "Auckland", "region": "Auckland"},
    {"name": "Mission Bay", "city": "Auckland", "region": "Auckland"},
    {"name": "St Heliers", "city": "Auckland", "region": "Auckland"},
    {"name": "Kohimarama", "city": "Auckland", "region": "Auckland"},
    {"name": "Orakei", "city": "Auckland", "region": "Auckland"},
    {"name": "Meadowbank", "city": "Auckland", "region": "Auckland"},
    {"name": "Glen Innes", "city": "Auckland", "region": "Auckland"},
    {"name": "Panmure", "city": "Auckland", "region": "Auckland"},
    {"name": "Mt Wellington", "city": "Auckland", "region": "Auckland"},
    {"name": "Sylvia Park", "city": "Auckland", "region": "Auckland"},
    {"name": "Otahuhu", "city": "Auckland", "region": "Auckland"},
    {"name": "Mangere", "city": "Auckland", "region": "Auckland"},
    {"name": "Mangere Bridge", "city": "Auckland", "region": "Auckland"},
    {"name": "Papatoetoe", "city": "Auckland", "region": "Auckland"},
    {"name": "Otara", "city": "Auckland", "region": "Auckland"},
    {"name": "Manukau", "city": "Auckland", "region": "Auckland"},
    {"name": "Manurewa", "city": "Auckland", "region": "Auckland"},
    {"name": "Papakura", "city": "Auckland", "region": "Auckland"},
    {"name": "Takanini", "city": "Auckland", "region": "Auckland"},
    {"name": "Drury", "city": "Auckland", "region": "Auckland"},
    {"name": "Pukekohe", "city": "Auckland", "region": "Auckland"},
    {"name": "Waiuku", "city": "Auckland", "region": "Auckland"},
    {"name": "Howick", "city": "Auckland", "region": "Auckland"},
    {"name": "Pakuranga", "city": "Auckland", "region": "Auckland"},
    {"name": "Botany", "city": "Auckland", "region": "Auckland"},
    {"name": "Flat Bush", "city": "Auckland", "region": "Auckland"},
    {"name": "East Tamaki", "city": "Auckland", "region": "Auckland"},
    {"name": "Half Moon Bay", "city": "Auckland", "region": "Auckland"},
    {"name": "Beachlands", "city": "Auckland", "region": "Auckland"},
    {"name": "Maraetai", "city": "Auckland", "region": "Auckland"},
    {"name": "Clevedon", "city": "Auckland", "region": "Auckland"},
    {"name": "Devonport", "city": "Auckland", "region": "Auckland"},
    {"name": "Takapuna", "city": "Auckland", "region": "Auckland"},
    {"name": "Milford", "city": "Auckland", "region": "Auckland"},
    {"name": "Birkenhead", "city": "Auckland", "region": "Auckland"},
    {"name": "Northcote", "city": "Auckland", "region": "Auckland"},
    {"name": "Glenfield", "city": "Auckland", "region": "Auckland"},
    {"name": "Albany", "city": "Auckland", "region": "Auckland"},
    {"name": "Browns Bay", "city": "Auckland", "region": "Auckland"},
    {"name": "Mairangi Bay", "city": "Auckland", "region": "Auckland"},
    {"name": "Torbay", "city": "Auckland", "region": "Auckland"},
    {"name": "Long Bay", "city": "Auckland", "region": "Auckland"},
    {"name": "Orewa", "city": "Auckland", "region": "Auckland"},
    {"name": "Whangaparaoa", "city": "Auckland", "region": "Auckland"},
    {"name": "Silverdale", "city": "Auckland", "region": "Auckland"},
    {"name": "Kumeu", "city": "Auckland", "region": "Auckland"},
    {"name": "Huapai", "city": "Auckland", "region": "Auckland"},
    {"name": "Henderson", "city": "Auckland", "region": "Auckland"},
    {"name": "Te Atatu", "city": "Auckland", "region": "Auckland"},
    {"name": "New Lynn", "city": "Auckland", "region": "Auckland"},
    {"name": "Avondale", "city": "Auckland", "region": "Auckland"},
    {"name": "Blockhouse Bay", "city": "Auckland", "region": "Auckland"},
    {"name": "Titirangi", "city": "Auckland", "region": "Auckland"},
    {"name": "Glen Eden", "city": "Auckland", "region": "Auckland"},
    {"name": "Kelston", "city": "Auckland", "region": "Auckland"},
    {"name": "Massey", "city": "Auckland", "region": "Auckland"},
    {"name": "Westgate", "city": "Auckland", "region": "Auckland"},
    {"name": "Hobsonville", "city": "Auckland", "region": "Auckland"},
    {"name": "Herald Island", "city": "Auckland", "region": "Auckland"},
    {"name": "Waiheke Island", "city": "Auckland", "region": "Auckland"},
    {"name": "Great Barrier Island", "city": "Auckland", "region": "Auckland"},
    {"name": "Wellsford", "city": "Auckland", "region": "Auckland"},
    {"name": "Warkworth", "city": "Auckland", "region": "Auckland"},
    {"name": "Matakana", "city": "Auckland", "region": "Auckland"},
    {"name": "Helensville", "city": "Auckland", "region": "Auckland"},
    {"name": "Riverhead", "city": "Auckland", "region": "Auckland"},
    {"name": "Swanson", "city": "Auckland", "region": "Auckland"},
    {"name": "Ranui", "city": "Auckland", "region": "Auckland"},
    {"name": "Sunnyvale", "city": "Auckland", "region": "Auckland"},
    {"name": "Te Atatu Peninsula", "city": "Auckland", "region": "Auckland"},
    {"name": "Point Chevalier", "city": "Auckland", "region": "Auckland"},
    {"name": "Westmere", "city": "Auckland", "region": "Auckland"},
    {"name": "Herne Bay", "city": "Auckland", "region": "Auckland"},
    {"name": "Freemans Bay", "city": "Auckland", "region": "Auckland"},
    {"name": "Grafton", "city": "Auckland", "region": "Auckland"},
    {"name": "Eden Terrace", "city": "Auckland", "region": "Auckland"},
    {"name": "Morningside", "city": "Auckland", "region": "Auckland"},
    {"name": "Waterview", "city": "Auckland", "region": "Auckland"},
    {"name": "Owairaka", "city": "Auckland", "region": "Auckland"},
    {"name": "Three Kings", "city": "Auckland", "region": "Auckland"},
    {"name": "Royal Oak", "city": "Auckland", "region": "Auckland"},
    {"name": "Hillsborough", "city": "Auckland", "region": "Auckland"},
    {"name": "Waikowhai", "city": "Auckland", "region": "Auckland"},
    {"name": "Lynfield", "city": "Auckland", "region": "Auckland"},
    {"name": "Mangere East", "city": "Auckland", "region": "Auckland"},
    {"name": "Favona", "city": "Auckland", "region": "Auckland"},
    {"name": "Clendon Park", "city": "Auckland", "region": "Auckland"},
    {"name": "Wattle Downs", "city": "Auckland", "region": "Auckland"},
    {"name": "Karaka", "city": "Auckland", "region": "Auckland"},

    # Wellington Region
    {"name": "Wellington CBD", "city": "Wellington", "region": "Wellington"},
    {"name": "Te Aro", "city": "Wellington", "region": "Wellington"},
    {"name": "Lambton Quay", "city": "Wellington", "region": "Wellington"},
    {"name": "Thorndon", "city": "Wellington", "region": "Wellington"},
    {"name": "Kelburn", "city": "Wellington", "region": "Wellington"},
    {"name": "Aro Valley", "city": "Wellington", "region": "Wellington"},
    {"name": "Brooklyn", "city": "Wellington", "region": "Wellington"},
    {"name": "Newtown", "city": "Wellington", "region": "Wellington"},
    {"name": "Berhampore", "city": "Wellington", "region": "Wellington"},
    {"name": "Island Bay", "city": "Wellington", "region": "Wellington"},
    {"name": "Owhiro Bay", "city": "Wellington", "region": "Wellington"},
    {"name": "Karori", "city": "Wellington", "region": "Wellington"},
    {"name": "Northland", "city": "Wellington", "region": "Wellington"},
    {"name": "Wadestown", "city": "Wellington", "region": "Wellington"},
    {"name": "Wilton", "city": "Wellington", "region": "Wellington"},
    {"name": "Crofton Downs", "city": "Wellington", "region": "Wellington"},
    {"name": "Ngaio", "city": "Wellington", "region": "Wellington"},
    {"name": "Khandallah", "city": "Wellington", "region": "Wellington"},
    {"name": "Johnsonville", "city": "Wellington", "region": "Wellington"},
    {"name": "Newlands", "city": "Wellington", "region": "Wellington"},
    {"name": "Churton Park", "city": "Wellington", "region": "Wellington"},
    {"name": "Tawa", "city": "Wellington", "region": "Wellington"},
    {"name": "Porirua", "city": "Wellington", "region": "Wellington"},
    {"name": "Titahi Bay", "city": "Wellington", "region": "Wellington"},
    {"name": "Plimmerton", "city": "Wellington", "region": "Wellington"},
    {"name": "Paremata", "city": "Wellington", "region": "Wellington"},
    {"name": "Mana", "city": "Wellington", "region": "Wellington"},
    {"name": "Pukerua Bay", "city": "Wellington", "region": "Wellington"},
    {"name": "Paekakariki", "city": "Wellington", "region": "Wellington"},
    {"name": "Paraparaumu", "city": "Wellington", "region": "Wellington"},
    {"name": "Waikanae", "city": "Wellington", "region": "Wellington"},
    {"name": "Otaki", "city": "Wellington", "region": "Wellington"},
    {"name": "Petone", "city": "Wellington", "region": "Wellington"},
    {"name": "Lower Hutt", "city": "Wellington", "region": "Wellington"},
    {"name": "Hutt Central", "city": "Wellington", "region": "Wellington"},
    {"name": "Waterloo", "city": "Wellington", "region": "Wellington"},
    {"name": "Eastbourne", "city": "Wellington", "region": "Wellington"},
    {"name": "Wainuiomata", "city": "Wellington", "region": "Wellington"},
    {"name": "Naenae", "city": "Wellington", "region": "Wellington"},
    {"name": "Taita", "city": "Wellington", "region": "Wellington"},
    {"name": "Stokes Valley", "city": "Wellington", "region": "Wellington"},
    {"name": "Upper Hutt", "city": "Wellington", "region": "Wellington"},
    {"name": "Silverstream", "city": "Wellington", "region": "Wellington"},
    {"name": "Trentham", "city": "Wellington", "region": "Wellington"},
    {"name": "Maungaraki", "city": "Wellington", "region": "Wellington"},
    {"name": "Korokoro", "city": "Wellington", "region": "Wellington"},
    {"name": "Hataitai", "city": "Wellington", "region": "Wellington"},
    {"name": "Mt Victoria", "city": "Wellington", "region": "Wellington"},
    {"name": "Oriental Bay", "city": "Wellington", "region": "Wellington"},
    {"name": "Roseneath", "city": "Wellington", "region": "Wellington"},
    {"name": "Evans Bay", "city": "Wellington", "region": "Wellington"},
    {"name": "Kilbirnie", "city": "Wellington", "region": "Wellington"},
    {"name": "Lyall Bay", "city": "Wellington", "region": "Wellington"},
    {"name": "Rongotai", "city": "Wellington", "region": "Wellington"},
    {"name": "Miramar", "city": "Wellington", "region": "Wellington"},
    {"name": "Seatoun", "city": "Wellington", "region": "Wellington"},
    {"name": "Strathmore Park", "city": "Wellington", "region": "Wellington"},
    {"name": "Karaka Bays", "city": "Wellington", "region": "Wellington"},
    {"name": "Kapiti Coast", "city": "Wellington", "region": "Wellington"},
    {"name": "Raumati", "city": "Wellington", "region": "Wellington"},

    # Christchurch / Canterbury
    {"name": "Christchurch CBD", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Riccarton", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Ilam", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Fendalton", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Merivale", "city": "Christchurch", "region": "Canterbury"},
    {"name": "St Albans", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Papanui", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Bishopdale", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Burnside", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Bryndwr", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Avonhead", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Russley", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Hornby", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Sockburn", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Wigram", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Halswell", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Spreydon", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Sydenham", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Addington", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Cashmere", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Huntsbury", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Sumner", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Redcliffs", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Ferrymead", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Woolston", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Opawa", "city": "Christchurch", "region": "Canterbury"},
    {"name": "St Martins", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Beckenham", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Somerfield", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Hillmorton", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Barrington", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Linwood", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Phillipstown", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Waltham", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Brighton", "city": "Christchurch", "region": "Canterbury"},
    {"name": "New Brighton", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Burwood", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Dallington", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Avonside", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Richmond", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Shirley", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Mairehau", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Casebrook", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Belfast", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Redwood", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Northwood", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Harewood", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Yaldhurst", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Templeton", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Prebbleton", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Lincoln", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Rolleston", "city": "Christchurch", "region": "Canterbury"},
    {"name": "West Melton", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Lyttelton", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Diamond Harbour", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Akaroa", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Kaiapoi", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Rangiora", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Woodend", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Pegasus", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Amberley", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Ashburton", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Timaru", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Bromley", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Aranui", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Wainoni", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Bexley", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Parklands", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Marshland", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Styx", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Spencerville", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Heathcote", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Governors Bay", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Hei Hei", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Islington", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Middleton", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Upper Riccarton", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Strowan", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Edgeware", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Waimairi Beach", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Bottle Lake", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Travis", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Queenspark", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Prestons", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Leeston", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Darfield", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Oxford", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Methven", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Geraldine", "city": "Christchurch", "region": "Canterbury"},
    {"name": "Temuka", "city": "Christchurch", "region": "Canterbury"},

    # Wider Canterbury
    {"name": "West Eyreton", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Ohoka", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Mandeville", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Sefton", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Loburn", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Waipara", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Cheviot", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Kaikoura", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Hanmer Springs", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Culverden", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Rakaia", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Hinds", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Pleasant Point", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Fairlie", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Twizel", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Tekapo", "city": "Canterbury", "region": "Canterbury"},

    # Hamilton / Waikato
    {"name": "Hamilton CBD", "city": "Hamilton", "region": "Waikato"},
    {"name": "Frankton", "city": "Hamilton", "region": "Waikato"},
    {"name": "Dinsdale", "city": "Hamilton", "region": "Waikato"},
    {"name": "Nawton", "city": "Hamilton", "region": "Waikato"},
    {"name": "Chartwell", "city": "Hamilton", "region": "Waikato"},
    {"name": "Rototuna", "city": "Hamilton", "region": "Waikato"},
    {"name": "Hillcrest", "city": "Hamilton", "region": "Waikato"},
    {"name": "Silverdale", "city": "Hamilton", "region": "Waikato"},
    {"name": "Melville", "city": "Hamilton", "region": "Waikato"},
    {"name": "Glenview", "city": "Hamilton", "region": "Waikato"},
    {"name": "Claudelands", "city": "Hamilton", "region": "Waikato"},
    {"name": "Hamilton East", "city": "Hamilton", "region": "Waikato"},
    {"name": "Queenwood", "city": "Hamilton", "region": "Waikato"},
    {"name": "Fairfield", "city": "Hamilton", "region": "Waikato"},
    {"name": "Hamilton Lake", "city": "Hamilton", "region": "Waikato"},
    {"name": "Te Rapa", "city": "Hamilton", "region": "Waikato"},
    {"name": "Pukete", "city": "Hamilton", "region": "Waikato"},
    {"name": "St Andrews", "city": "Hamilton", "region": "Waikato"},
    {"name": "Flagstaff", "city": "Hamilton", "region": "Waikato"},
    {"name": "Huntington", "city": "Hamilton", "region": "Waikato"},
    {"name": "Cambridge", "city": "Hamilton", "region": "Waikato"},
    {"name": "Te Awamutu", "city": "Hamilton", "region": "Waikato"},
    {"name": "Matamata", "city": "Hamilton", "region": "Waikato"},
    {"name": "Morrinsville", "city": "Hamilton", "region": "Waikato"},
    {"name": "Ngaruawahia", "city": "Hamilton", "region": "Waikato"},
    {"name": "Huntly", "city": "Hamilton", "region": "Waikato"},
    {"name": "Raglan", "city": "Hamilton", "region": "Waikato"},
    {"name": "Thames", "city": "Hamilton", "region": "Waikato"},
    {"name": "Paeroa", "city": "Hamilton", "region": "Waikato"},
    {"name": "Waihi", "city": "Hamilton", "region": "Waikato"},
    {"name": "Te Kauwhata", "city": "Hamilton", "region": "Waikato"},
    {"name": "Tokoroa", "city": "Hamilton", "region": "Waikato"},
    {"name": "Putaruru", "city": "Hamilton", "region": "Waikato"},

    # Tauranga / Bay of Plenty
    {"name": "Tauranga", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Mt Maunganui", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Papamoa", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Te Puke", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Bethlehem", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Otumoetai", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Greerton", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Pyes Pa", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Welcome Bay", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Gate Pa", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Hairini", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Matapihi", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Whakatane", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Rotorua", "city": "Rotorua", "region": "Bay of Plenty"},
    {"name": "Ohinemutu", "city": "Rotorua", "region": "Bay of Plenty"},
    {"name": "Ngongotaha", "city": "Rotorua", "region": "Bay of Plenty"},
    {"name": "Kawerau", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Opotiki", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Katikati", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Waihi Beach", "city": "Tauranga", "region": "Bay of Plenty"},

    # Dunedin / Otago
    {"name": "Dunedin CBD", "city": "Dunedin", "region": "Otago"},
    {"name": "St Kilda", "city": "Dunedin", "region": "Otago"},
    {"name": "South Dunedin", "city": "Dunedin", "region": "Otago"},
    {"name": "Caversham", "city": "Dunedin", "region": "Otago"},
    {"name": "St Clair", "city": "Dunedin", "region": "Otago"},
    {"name": "Andersons Bay", "city": "Dunedin", "region": "Otago"},
    {"name": "Musselburgh", "city": "Dunedin", "region": "Otago"},
    {"name": "Mosgiel", "city": "Dunedin", "region": "Otago"},
    {"name": "Green Island", "city": "Dunedin", "region": "Otago"},
    {"name": "Abbotsford", "city": "Dunedin", "region": "Otago"},
    {"name": "Mornington", "city": "Dunedin", "region": "Otago"},
    {"name": "Roslyn", "city": "Dunedin", "region": "Otago"},
    {"name": "Maori Hill", "city": "Dunedin", "region": "Otago"},
    {"name": "North East Valley", "city": "Dunedin", "region": "Otago"},
    {"name": "Port Chalmers", "city": "Dunedin", "region": "Otago"},
    {"name": "Ravensbourne", "city": "Dunedin", "region": "Otago"},
    {"name": "Oamaru", "city": "Dunedin", "region": "Otago"},
    {"name": "Wanaka", "city": "Queenstown", "region": "Otago"},
    {"name": "Queenstown", "city": "Queenstown", "region": "Otago"},
    {"name": "Arrowtown", "city": "Queenstown", "region": "Otago"},
    {"name": "Frankton", "city": "Queenstown", "region": "Otago"},
    {"name": "Cromwell", "city": "Queenstown", "region": "Otago"},
    {"name": "Alexandra", "city": "Queenstown", "region": "Otago"},
    {"name": "Balclutha", "city": "Dunedin", "region": "Otago"},
    {"name": "Milton", "city": "Dunedin", "region": "Otago"},
    {"name": "Palmerston", "city": "Dunedin", "region": "Otago"},
    {"name": "Clyde", "city": "Queenstown", "region": "Otago"},

    # Napier / Hawke's Bay
    {"name": "Napier", "city": "Napier", "region": "Hawke's Bay"},
    {"name": "Hastings", "city": "Napier", "region": "Hawke's Bay"},
    {"name": "Havelock North", "city": "Napier", "region": "Hawke's Bay"},
    {"name": "Taradale", "city": "Napier", "region": "Hawke's Bay"},
    {"name": "Greenmeadows", "city": "Napier", "region": "Hawke's Bay"},
    {"name": "Bay View", "city": "Napier", "region": "Hawke's Bay"},
    {"name": "Flaxmere", "city": "Napier", "region": "Hawke's Bay"},
    {"name": "Clive", "city": "Napier", "region": "Hawke's Bay"},
    {"name": "Waipukurau", "city": "Napier", "region": "Hawke's Bay"},
    {"name": "Wairoa", "city": "Napier", "region": "Hawke's Bay"},

    # Palmerston North / Manawatu
    {"name": "Palmerston North", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Hokowhitu", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Kelvin Grove", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Milson", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Roslyn", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Terrace End", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Awapuni", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Feilding", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Levin", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Foxton", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Shannon", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Dannevirke", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Woodville", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Marton", "city": "Palmerston North", "region": "Manawatu"},
    {"name": "Bulls", "city": "Palmerston North", "region": "Manawatu"},

    # New Plymouth / Taranaki
    {"name": "New Plymouth", "city": "New Plymouth", "region": "Taranaki"},
    {"name": "Bell Block", "city": "New Plymouth", "region": "Taranaki"},
    {"name": "Fitzroy", "city": "New Plymouth", "region": "Taranaki"},
    {"name": "Merrilands", "city": "New Plymouth", "region": "Taranaki"},
    {"name": "Vogeltown", "city": "New Plymouth", "region": "Taranaki"},
    {"name": "Westown", "city": "New Plymouth", "region": "Taranaki"},
    {"name": "Strandon", "city": "New Plymouth", "region": "Taranaki"},
    {"name": "Waitara", "city": "New Plymouth", "region": "Taranaki"},
    {"name": "Inglewood", "city": "New Plymouth", "region": "Taranaki"},
    {"name": "Stratford", "city": "New Plymouth", "region": "Taranaki"},
    {"name": "Hawera", "city": "New Plymouth", "region": "Taranaki"},
    {"name": "Eltham", "city": "New Plymouth", "region": "Taranaki"},
    {"name": "Opunake", "city": "New Plymouth", "region": "Taranaki"},

    # Whangarei / Northland
    {"name": "Whangarei", "city": "Whangarei", "region": "Northland"},
    {"name": "Kamo", "city": "Whangarei", "region": "Northland"},
    {"name": "Tikipunga", "city": "Whangarei", "region": "Northland"},
    {"name": "Onerahi", "city": "Whangarei", "region": "Northland"},
    {"name": "Regent", "city": "Whangarei", "region": "Northland"},
    {"name": "Morningside", "city": "Whangarei", "region": "Northland"},
    {"name": "Maunu", "city": "Whangarei", "region": "Northland"},
    {"name": "Kerikeri", "city": "Whangarei", "region": "Northland"},
    {"name": "Paihia", "city": "Whangarei", "region": "Northland"},
    {"name": "Kaitaia", "city": "Whangarei", "region": "Northland"},
    {"name": "Mangonui", "city": "Whangarei", "region": "Northland"},
    {"name": "Dargaville", "city": "Whangarei", "region": "Northland"},
    {"name": "Kawakawa", "city": "Whangarei", "region": "Northland"},
    {"name": "Moerewa", "city": "Whangarei", "region": "Northland"},
    {"name": "Ruakaka", "city": "Whangarei", "region": "Northland"},
    {"name": "Waipu", "city": "Whangarei", "region": "Northland"},

    # Nelson / Marlborough / Tasman
    {"name": "Nelson", "city": "Nelson", "region": "Nelson"},
    {"name": "Stoke", "city": "Nelson", "region": "Nelson"},
    {"name": "Tahunanui", "city": "Nelson", "region": "Nelson"},
    {"name": "Atawhai", "city": "Nelson", "region": "Nelson"},
    {"name": "Richmond", "city": "Nelson", "region": "Nelson"},
    {"name": "Motueka", "city": "Nelson", "region": "Nelson"},
    {"name": "Mapua", "city": "Nelson", "region": "Nelson"},
    {"name": "Takaka", "city": "Nelson", "region": "Nelson"},
    {"name": "Blenheim", "city": "Blenheim", "region": "Marlborough"},
    {"name": "Picton", "city": "Blenheim", "region": "Marlborough"},
    {"name": "Renwick", "city": "Blenheim", "region": "Marlborough"},
    {"name": "Havelock", "city": "Blenheim", "region": "Marlborough"},

    # Invercargill / Southland
    {"name": "Invercargill", "city": "Invercargill", "region": "Southland"},
    {"name": "Bluff", "city": "Invercargill", "region": "Southland"},
    {"name": "Oreti Beach", "city": "Invercargill", "region": "Southland"},
    {"name": "Otatara", "city": "Invercargill", "region": "Southland"},
    {"name": "Waikiwi", "city": "Invercargill", "region": "Southland"},
    {"name": "Richmond", "city": "Invercargill", "region": "Southland"},
    {"name": "Newfield", "city": "Invercargill", "region": "Southland"},
    {"name": "Kew", "city": "Invercargill", "region": "Southland"},
    {"name": "Gore", "city": "Invercargill", "region": "Southland"},
    {"name": "Winton", "city": "Invercargill", "region": "Southland"},
    {"name": "Riverton", "city": "Invercargill", "region": "Southland"},
    {"name": "Te Anau", "city": "Invercargill", "region": "Southland"},
    {"name": "Lumsden", "city": "Invercargill", "region": "Southland"},

    # Gisborne
    {"name": "Gisborne", "city": "Gisborne", "region": "Gisborne"},
    {"name": "Kaiti", "city": "Gisborne", "region": "Gisborne"},
    {"name": "Mangapapa", "city": "Gisborne", "region": "Gisborne"},
    {"name": "Te Hapara", "city": "Gisborne", "region": "Gisborne"},
    {"name": "Elgin", "city": "Gisborne", "region": "Gisborne"},
    {"name": "Wainui", "city": "Gisborne", "region": "Gisborne"},

    # Rotorua
    {"name": "Ngapuna", "city": "Rotorua", "region": "Bay of Plenty"},
    {"name": "Fairy Springs", "city": "Rotorua", "region": "Bay of Plenty"},
    {"name": "Kawaha Point", "city": "Rotorua", "region": "Bay of Plenty"},
    {"name": "Pukehangi", "city": "Rotorua", "region": "Bay of Plenty"},
    {"name": "Western Heights", "city": "Rotorua", "region": "Bay of Plenty"},
    {"name": "Fordlands", "city": "Rotorua", "region": "Bay of Plenty"},
    {"name": "Holdens Bay", "city": "Rotorua", "region": "Bay of Plenty"},

    # Taupo
    {"name": "Taupo", "city": "Taupo", "region": "Waikato"},
    {"name": "Acacia Bay", "city": "Taupo", "region": "Waikato"},
    {"name": "Wairakei", "city": "Taupo", "region": "Waikato"},
    {"name": "Turangi", "city": "Taupo", "region": "Waikato"},
    {"name": "Mangakino", "city": "Taupo", "region": "Waikato"},

    # Wanganui
    {"name": "Whanganui", "city": "Whanganui", "region": "Manawatu"},
    {"name": "Castlecliff", "city": "Whanganui", "region": "Manawatu"},
    {"name": "Gonville", "city": "Whanganui", "region": "Manawatu"},
    {"name": "St Johns Hill", "city": "Whanganui", "region": "Manawatu"},
    {"name": "Aramoho", "city": "Whanganui", "region": "Manawatu"},

    # West Coast
    {"name": "Greymouth", "city": "Greymouth", "region": "West Coast"},
    {"name": "Hokitika", "city": "Greymouth", "region": "West Coast"},
    {"name": "Westport", "city": "Greymouth", "region": "West Coast"},
    {"name": "Reefton", "city": "Greymouth", "region": "West Coast"},
    {"name": "Kumara", "city": "Greymouth", "region": "West Coast"},
    {"name": "Ross", "city": "Greymouth", "region": "West Coast"},

    # Major NZ towns/cities (standalone)
    {"name": "Tauranga", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Waikari", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Ngataha", "city": "Nelson", "region": "Nelson"},
    {"name": "Brookfield", "city": "Tauranga", "region": "Bay of Plenty"},
    {"name": "Otatara", "city": "Invercargill", "region": "Southland"},
    {"name": "Kaiwaka", "city": "Whangarei", "region": "Northland"},
    {"name": "Mangawhai", "city": "Whangarei", "region": "Northland"},
    {"name": "Maungaturoto", "city": "Whangarei", "region": "Northland"},
    {"name": "Waimate", "city": "Timaru", "region": "Canterbury"},
    {"name": "Omarama", "city": "Queenstown", "region": "Otago"},
    {"name": "Kurow", "city": "Oamaru", "region": "Otago"},
    {"name": "Twizel", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Lake Tekapo", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Hanmer Springs", "city": "Canterbury", "region": "Canterbury"},
    {"name": "Greytown", "city": "Wellington", "region": "Wellington"},
    {"name": "Martinborough", "city": "Wellington", "region": "Wellington"},
    {"name": "Carterton", "city": "Wellington", "region": "Wellington"},
    {"name": "Masterton", "city": "Wellington", "region": "Wellington"},
    {"name": "Featherston", "city": "Wellington", "region": "Wellington"},
    {"name": "Pahiatua", "city": "Wellington", "region": "Wellington"},
    {"name": "Eketahuna", "city": "Wellington", "region": "Wellington"},
    {"name": "Ohakune", "city": "Ruapehu", "region": "Manawatu"},
    {"name": "Taihape", "city": "Ruapehu", "region": "Manawatu"},
    {"name": "Whakapapa", "city": "Ruapehu", "region": "Manawatu"},
    {"name": "National Park", "city": "Ruapehu", "region": "Manawatu"},
    {"name": "Raetihi", "city": "Ruapehu", "region": "Manawatu"},
    {"name": "Waiouru", "city": "Ruapehu", "region": "Manawatu"},
    {"name": "Otorohanga", "city": "Hamilton", "region": "Waikato"},
    {"name": "Te Kuiti", "city": "Hamilton", "region": "Waikato"},
    {"name": "Taumarunui", "city": "Hamilton", "region": "Waikato"},
    {"name": "Whangamata", "city": "Hamilton", "region": "Waikato"},
    {"name": "Whitianga", "city": "Hamilton", "region": "Waikato"},
    {"name": "Coromandel", "city": "Hamilton", "region": "Waikato"},
    {"name": "Pauanui", "city": "Hamilton", "region": "Waikato"},
    {"name": "Tairua", "city": "Hamilton", "region": "Waikato"},
    {"name": "Waiuku", "city": "Auckland", "region": "Auckland"},
    {"name": "Tuakau", "city": "Auckland", "region": "Auckland"},
    {"name": "Pokeno", "city": "Auckland", "region": "Auckland"},
    {"name": "Port Waikato", "city": "Hamilton", "region": "Waikato"},
]

# Load from JSON source of truth (or fallback)
NZ_LOCATIONS = load_locations()


# ---------------------------------------------------------------------------
# Pre-seeded aliases (known STT mishearings from real call data)
# ---------------------------------------------------------------------------

PRE_SEEDED_ALIASES = {
    "illinois": "Ilam",
    "lisbon": "Leeston",
    "kokora": "Kaikoura",
    "taronga": "Tauranga",
    "tapuna": "Takapuna",
    "sprayed inn": "Spreydon",
    "kavisham": "Cashmere",
    "west earthen": "West Eyreton",
    "waikerie": "Waikari",
    "metapie": "Matapihi",
    "nongataha": "Ngataha",
    "kashmir": "Cashmere",
    "brookwood": "Brookfield",
    "oh tah rah": "Otara",
    "oh tara": "Otara",
    "otarah": "Otara",
    "carry": "Karori",
    "curry": "Karori",
    "carori": "Karori",
    "ta rah dale": "Taradale",
    "pap a new ee": "Papanui",
    "papa new ee": "Papanui",
    "lincon": "Lincoln",
    "roleston": "Rolleston",
    "temaru": "Timaru",
    "manuwera": "Manurewa",
    "manurera": "Manurewa",
    "riccarton": "Riccarton",
    "ricker ton": "Riccarton",
    "lynn wood": "Linwood",
    "horn bee": "Hornby",
    "some ner": "Sumner",
    "re mu era": "Remuera",
    "ponson bee": "Ponsonby",
    "one hung a": "Onehunga",
    "man a cow": "Manukau",
    "man you cow": "Manukau",
    "papa toe toe": "Papatoetoe",
    "taka poo na": "Takapuna",
}


# ---------------------------------------------------------------------------
# Build lookup tables
# ---------------------------------------------------------------------------

def build_lookup_tables(locations, city_filter=None):
    """
    Build name map and phonetic map from location database.

    Returns:
        name_map: {lowercase_name: location_dict}
        phonetic_map: {caverphone_code: [location_dict, ...]}
    """
    name_map = {}
    phonetic_map = {}

    seen = set()
    for loc in locations:
        if city_filter and loc["city"].lower() != city_filter.lower():
            continue

        name = loc["name"]
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)

        # Exact name lookup
        name_map[key] = loc

        # Caverphone encoding (join multi-word names)
        joined = re.sub(r'[^a-z]', '', key)
        code = caverphone2(joined)
        if code not in phonetic_map:
            phonetic_map[code] = []
        phonetic_map[code].append(loc)

    return name_map, phonetic_map


# ---------------------------------------------------------------------------
# Alias table management
# ---------------------------------------------------------------------------

ALIAS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nz-location-aliases.json")


def load_aliases():
    """Load alias table from disk, merge with pre-seeded aliases."""
    aliases = dict(PRE_SEEDED_ALIASES)
    try:
        with open(ALIAS_FILE) as f:
            user_aliases = json.load(f)
            aliases.update(user_aliases)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return aliases


def save_alias(garbled, correct):
    """Add a new alias to the persistent alias file."""
    try:
        with open(ALIAS_FILE) as f:
            aliases = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        aliases = {}

    aliases[garbled.lower().strip()] = correct.strip()

    os.makedirs(os.path.dirname(ALIAS_FILE), exist_ok=True)
    with open(ALIAS_FILE, 'w') as f:
        json.dump(aliases, f, indent=2, sort_keys=True)

    return aliases


def list_aliases():
    """List all aliases (pre-seeded + user-added)."""
    all_aliases = load_aliases()
    try:
        with open(ALIAS_FILE) as f:
            user_aliases = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user_aliases = {}

    return all_aliases, user_aliases


# ---------------------------------------------------------------------------
# Filler words to strip from STT output
# ---------------------------------------------------------------------------

FILLER_WORDS = {
    'in', 'at', 'near', 'around', 'the', 'im', 'its', 'just', 'out',
    'from', 'over', 'up', 'down', 'here', 'based', 'like', 'called',
    'suburb', 'area', 'place', 'of', 'yeah', 'yep', 'um', 'uh', 'ah',
    'so', 'well', 'actually', 'basically',
}


def clean_input(raw_text):
    """Strip filler words and non-alpha characters from STT output."""
    words = re.sub(r'[^a-zA-Z\s]', '', raw_text.lower()).split()
    cleaned = [w for w in words if w not in FILLER_WORDS]
    return ' '.join(cleaned)


# ---------------------------------------------------------------------------
# 5-Layer Matching Pipeline
# ---------------------------------------------------------------------------

def match_location(raw_input, name_map, phonetic_map, aliases, verbose=False):
    """
    Match garbled input to a real NZ location using 5-layer pipeline.

    Returns dict with: input, match, city, region, confidence, method, alternatives, caverphone
    """
    cleaned = clean_input(raw_input)
    if not cleaned:
        return _no_match(raw_input)

    if verbose:
        print(f"  Cleaned input: '{cleaned}'")

    # Layer 1: Exact match
    if cleaned in name_map:
        loc = name_map[cleaned]
        return _result(raw_input, loc, "high", "exact", caverphone2(re.sub(r'[^a-z]', '', cleaned)))

    if verbose:
        print("  Layer 1 (exact): miss")

    # Layer 2: Alias table
    if cleaned in aliases:
        target_name = aliases[cleaned]
        loc = _find_location_by_name(target_name, name_map)
        if loc:
            return _result(raw_input, loc, "high", "alias", caverphone2(re.sub(r'[^a-z]', '', cleaned)))
        # Alias exists but location not in DB - return partial result
        return {
            "input": raw_input, "match": target_name, "city": "Unknown",
            "region": "Unknown", "confidence": "high", "method": "alias",
            "alternatives": [], "caverphone": caverphone2(re.sub(r'[^a-z]', '', cleaned))
        }

    if verbose:
        print("  Layer 2 (alias): miss")

    # Layer 3: Caverphone phonetic match
    joined = re.sub(r'[^a-z]', '', cleaned)
    code = caverphone2(joined)

    if verbose:
        print(f"  Caverphone code: {code}")

    matches = phonetic_map.get(code, [])
    if len(matches) == 1:
        return _result(raw_input, matches[0], "high", "caverphone", code)
    if len(matches) > 1:
        # Pick best by string similarity
        best = min(matches, key=lambda m: -difflib.SequenceMatcher(
            None, cleaned, m["name"].lower()).ratio())
        alts = [m["name"] for m in matches if m["name"] != best["name"]]
        r = _result(raw_input, best, "medium", "caverphone", code)
        r["alternatives"] = alts
        return r

    if verbose:
        print("  Layer 3 (caverphone): miss")

    # Layer 4: Fuzzy match (SequenceMatcher)
    best_ratio = 0
    best_loc = None
    second_best = None
    all_names = list(name_map.keys())

    for name_key in all_names:
        ratio = difflib.SequenceMatcher(None, cleaned, name_key).ratio()
        if ratio > best_ratio:
            second_best = best_loc
            best_ratio = ratio
            best_loc = name_map[name_key]
        elif ratio > 0.5 and (second_best is None or ratio > difflib.SequenceMatcher(
                None, cleaned, second_best["name"].lower()).ratio()):
            second_best = name_map[name_key]

    if best_ratio >= 0.6:
        confidence = "high" if best_ratio >= 0.8 else "medium" if best_ratio >= 0.7 else "low"
        r = _result(raw_input, best_loc, confidence, "fuzzy", code)
        if second_best and best_ratio - difflib.SequenceMatcher(
                None, cleaned, second_best["name"].lower()).ratio() < 0.1:
            r["alternatives"] = [second_best["name"]]
        return r

    if verbose:
        print(f"  Layer 4 (fuzzy): miss (best ratio: {best_ratio:.2f})")

    # Layer 5: Multi-word decomposition
    words = cleaned.split()
    if len(words) > 1:
        # Try joining all words (e.g. "oh tah rah" -> "ohtahrah")
        joined_all = ''.join(words)
        code_joined = caverphone2(joined_all)
        matches = phonetic_map.get(code_joined, [])
        if matches:
            best = matches[0] if len(matches) == 1 else min(
                matches, key=lambda m: -difflib.SequenceMatcher(
                    None, joined_all, m["name"].lower()).ratio())
            confidence = "high" if len(matches) == 1 else "medium"
            r = _result(raw_input, best, confidence, "decomposition", code_joined)
            if len(matches) > 1:
                r["alternatives"] = [m["name"] for m in matches if m["name"] != best["name"]]
            return r

        # Try fuzzy match on the joined string
        best_ratio = 0
        best_loc = None
        for name_key in all_names:
            ratio = difflib.SequenceMatcher(None, joined_all, name_key.replace(' ', '')).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_loc = name_map[name_key]

        if best_ratio >= 0.6:
            confidence = "high" if best_ratio >= 0.8 else "medium" if best_ratio >= 0.7 else "low"
            return _result(raw_input, best_loc, confidence, "decomposition", code_joined)

        # Try subsets of words (first 2 words, last 2 words, etc.)
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                subset = ''.join(words[i:j])
                if len(subset) < 3:
                    continue
                sub_code = caverphone2(subset)
                sub_matches = phonetic_map.get(sub_code, [])
                if sub_matches:
                    best = sub_matches[0]
                    return _result(raw_input, best, "low", "decomposition", sub_code)

    if verbose:
        print("  Layer 5 (decomposition): miss")

    return _no_match(raw_input, code)


def _find_location_by_name(name, name_map):
    """Find a location by name (case-insensitive)."""
    key = name.lower()
    if key in name_map:
        return name_map[key]
    # Try partial match
    for k, v in name_map.items():
        if v["name"].lower() == key:
            return v
    return None


def _result(raw_input, loc, confidence, method, code=""):
    return {
        "input": raw_input,
        "match": loc["name"],
        "city": loc["city"],
        "region": loc["region"],
        "confidence": confidence,
        "method": method,
        "alternatives": [],
        "caverphone": code,
    }


def _no_match(raw_input, code=""):
    return {
        "input": raw_input,
        "match": None,
        "city": None,
        "region": None,
        "confidence": "none",
        "method": "none",
        "alternatives": [],
        "caverphone": code,
    }


# ---------------------------------------------------------------------------
# Transcript Scanner
# ---------------------------------------------------------------------------

# Regex patterns to find location context in transcripts
LOCATION_PATTERNS = [
    r"(?:i'?m|we'?re|i am|we are)\s+(?:in|from|based in|located in|over in)\s+([A-Za-z][A-Za-z\s]{1,30}?)(?:\.|,|$|\?|!)",
    r"(?:live|living|based|located|staying)\s+(?:in|at|near)\s+([A-Za-z][A-Za-z\s]{1,30}?)(?:\.|,|$|\?|!)",
    r"(?:come|coming)\s+from\s+([A-Za-z][A-Za-z\s]{1,30}?)(?:\.|,|$|\?|!)",
    r"(?:address is|located at)\s+\d+\s+\w+\s+(?:street|road|avenue|drive|place|way|lane|crescent|terrace)\s*[,.]?\s*([A-Za-z][A-Za-z\s]{1,30}?)(?:\.|,|$|\?|!)",
    r"(?:suburb|area|town)\s+(?:is|called|of)\s+([A-Za-z][A-Za-z\s]{1,30}?)(?:\.|,|$|\?|!)",
    r"(?:out in|down in|up in|over in)\s+([A-Za-z][A-Za-z\s]{1,30}?)(?:\.|,|$|\?|!)",
]


def scan_transcript(transcript_text, name_map, phonetic_map, aliases):
    """
    Scan a transcript for location mentions and try to match them.

    Returns list of match results.
    """
    results = []
    seen = set()

    for pattern in LOCATION_PATTERNS:
        for m in re.finditer(pattern, transcript_text, re.IGNORECASE):
            candidate = m.group(1).strip()
            # Clean up: remove trailing punctuation and common suffixes
            candidate = re.sub(r'[.,!?;:]+$', '', candidate).strip()
            # Skip very short or very long candidates
            if len(candidate) < 2 or len(candidate) > 40:
                continue
            # Skip if it's just numbers or a street address
            if re.match(r'^\d+\s', candidate):
                continue

            candidate_key = candidate.lower().strip()
            if candidate_key in seen:
                continue
            seen.add(candidate_key)

            # Skip common false positives
            skip_words = {'here', 'there', 'this', 'that', 'what', 'where', 'how',
                          'your', 'their', 'about', 'know', 'have', 'good', 'hello',
                          'yeah', 'alright', 'just', 'sort', 'kinda', 'like', 'really',
                          'been', 'okay', 'right', 'well', 'very', 'some', 'much',
                          'the morning', 'the afternoon', 'the evening', 'morning',
                          'afternoon', 'evening', 'the engine', 'engine', 'today',
                          'tomorrow', 'yesterday', 'the weekend', 'the car', 'the house',
                          'the office', 'work', 'home', 'town', 'the city'}
            if candidate_key in skip_words:
                continue
            # Also check the cleaned (filler-stripped) version
            candidate_cleaned = clean_input(candidate_key)
            if not candidate_cleaned:
                continue
            # Skip if any word in the cleaned candidate is a common non-location word
            cleaned_words = set(candidate_cleaned.split())
            if cleaned_words & skip_words:
                continue

            result = match_location(candidate, name_map, phonetic_map, aliases)
            # Only include medium+ confidence matches from transcript scanning
            if result["match"] and result["confidence"] in ("high", "medium"):
                result["context"] = m.group(0)
                results.append(result)

    return results


def scan_transcript_file(filepath, name_map, phonetic_map, aliases):
    """Scan a JSON transcript file (array of transcript objects)."""
    with open(filepath) as f:
        transcripts = json.load(f)

    all_results = []
    for t in transcripts:
        text = t.get("transcript_full", "")
        if not text:
            continue

        matches = scan_transcript(text, name_map, phonetic_map, aliases)
        if matches:
            for m in matches:
                m["recording_sid"] = t.get("recording_sid", "unknown")
                m["date"] = t.get("date", "unknown")
            all_results.extend(matches)

    return all_results


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

TEST_CASES = [
    # (input, expected_match, expected_method_or_any)
    ("Bromley", "Bromley", "exact"),
    ("Riccarton", "Riccarton", "exact"),
    ("Ilam", "Ilam", "exact"),
    ("Papanui", "Papanui", "exact"),
    ("Spreydon", "Spreydon", "exact"),

    # Alias matches
    ("Illinois", "Ilam", "alias"),
    ("Lisbon", "Leeston", "alias"),
    ("Sprayed Inn", "Spreydon", "alias"),
    ("oh tah rah", "Otara", "alias"),
    ("Kashmir", "Cashmere", "alias"),
    ("Carry", "Karori", "alias"),
    ("West Earthen", "West Eyreton", "alias"),
    ("Waikerie", "Waikari", "alias"),
    ("Metapie", "Matapihi", "alias"),
    ("Taronga", "Tauranga", "alias"),
    ("Tapuna", "Takapuna", "alias"),
    ("Kokora", "Kaikoura", "alias"),

    # Caverphone / fuzzy / decomposition matches
    ("Ashberton", "Ashburton", None),
    ("Lincon", "Lincoln", None),
    ("Roleston", "Rolleston", None),
    ("Temaru", "Timaru", None),
    ("Papamoah", "Papamoa", None),
    ("Otahuhu", "Otahuhu", "exact"),
    ("Manuwera", "Manurewa", None),
    ("Papakurra", "Papakura", None),
    ("Remuerra", "Remuera", None),
]


def run_tests():
    """Run all test cases and report results."""
    name_map, phonetic_map = build_lookup_tables(NZ_LOCATIONS)
    aliases = load_aliases()

    passed = 0
    failed = 0
    total = len(TEST_CASES)

    print(f"Running {total} test cases...\n")

    for raw_input, expected_match, expected_method in TEST_CASES:
        result = match_location(raw_input, name_map, phonetic_map, aliases)
        match_ok = result["match"] == expected_match
        method_ok = expected_method is None or result["method"] == expected_method

        if match_ok and method_ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        marker = "  " if status == "PASS" else ">>"
        print(f"  {marker} {status}: \"{raw_input}\" -> {result['match']} "
              f"(expected: {expected_match}) [{result['method']}]"
              + ("" if method_ok else f" (expected method: {expected_method})"))

    print(f"\nResults: {passed}/{total} passed, {failed} failed")
    return failed == 0


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_result(result, json_output=False, verbose=False):
    """Format a match result for display."""
    if json_output:
        return json.dumps(result, indent=2)

    if result["match"] is None:
        lines = [f"No match found for: {result['input']}"]
        if result["caverphone"]:
            lines.append(f"Caverphone: {result['caverphone']}")
        return '\n'.join(lines)

    lines = [
        f"Match: {result['match']} ({result['city']}, {result['region']})",
        f"Confidence: {result['confidence']} | Method: {result['method']}",
    ]
    if verbose:
        lines.append(f"Caverphone: {result['caverphone']}")
    if result.get("alternatives"):
        lines.append(f"Alternatives: {', '.join(result['alternatives'])}")
    if result.get("context"):
        lines.append(f"Context: \"{result['context']}\"")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode():
    """Interactive REPL for testing location matching."""
    name_map, phonetic_map = build_lookup_tables(NZ_LOCATIONS)
    aliases = load_aliases()

    print("NZ Location Matcher - Interactive Mode")
    print("Type a garbled location name, or 'quit' to exit.")
    print("Commands: :city CITY, :alias GARBLED=CORRECT, :code WORD\n")

    city_filter = None

    while True:
        try:
            raw = input(f"{'[' + city_filter + '] ' if city_filter else ''}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw or raw.lower() in ('quit', 'exit', 'q'):
            break

        # Commands
        if raw.startswith(':city'):
            city = raw[5:].strip()
            if city.lower() in ('all', 'none', 'clear', ''):
                city_filter = None
                name_map, phonetic_map = build_lookup_tables(NZ_LOCATIONS)
                print("City filter cleared.")
            else:
                city_filter = city
                name_map, phonetic_map = build_lookup_tables(NZ_LOCATIONS, city_filter)
                count = len(name_map)
                print(f"Filtered to {city}: {count} locations")
            continue

        if raw.startswith(':alias'):
            parts = raw[6:].strip()
            if '=' in parts:
                garbled, correct = parts.split('=', 1)
                save_alias(garbled.strip(), correct.strip())
                aliases = load_aliases()
                print(f"Added alias: \"{garbled.strip()}\" -> \"{correct.strip()}\"")
            else:
                print("Usage: :alias GARBLED=CORRECT")
            continue

        if raw.startswith(':code'):
            word = raw[5:].strip()
            print(f"Caverphone(\"{word}\") = {caverphone2(word)}")
            continue

        # Match
        result = match_location(raw, name_map, phonetic_map, aliases, verbose=True)
        print(format_result(result, verbose=True))
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NZ Location Phonetic Matcher - Match garbled location names to real NZ locations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --match "oh tah rah"
  %(prog)s --match "carry" --city Wellington
  %(prog)s --scan-transcript /path/to/transcripts.json
  %(prog)s --add-alias "oh tah rah" "Otara"
  %(prog)s --list-aliases
  %(prog)s --test
  %(prog)s --interactive
        """
    )

    # Modes
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--match', '-m', metavar='INPUT',
                      help='Match a single garbled input to a location')
    mode.add_argument('--scan-transcript', '-s', metavar='FILE',
                      help='Scan a JSON transcript file for location mentions')
    mode.add_argument('--add-alias', nargs=2, metavar=('GARBLED', 'CORRECT'),
                      help='Add a new alias mapping')
    mode.add_argument('--list-aliases', action='store_true',
                      help='List all known aliases')
    mode.add_argument('--test', '-t', action='store_true',
                      help='Run the built-in test suite')
    mode.add_argument('--interactive', '-i', action='store_true',
                      help='Start interactive matching mode')
    mode.add_argument('--caverphone', metavar='WORD',
                      help='Show the Caverphone code for a word')

    # Options
    parser.add_argument('--city', '-c', metavar='CITY',
                        help='Filter locations to a specific city (e.g. Auckland, Wellington, Christchurch)')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output results as JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed matching info')

    args = parser.parse_args()

    # --caverphone
    if args.caverphone:
        code = caverphone2(args.caverphone)
        if args.json:
            print(json.dumps({"word": args.caverphone, "code": code}))
        else:
            print(f"Caverphone(\"{args.caverphone}\") = {code}")
        return

    # --test
    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    # --interactive
    if args.interactive:
        interactive_mode()
        return

    # --add-alias
    if args.add_alias:
        garbled, correct = args.add_alias
        save_alias(garbled, correct)
        print(f"Added alias: \"{garbled}\" -> \"{correct}\"")
        return

    # --list-aliases
    if args.list_aliases:
        all_aliases, user_aliases = list_aliases()
        if args.json:
            print(json.dumps({"pre_seeded": PRE_SEEDED_ALIASES, "user_added": user_aliases, "total": len(all_aliases)}))
        else:
            print(f"Total aliases: {len(all_aliases)} ({len(PRE_SEEDED_ALIASES)} pre-seeded, {len(user_aliases)} user-added)\n")
            print("Pre-seeded aliases:")
            for k, v in sorted(PRE_SEEDED_ALIASES.items()):
                print(f"  \"{k}\" -> {v}")
            if user_aliases:
                print("\nUser-added aliases:")
                for k, v in sorted(user_aliases.items()):
                    print(f"  \"{k}\" -> {v}")
        return

    # Build lookup tables
    name_map, phonetic_map = build_lookup_tables(NZ_LOCATIONS, args.city)
    aliases = load_aliases()

    if args.verbose:
        print(f"Loaded {len(name_map)} locations" + (f" (filtered: {args.city})" if args.city else ""))
        print(f"Loaded {len(aliases)} aliases")
        print(f"Unique Caverphone codes: {len(phonetic_map)}\n")

    # --match
    if args.match:
        result = match_location(args.match, name_map, phonetic_map, aliases, verbose=args.verbose)
        print(format_result(result, json_output=args.json, verbose=args.verbose))
        return

    # --scan-transcript
    if args.scan_transcript:
        if not os.path.exists(args.scan_transcript):
            print(f"Error: File not found: {args.scan_transcript}", file=sys.stderr)
            sys.exit(1)

        results = scan_transcript_file(args.scan_transcript, name_map, phonetic_map, aliases)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            if not results:
                print("No location mentions found in transcripts.")
            else:
                print(f"Found {len(results)} location mentions:\n")
                for r in results:
                    sid = r.get("recording_sid", "")[:12]
                    date = r.get("date", "")[:10]
                    print(f"  [{sid}] {date}")
                    print(f"    {format_result(r, verbose=args.verbose)}")
                    print()


if __name__ == "__main__":
    main()
