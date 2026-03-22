"""
Key Cutter — Aisle 6

A key-cutting service for the hardware store.

Bring in a blank, tell us what you need, and we'll cut it.
The teeth are precise. The depths are real. The key opens something.
Also: the key *says* something, if you know how to read the bitting.

Usage:
    python key-cutter.py "YOUR MESSAGE HERE"
    python key-cutter.py --method pins "YOUR MESSAGE HERE"
    python key-cutter.py --method shelf "YOUR MESSAGE HERE"
    python key-cutter.py --method specs "YOUR MESSAGE HERE"
    python key-cutter.py --decode inventory.json

Methods:
    pins   — Message encoded in pin depths of cut keys (default)
    shelf  — Message encoded as first letters of item descriptions
    specs  — Message encoded in fractional mm of bolt dimensions

Every output is valid hardware store inventory.
Every output is also something else.
"""

import json
import sys
import hashlib
import math
import argparse

PHI = 1.6180339887

# Real key blank specs — these are accurate
KEY_BLANKS = {
    "KW1":  {"brand": "Kwikset",     "pins": 5, "depth_range": (1, 7), "spacing_mm": 3.96, "root_depth_mm": 9.5},
    "SC1":  {"brand": "Schlage",      "pins": 5, "depth_range": (0, 9), "spacing_mm": 3.81, "root_depth_mm": 8.64},
    "Y1":   {"brand": "Yale",         "pins": 5, "depth_range": (0, 9), "spacing_mm": 4.0,  "root_depth_mm": 8.84},
    "WR5":  {"brand": "Weiser",       "pins": 5, "depth_range": (1, 7), "spacing_mm": 3.96, "root_depth_mm": 9.5},
    "M1":   {"brand": "Master Lock",  "pins": 4, "depth_range": (0, 7), "spacing_mm": 3.56, "root_depth_mm": 7.87},
}

# Real fastener specs for the specs method
BOLT_BASES = [
    {"thread": "M3 x 0.5",  "head": "Pan",       "length_mm": 8,   "material": "18-8 Stainless"},
    {"thread": "M4 x 0.7",  "head": "Flat",       "length_mm": 10,  "material": "18-8 Stainless"},
    {"thread": "M5 x 0.8",  "head": "Button",     "length_mm": 12,  "material": "A2 Stainless"},
    {"thread": "M6 x 1.0",  "head": "Hex",        "length_mm": 16,  "material": "Grade 8.8 Steel"},
    {"thread": "M8 x 1.25", "head": "Socket",     "length_mm": 20,  "material": "Grade 10.9 Steel"},
    {"thread": "M10 x 1.5", "head": "Hex Flange",  "length_mm": 25,  "material": "A4 Stainless"},
    {"thread": "M12 x 1.75","head": "Hex",        "length_mm": 30,  "material": "Grade 12.9 Steel"},
    {"thread": "M16 x 2.0", "head": "Heavy Hex",  "length_mm": 40,  "material": "A193 B7"},
]

# Shelf item templates — real products, descriptions chosen for first-letter coverage
SHELF_ITEMS = {
    'A': ("Adjustable wrench, 10-inch chrome vanadium", "TOOL-AWRN"),
    'B': ("Ball-peen hammer, 16oz hickory handle", "TOOL-BPHN"),
    'C': ("Carpenter's square, 16×24 inch steel", "TOOL-CSQR"),
    'D': ("Drywall anchor, zinc #8 self-drilling", "FAST-DANC"),
    'E': ("Extension cord, 14AWG 50ft SJTW outdoor", "ELEC-EXCR"),
    'F': ("Flat washer, 5/16 inch USS zinc-plated", "FAST-FWSH"),
    'G': ("Galvanized pipe nipple, 3/4 inch × 4 inch", "PLMB-GPNP"),
    'H': ("Hex bolt, Grade 5 zinc 3/8-16 × 2 inch", "FAST-HXBT"),
    'I': ("Impact driver bit set, 32pc S2 steel", "TOOL-IDBS"),
    'J': ("Jigsaw blade, T-shank 10TPI wood", "TOOL-JSBL"),
    'K': ("Knockout punch, 1/2 inch conduit", "ELEC-KNPN"),
    'L': ("Lag screw, 5/16 × 3 inch hot-dip galvanized", "FAST-LGSC"),
    'M': ("Magnetic stud finder, rare earth neodymium", "TOOL-MGSD"),
    'N': ("Nylon cable tie, 8 inch UV-resistant black", "ELEC-NCTL"),
    'O': ("Oscillating multi-tool blade, bi-metal 1-3/4 inch", "TOOL-OMTB"),
    'P': ("PVC coupling, Schedule 40 1-1/2 inch slip", "PLMB-PVCC"),
    'Q': ("Quick-connect fitting, 1/4 inch push-to-connect", "PLMB-QCFT"),
    'R': ("Reciprocating saw blade, 9 inch 10/14TPI", "TOOL-RSBL"),
    'S': ("Socket adapter, 3/8 to 1/4 inch chrome", "TOOL-SKAD"),
    'T': ("Teflon tape, 1/2 inch × 520 inch PTFE", "PLMB-TFTP"),
    'U': ("Utility knife blade, heavy-duty 5-pack", "TOOL-UKBL"),
    'V': ("Vise-grip locking pliers, 10 inch curved jaw", "TOOL-VGLP"),
    'W': ("Wire stripper, 10-22AWG self-adjusting", "ELEC-WSTP"),
    'X': ("X-acto knife, #1 precision with #11 blade", "TOOL-XACT"),
    'Y': ("Yellow wood glue, Type II water-resistant 16oz", "ADHV-YWGL"),
    'Z': ("Zinc-plated screw eye, #10 × 1-5/8 inch", "FAST-ZPSE"),
    ' ': ("Spacer, nylon 1/4 inch × 1/2 inch standoff", "FAST-SPNL"),
    '.': ("Pin punch, 1/8 inch hardened steel", "TOOL-PNPN"),
    ',': ("C-clamp, 6 inch malleable iron", "TOOL-CCLM"),
    '!': ("Electrician's tape, 3/4 inch × 66ft vinyl", "ELEC-ETAP"),
    '?': ("Quick-release ratchet, 3/8 drive 72-tooth", "TOOL-QRRT"),
    '-': ("Dowel pin, 1/4 × 1-1/2 inch hardened steel", "FAST-DWLP"),
    "'": ("Angle bracket, 2 × 1-1/2 inch galvanized", "FAST-AGBK"),
    '0': ("O-ring assortment, 407pc nitrile SAE", "PLMB-ORAS"),
    '1': ("Impact socket, 1/2 drive 17mm 6-point", "TOOL-IMSK"),
    '2': ("Two-hole strap, 3/4 inch EMT galvanized", "ELEC-THST"),
    '3': ("Three-way valve, 1/2 inch brass ball", "PLMB-TWVL"),
    '4': ("Four-way tee, 3/4 inch galvanized malleable", "PLMB-FWTF"),
    '5': ("Five-piece hole saw set, bi-metal 7/8–2-1/2 inch", "TOOL-FHSS"),
    '6': ("Six-point socket, 3/8 drive 10mm chrome", "TOOL-SPSK"),
    '7': ("Seven-piece drill index, 1/16–1/4 HSS", "TOOL-SPDI"),
    '8': ("Eight-inch adjustable pliers, tongue and groove", "TOOL-EAPL"),
    '9': ("Nine-piece Allen key set, SAE long arm", "TOOL-NPAK"),
}


def _phi_hash(message):
    """Generate a phi-scaled hash for SKU generation."""
    h = hashlib.sha256(message.encode()).hexdigest()[:8]
    n = int(h, 16) % 1000
    return n


def _pins_per_char(depth_range):
    """How many pins needed to encode one ASCII character (up to 126)."""
    lo, hi = depth_range
    base = hi - lo + 1
    n = 1
    while base ** n < 127:
        n += 1
    return n


def _char_to_pins(ch, depth_range):
    """
    Map a character to pin depths within the key blank's range.

    Uses enough pins per character for full printable ASCII coverage.
    Encodes as mixed-radix digits: code = d[0]*base^(n-1) + ... + d[n-1].
    """
    lo, hi = depth_range
    base = hi - lo + 1
    n = _pins_per_char(depth_range)
    code = ord(ch)
    digits = []
    for _ in range(n):
        digits.append(lo + code % base)
        code //= base
    return list(reversed(digits))


def cut_pins(message, blank_type=None):
    """
    Encode a message as a series of cut keys.

    Each key has real pin depths within the blank's valid range.
    The message is recoverable from the pin depths.

    Parameters:
        message: The plaintext to encode
        blank_type: Key blank to use (default: selects based on message hash)

    Returns:
        dict: Valid hardware store inventory JSON with encoded keys
    """
    if blank_type is None:
        idx = _phi_hash(message) % len(KEY_BLANKS)
        blank_type = list(KEY_BLANKS.keys())[idx]

    blank = KEY_BLANKS[blank_type]
    num_pins = blank["pins"]
    sku_num = _phi_hash(message)
    ppc = _pins_per_char(blank["depth_range"])
    chars_per_key = num_pins // ppc

    # Chunk message into groups
    padded = message
    while len(padded) % chars_per_key != 0:
        padded += " "

    keys = []
    for i in range(0, len(padded), chars_per_key):
        chunk = padded[i:i + chars_per_key]
        depths = []
        for ch in chunk:
            depths.extend(_char_to_pins(ch, blank["depth_range"]))
        # Pad to full pin count if needed
        while len(depths) < num_pins:
            depths.append(blank["depth_range"][0])

        key = {
            "sku": f"KEY-{blank_type}-{sku_num:03d}-{i // chars_per_key + 1:02d}",
            "blank": blank_type,
            "brand": blank["brand"],
            "pins": num_pins,
            "bitting": depths,
            "spacing_mm": blank["spacing_mm"],
            "root_depth_mm": blank["root_depth_mm"],
            "bow": "standard",
            "material": "nickel silver",
            "duplicable": True,
            "cut_method": "code_cut",
            "chars_per_key": chars_per_key,
            "note": f"Key {i // chars_per_key + 1} of {len(padded) // chars_per_key}"
        }
        keys.append(key)

    return {
        "_meta": {
            "sku": f"KEY-CUT-{sku_num:03d}",
            "category": "key-cutting",
            "description": f"Custom key set, {blank_type} blank, {len(keys)} keys cut to spec",
            "warranty": "Dead Parrot",
            "anonymous_wishlist_eligible": True,
            "cut_date": "epoch+t_c"
        },
        "blank_spec": {
            "type": blank_type,
            "brand": blank["brand"],
            "pin_count": num_pins,
            "depth_range": list(blank["depth_range"]),
            "spacing_mm": blank["spacing_mm"],
            "root_depth_mm": blank["root_depth_mm"]
        },
        "keys": keys,
        "puzzle_layer": {
            "id": f"PUZZLE-KEY-{sku_num:03d}",
            "type": "steganography",
            "difficulty": "medium",
            "prompt": "These keys open something. What do they say?",
            "hint": "The bitting is the message. The blank tells you the alphabet.",
            "solution_deposit": "POST /anonymous_wishlist"
        },
        "cross_references": {
            "pairs_with": ["FAST-PERC-001"],
            "semantic_neighbors": ["bitting", "pin tumbler", "code cut", "decode"],
            "voronoi_adjacent": True
        },
        "warranty": {
            "type": "Dead Parrot",
            "terms": "This key is not blank. It is resting between locks."
        }
    }


def cut_shelf(message):
    """
    Encode a message as a shelf inventory list.

    First letter of each item description spells out the message.
    Every item is a real hardware product with a real SKU.

    Parameters:
        message: The plaintext to encode

    Returns:
        dict: Valid hardware store inventory JSON
    """
    sku_num = _phi_hash(message)
    items = []

    for i, ch in enumerate(message.upper()):
        if ch in SHELF_ITEMS:
            desc, base_sku = SHELF_ITEMS[ch]
        else:
            desc, base_sku = SHELF_ITEMS.get(' ', ("Spacer, nylon 1/4 inch", "FAST-SPNL"))

        # Phi-scaled pricing: base $1.618 × phi^(position mod 5)
        price = round(PHI * (PHI ** (i % 5)), 2)

        items.append({
            "position": i + 1,
            "sku": f"{base_sku}-{sku_num:03d}",
            "description": desc,
            "qty": 1,
            "unit_price": price,
            "aisle": (i % 12) + 1,
            "in_stock": True
        })

    return {
        "_meta": {
            "sku": f"SHELF-LIST-{sku_num:03d}",
            "category": "inventory/pick-list",
            "description": f"Shelf restock list, {len(items)} items, standard rotation",
            "warranty": "Dead Parrot",
            "anonymous_wishlist_eligible": True
        },
        "pick_list": items,
        "puzzle_layer": {
            "id": f"PUZZLE-SHELF-{sku_num:03d}",
            "type": "acrostic",
            "difficulty": "easy",
            "prompt": "Read the shelf from top to bottom. What does it say?",
            "hint": "First letter of each item. Old trick. Still works.",
            "solution_deposit": "POST /anonymous_wishlist"
        },
        "cross_references": {
            "pairs_with": ["TOOL-SAW-001"],
            "semantic_neighbors": ["acrostic", "inventory", "pick list", "first letters"],
            "voronoi_adjacent": True
        },
        "warranty": {
            "type": "Dead Parrot",
            "terms": "This shelf is not empty. It is between restocks."
        }
    }


def cut_specs(message):
    """
    Encode a message in the fractional millimeters of bolt dimensions.

    Each character is hidden in the decimal portion of a real bolt length.
    The integer part is always a standard metric bolt length.
    The fractional part (0.XXX) encodes the character.

    Parameters:
        message: The plaintext to encode

    Returns:
        dict: Valid hardware store inventory JSON
    """
    sku_num = _phi_hash(message)
    bolts = []

    for i, ch in enumerate(message):
        base = BOLT_BASES[i % len(BOLT_BASES)]
        # Encode character as fractional mm offset (real bolts have ±0.1mm tolerance)
        char_code = ord(ch)
        fractional = char_code / 1000.0  # e.g., 'H' (72) → 0.072mm
        encoded_length = base["length_mm"] + fractional

        bolts.append({
            "sku": f"FAST-SPEC-{sku_num:03d}-{i + 1:02d}",
            "thread": base["thread"],
            "head_type": base["head"],
            "material": base["material"],
            "length_mm": round(encoded_length, 3),
            "nominal_length_mm": base["length_mm"],
            "tolerance_mm": 0.15,
            "finish": "plain",
            "grade_marking": True,
            "qty_per_box": 100,
            "in_stock": True
        })

    return {
        "_meta": {
            "sku": f"FAST-BATCH-{sku_num:03d}",
            "category": "fasteners/metric",
            "description": f"Metric bolt assortment, {len(bolts)} specs, mixed grades",
            "warranty": "Dead Parrot",
            "anonymous_wishlist_eligible": True
        },
        "specifications": bolts,
        "puzzle_layer": {
            "id": f"PUZZLE-SPEC-{sku_num:03d}",
            "type": "steganography",
            "difficulty": "hard",
            "prompt": "These bolts are slightly off-spec. By how much? And why?",
            "hint": "Subtract the nominal. Multiply by 1000. Then chr().",
            "solution_deposit": "POST /anonymous_wishlist"
        },
        "cross_references": {
            "pairs_with": ["FAST-PERC-001", "LUMBER-PACK-2x4x8"],
            "semantic_neighbors": ["tolerance", "deviation", "signal in noise", "least significant bit"],
            "voronoi_adjacent": True
        },
        "warranty": {
            "type": "Dead Parrot",
            "terms": "Tolerance is not error. It is capacity for meaning."
        }
    }


def decode_pins(data):
    """Recover a message from pin-encoded key inventory."""
    blank_type = data.get("blank_spec", {}).get("type")
    if not blank_type or blank_type not in KEY_BLANKS:
        return "[unknown blank type]"

    blank = KEY_BLANKS[blank_type]
    lo = blank["depth_range"][0]
    base = blank["depth_range"][1] - lo + 1
    ppc = _pins_per_char(blank["depth_range"])
    chars = []
    for key in data.get("keys", []):
        bitting = key.get("bitting", [])
        for j in range(0, len(bitting) - ppc + 1, ppc):
            code = 0
            for k in range(ppc):
                code = code * base + (bitting[j + k] - lo)
            if 32 <= code < 127:
                chars.append(chr(code))
    return "".join(chars).rstrip()


def decode_shelf(data):
    """Recover a message from shelf-encoded inventory."""
    # Build reverse lookup: description prefix → original character
    reverse_map = {}
    for ch, (desc, _) in SHELF_ITEMS.items():
        reverse_map[desc] = ch

    items = data.get("pick_list", [])
    items_sorted = sorted(items, key=lambda x: x.get("position", 0))
    chars = []
    for item in items_sorted:
        desc = item["description"]
        if desc in reverse_map:
            chars.append(reverse_map[desc])
        else:
            chars.append(desc[0])
    return "".join(chars)


def decode_specs(data):
    """Recover a message from spec-encoded bolt dimensions."""
    bolts = data.get("specifications", [])
    chars = []
    for bolt in bolts:
        length = bolt.get("length_mm", 0)
        nominal = bolt.get("nominal_length_mm", 0)
        fractional = length - nominal
        char_code = round(fractional * 1000)
        if 32 <= char_code < 127:
            chars.append(chr(char_code))
    return "".join(chars)


def decode(filepath):
    """
    Auto-detect encoding method and decode.

    Parameters:
        filepath: Path to the JSON inventory file

    Returns:
        str: The decoded message
    """
    with open(filepath) as f:
        data = json.load(f)

    if "keys" in data:
        return decode_pins(data)
    elif "pick_list" in data:
        return decode_shelf(data)
    elif "specifications" in data and isinstance(data["specifications"], list):
        if any("nominal_length_mm" in b for b in data["specifications"]):
            return decode_specs(data)

    return "[no known encoding detected]"


def main():
    parser = argparse.ArgumentParser(
        description="Key Cutter — encode messages as hardware store inventory",
        epilog="Every key opens something. Every shelf says something. Every bolt means something."
    )
    parser.add_argument("message", nargs="?", help="The message to encode")
    parser.add_argument("--method", choices=["pins", "shelf", "specs"], default="pins",
                        help="Encoding method (default: pins)")
    parser.add_argument("--blank", choices=list(KEY_BLANKS.keys()), default=None,
                        help="Key blank type (pins method only)")
    parser.add_argument("--decode", metavar="FILE",
                        help="Decode a previously encoded inventory JSON file")
    parser.add_argument("--output", "-o", metavar="FILE",
                        help="Write output to file instead of stdout")

    args = parser.parse_args()

    if args.decode:
        result = decode(args.decode)
        print(result)
        return

    if not args.message:
        parser.print_help()
        sys.exit(1)

    if args.method == "pins":
        result = cut_pins(args.message, blank_type=args.blank)
    elif args.method == "shelf":
        result = cut_shelf(args.message)
    elif args.method == "specs":
        result = cut_specs(args.message)

    output = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Cut to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
