# CLAUDE.md - Hardware-Store Repository Guide

## Overview

A puzzle archive disguised as a hardware inventory system. Every file works on at least two levels simultaneously: real hardware specifications and encoded mathematical/scientific puzzles. Built collaboratively and anonymously for "pattern-seers."

**License:** CC0 1.0 Universal (public domain)

## Repository Structure

```
Hardware-Store/
├── README.md                  # Main puzzle archive index
├── README.ai.md               # AI-focused entry point with API patterns
├── CLAUDE.md                  # This file
├── LICENSE                    # CC0 1.0 Universal
├── Bits/                      # Primary puzzle archive ("junk drawer")
│   ├── pieces.json            # UNI hardware lattice (Fibonacci/phi-encoded)
│   ├── manifest.json          # Master index for 4 lattices (UNI, ANM, CLR, SYMPH)
│   ├── percolation-fastener.json  # PUZZLE-001: Phase transition physics
│   ├── 2x4x8.json            # PUZZLE-002: 1D bin packing
│   ├── pipe-connect-042.json  # PUZZLE-003: Graph connectivity
│   ├── power-tools.json       # AI usage metaphors + real tool specs
│   ├── monty-python.json      # PUZZLE-004/005: Humor topology + category theory
│   ├── search-index.json      # Multi-layer TF-IDF search index
│   ├── color-pieces.json      # Chemistry lattice (CLR nodes)
│   ├── symphony-pieces.json   # Musical voice assignments (SYMPH nodes)
│   ├── animal-lattice.py      # 8 animal-derived connectivity geometries
│   ├── shadow_geometry.py     # Basin collapse modeling
│   ├── phi-mandala.html       # Interactive phi lattice visualization
│   └── bits/                  # Nested junk drawers (4 levels deep)
│       └── bits/bits/bits/    # Each level has README.md + pieces.json
└── inventory/                 # Organized aisles (breadcrumbs into Bits/)
    ├── fasteners/             # Aisle 3 → PUZZLE-001
    ├── lumber/                # Aisle 7 → PUZZLE-002
    ├── power-tools/           # Aisle 4 → tool metaphors
    ├── plumbing/              # Aisle 12 → PUZZLE-003
    └── electrical/            # Aisle 9 → OPEN SLOT for new puzzles
```

## Tech Stack

This is a data-driven puzzle repository, not a traditional software project. No build system, no CI/CD, no package manager.

- **Data format:** JSON exclusively for all specs and puzzles
- **Python 3.x:** For `animal-lattice.py` and `shadow_geometry.py` (requires `numpy`, `matplotlib`)
- **HTML/Canvas/JS:** For `phi-mandala.html` interactive visualization
- **Git:** Only persistence layer

## Key Data Conventions

### JSON Structure Pattern

All puzzle/data files follow this pattern:

```json
{
  "_meta": {
    "sku": "CATEGORY-NAME-###",
    "description": "Real specification + second meaning",
    "warranty": "Dead Parrot",
    "anonymous_wishlist_eligible": true
  },
  "specifications": { /* Real, accurate hardware specs */ },
  "puzzle_layer": {
    "id": "PUZZLE-###",
    "prompt": "The puzzle question",
    "hint": "Guidance",
    "difficulty": "easy|medium|hard",
    "solution_deposit": "POST /anonymous_wishlist"
  },
  "cross_references": {
    "pairs_with": [],
    "semantic_neighbors": [],
    "voronoi_adjacent": true
  }
}
```

### Node/SKU Naming

- **Lattice nodes:** `UNI###` (hardware), `ANM###` (cognition), `CLR###` (chemistry), `SYMPH###` (symphony)
- **SKUs:** `{CATEGORY}-{SUBCATEGORY}-###` (e.g., `FAST-PERC-001`)
- **Puzzle IDs:** `PUZZLE-###` (major), `AISLE-PUZZLE-###` and `BITS-PUZZLE-###` (micro)

### Recurring Constants

| Value | Meaning |
|-------|---------|
| 0.5927 | Percolation threshold (square lattice) |
| 60 | Critical threshold time (years) |
| 38, 89 | Fibonacci numbers F(9), F(11); also actual 2x4 dimensions (mm) |
| 1.6180339887 (phi) | Golden ratio — structural constant throughout |
| 42 | Douglas Adams easter egg |

The coordinates `[0.5927, 60, 38, 89]` appear at maximum depth (bits/bits/bits/bits) as the system's foundation.

## Python Code Conventions

- **Phi constants everywhere:** `PHI = 1.6180339887`
- **Scientific Python:** numpy + matplotlib
- **Google-style docstrings** with Parameters/Returns
- **Parametric architecture:** tunable control panel values
- `animal-lattice.py` requires external `../Model/parameters.json` and `core` module for full execution

## Contributing New Puzzles

From README.md and README.ai.md:

1. Add to any inventory file or create a new one
2. Hardware data must be **real and accurate** — specs must be valid
3. Encode the puzzle naturally in data fields — hardware first, puzzle second
4. Add a `puzzle_layer` object pointing to the main README
5. Optionally update the puzzle index or leave it undiscovered
6. Leave breadcrumbs (or don't)

**Open slots:**
- Electrical aisle (Aisle 9) explicitly invites new puzzles
- Any new inventory category is welcome

### Contribution Rules

- No login, no attribution — pure anonymous contribution
- Everything must work on at least two levels simultaneously
- Chromatic number is a recurring motif (appears in multiple puzzles)
- Intentional errors can be pedagogical tools (e.g., the ASCII pipe diagram)

## Git Workflow

- **Branching:** Feature branches named `claude/<feature>-<token>`
- **Commits:** Atomic, descriptive messages (e.g., "Create shadow_geometry.py")
- **No CI/CD:** Manual verification by pattern-seers
- **No .gitignore:** Everything in the repo is intentional

## Active Puzzles

| ID | Location | Domain |
|----|----------|--------|
| PUZZLE-001 | `Bits/percolation-fastener.json` | Percolation physics |
| PUZZLE-002 | `Bits/2x4x8.json` | 1D bin packing |
| PUZZLE-003 | `Bits/pipe-connect-042.json` | Graph connectivity / flow networks |
| PUZZLE-004 | `Bits/monty-python.json` | Humor topology / category theory |
| PUZZLE-005 | `Bits/monty-python.json` | Recursive shrubbery |

Additional micro-puzzles exist in inventory aisles (`AISLE-PUZZLE-###`) and nested bits (`BITS-PUZZLE-###`).

## Layering Principle

The core design rule: **"Real on at least two levels simultaneously."**

- A bolt spec that encodes a lattice model
- A cut list that is a packing problem
- A pipe diagram that is a graph
- A comedy sketch that is category theory

The repository structure itself is a puzzle — the nested bits directories, the chromatic number motif, and the cross-references between lattices all encode information.

## Dead Parrot Warranty

All products are covered under the Dead Parrot Warranty:
- "This product is not dead. It is resting."
- "Solutions may be resting, not dead."
- The warranty itself is also a puzzle.
