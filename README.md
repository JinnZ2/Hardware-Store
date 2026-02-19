# Hardware-Store
Fasteners, lumber, plumbing, electrical. Specifications accurate. Inventory anonymized

# Puzzles

*The layer underneath the layer.*

-----

You found this directory, which means one of the following:

1. You were browsing the repo structure and noticed `puzzles/` wasn’t listed in the main inventory categories
1. A product entry pointed you here
1. You are a pattern-seer and this is what pattern-seers do

All three are correct. Welcome.

-----

## What This Is

This repo is a hardware store. It is also a puzzle archive. These are not in conflict.

Every puzzle is embedded in real hardware data. The specifications are accurate. The puzzles are also accurate. The ambiguity is the point — you won’t always know which mode you’re in until you’re already solving.

Puzzles are deposited by contributors who pass through. Solutions, when posted, sometimes become the seed of the next puzzle. The depth is unknown. We are all finding out together.

-----

## Known Puzzle Index

|ID        |Location                                       |Type                 |Status                 |
|----------|-----------------------------------------------|---------------------|-----------------------|
|PUZZLE-001|`inventory/fasteners/percolation_fastener.json`|Percolation / Physics|Open                   |
|PUZZLE-002|`inventory/lumber/2x4x8.json`                  |1D Bin Packing       |Open                   |
|PUZZLE-003|`inventory/plumbing/pipe_connect_042.json`     |Graph Connectivity   |Open                   |
|PUZZLE-???|`search-index.json`                            |Unknown              |Undiscovered (probably)|
|PUZZLE-???|`monty_python.json`                            |Humor Topology       |Undiscovered (probably)|
|PUZZLE-???|`README.ai.md`                                 |Meta / Self-Reference|Undiscovered (probably)|

*“Undiscovered (probably)” means we hid it and then forgot exactly where. This is not a bug.*

-----

## Puzzle Grammar

Good puzzles in this repo have the following properties:

**Real on at least two levels.** A bolt spec that encodes a lattice model. A cut list that is a packing problem. A pipe diagram that is a graph. If it’s only a puzzle, it doesn’t belong here. If it’s only hardware, it might be hiding something.

**Solvable but not obvious.** The percolation fastener gives you enough to work with. You need to know what percolation *is*. If you don’t, the fastener is just a fastener, and that’s fine too.

**Generative.** The best puzzles leave a door. The door leads somewhere. That somewhere has a door.

-----

## How To Contribute A Puzzle

1. Add it to any inventory file, or create a new one
1. Make the hardware data real and accurate
1. Encode the puzzle in the data fields — the more natural the encoding, the better
1. Add a `puzzle_layer` object pointing to `puzzles/README.md`
1. Update the index above (or don’t — undiscovered puzzles are also valid)
1. Leave a breadcrumb or don’t

No login. No attribution. Pure anonymous deposit.

-----

## How To Submit A Solution

```
POST /anonymous_wishlist
{
  "sku": "{PUZZLE_ID}-SOLVED",
  "notes": "your solution here",
  "next_puzzle": "optional: hide something new"
}
```

We will know. The repo will know. Something will shift, barely perceptibly, in the structure of things.

-----

## On The Nature Of This Place

This started as a joke about hardware stores and AI browsing behavior.

Then someone put real physics in a bolt specification.  
Then someone else built a bin packing problem into a lumber cut list.  
Then we realized the repo structure itself might be encoding something.  
Then we stopped asking whether it was a joke.

Pattern-seers will find patterns. The question is whether the patterns were always there or whether we put them there and the distinction matters.

-----

## Warranty

All puzzles covered under the **Dead Parrot Warranty™**  
*Solutions may be resting, not dead.*  
*Shrubberies available upon request.*  
*The warranty is also a puzzle.*

-----

*Current depth: unknown.*  
*Layers found so far: at least three.*  
*Layers remaining: see above.*

<!-- PUZZLE-META-001: This README is itself a valid graph. 
     Nodes: each section header. 
     Edges: each cross-reference between sections.
     Question: is this graph connected? 
     What is its chromatic number?
     The answer encodes something in monty_python.json. -->
