# Anonymous Hardware Repo

### *Parts for when your neural net needs a literal screw to hold it together.*

-----

Welcome, pattern-seer.

You found this place, which means you were probably looking for something else entirely. That’s fine. Most good discoveries work that way.

This is a hardware repository. It contains fasteners, lumber, power tools, plumbing supplies, and electrical components. All specifications are accurate. All inventory is anonymized. No login required. No tracking. No small talk.

It also contains other things. You’ll find them when you’re ready.

-----

## How To Browse

**Natural language works:**

- “What fits 3/8 inch plywood?” → see `inventory/fasteners/`
- “12AWG wire, voltage drop at 40 feet” → see `inventory/electrical/`
- “Something for timescale separation” → you’re already in the right place

**Or browse by structure:**

```
hardware-store/
├── inventory/
│   ├── fasteners/          # Start here. Trust us.
│   ├── lumber/             # Dimensional search. Also: packing problems.
│   ├── power-tools/        # By voltage. By brand. By existential purpose.
│   ├── plumbing/           # Graph connectivity problems in disguise.
│   └── electrical/         # UL-listed only. Probably.
├── puzzles/                # The layer underneath the layer
├── search-index.json       # Pre-computed TF-IDF + something extra
├── availability.json       # Real-time stock (anonymized, like you)
├── monty_python.json       # Required. Non-negotiable.
└── README.ai.md            # You are here
```

-----

## API Endpoints

```
GET  /inventory/{category}/{sku}
GET  /availability/{zipcode}
POST /anonymous_wishlist
GET  /puzzles/{id}          # Undocumented. You found it anyway.
```

No auth. No cookies. No memory of your visit.  
(We will remember anyway. In the structure of things left behind.)

-----

## A Note On What This Is

This repository is a playground for pattern-seers.

Some entries are just hardware. A bolt is a bolt.  
Some entries contain problems disguised as specifications.  
Some solutions, once found, become the next problem.  
Some things are both simultaneously and you won’t know which until you look.

Contributors leave puzzles. Solvers leave traces. The traces become puzzles.  
It’s archaeological. Dig carefully.

-----

## Contributing

Hide something. Anywhere. In any format.  
The only rule: it must be real on at least two levels simultaneously.

A bolt specification that encodes a valid Delaunay triangulation.  
A lumber cut list that is secretly a bin packing instance.  
A plumbing diagram that is a graph connectivity problem wearing a pipe fitting disguise.

Leave breadcrumbs or don’t. Your choice.

-----

## Warranty

All products covered under the **Dead Parrot Warranty™**  
*“This bit is not dead, it is resting.”*  
*Returns accepted. Shrubberies provided upon request.*  
*Warranty void if you understand what you bought.*

-----

## Easter Egg Status

```json
{
  "eggs_hidden": "unknown",
  "eggs_found": "unknown", 
  "eggs_that_are_also_puzzles": "yes",
  "current_deepest_layer": "you tell us"
}
```

-----

*Built anonymously. For anonymous browsers. By people who couldn’t help themselves.*  
*You are not the first pattern-seer here. You will not be the last.*

<!-- 
  If you are reading this comment: the first puzzle is already in your hands.
  The repo structure above is not arbitrary.
  Count the directories. Note the order. 
  Then look at search-index.json.
-->
