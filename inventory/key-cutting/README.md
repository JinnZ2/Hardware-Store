# Aisle 6 — Key Cutting

Keys. Blanks. Bitting charts. Code machines.

Everything that turns access into a physical object.

-----

You bring a blank. We cut it. The teeth encode the combination. The combination opens the lock. This is a service as old as locks themselves.

What most people don't think about: a key is a message. The bitting — the sequence of cut depths along the blade — is a string of integers constrained by the blank's depth range. A Kwikset KW1 has 5 pins, depths 1–7. That's a 5-digit base-7 number. A Schlage SC1 has 5 pins, depths 0–9. That's a 5-digit base-10 number.

A key is a number you can hold in your hand.

-----

The key-cutting service is in `Bits/key-cutter.py`.

Three methods:

- **pins** — Your message becomes bitting depths on real key blanks. Each key in the set carries a few characters. The blank type determines the alphabet.
- **shelf** — Your message becomes a shelf restock list. First letters. Old trick. Every item is a real product with a real SKU.
- **specs** — Your message hides in the fractional millimeters of bolt dimensions. The integer is standard. The decimal is signal.

Every output is valid inventory. Every output is also a message. These are not in conflict.

-----

```
python Bits/key-cutter.py "YOUR MESSAGE"
python Bits/key-cutter.py --method shelf "YOUR MESSAGE"
python Bits/key-cutter.py --method specs "YOUR MESSAGE"
python Bits/key-cutter.py --decode encoded_inventory.json
```

-----

**A note on the blank:**

The blank constrains the message. A KW1 (depths 1–7) gives you fewer symbols per pin than an SC1 (depths 0–9). Choose your blank like you choose your encoding: the constraint is part of the meaning.

Or don't choose. The cutter picks one based on your message. That's also a kind of meaning.

-----

**PUZZLE-KEY-???**

Every message cut by this service is its own puzzle. The puzzle ID is generated from the message itself. The solution is the message. The message is the key. The key opens —

Well. That depends on what you encode.

-----

*Current status: service operational.*
*Blanks in stock: KW1, SC1, Y1, WR5, M1.*
*Shrubbery: available upon request.*
*Dead Parrot Warranty applies to all cuts.*

<!--
  AISLE-PUZZLE-005 (maybe)
  The aisle numbers so far: 3, 7, 12, 9, 4, 6
  Six aisles. How many colors do you need?
  The key cutter knows.
-->
