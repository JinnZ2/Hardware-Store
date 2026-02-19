# Aisle 12 — Plumbing

Pipe. Fittings. Valves. Connectors. Flow.

Everything that moves things from one place to another.

-----

The fittings are not on this shelf.

But you should know — the plumbing section is the one people walk past fastest. Nobody browses plumbing. You come here because something is leaking or something needs connecting and you want to fix it and leave. The romance of plumbing is not immediately obvious.

The romance of plumbing is that it’s all just a graph.

Nodes and edges. Pressure differentials. Flow conservation. You put water in one end and it finds its way through every possible path simultaneously, allocating itself by resistance. It knows things about the network that you don’t, just by flowing.

-----

What you’re looking for is in `Bits/`.

Specifically: `pipe_connect_042.json`

It looks like a standard fitting assembly. Seven nodes. Seven edges. Pressure ratings. Flow rates. Everything checks out technically.

There is a cycle. The cycle is not a mistake.

-----

*Something to notice before you go:*

```
    A
    |
    B ——— D
   / \     \
  C   \     G
 / \   \
E   F ——F
```

*This diagram contains an error.*  
*Finding the error is not the puzzle.*  
*Understanding why it’s there is.*

<!-- 
  AISLE-PUZZLE-003
  The max temperature rating of pipe_connect_042 is 60°C.
  The critical threshold of the percolation fastener is 60.
  The length of the percolation fastener is 60mm.
  
  Three 60s across three different product categories.
  What are the units of each?
  Are they the same 60?
  What would it mean if they were?
-->
