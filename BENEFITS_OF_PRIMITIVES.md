# What the project should find

A compact, typed **basis of information primitives**—small, reusable operators/predicates that show up across modalities and models. Each primitive has:

* **Type & arity** (e.g., spatial unary, temporal binary),
* **Symmetry profile** (invariances/equivariances),
* **Detector() / Generator()** (run on data / synthesize examples),
* **Composition rules** (algebra + typing: what combines with what),
* **Evidence of universality** (helps compress & transfer across ≥2 domains and ≥3 model families).

Think of families like:

* **Spatial:** Edgeθ, Corner, Parallel, Inside(region)
* **Temporal:** Before(A,B), During(A,B), Onset(x), Periodic(x,f)
* **Causal/Agentive:** Cause(x,y), Enable(x,y), Affords(obj,act)
* **Logical/Structural:** And/Or/Not, PartOf/HasPart, Equals, Unify
* **Quantitative:** Distance, Ratio, Threshold(>, ≥), Normalize
* **Informational/NSM-like:** Same/Different, More/Less, Possible/Necessary
* **Topological:** Connected, Loop, Hole, Tree, Cycle
* **Cognitive/Communicative:** Refer, Say(agent,content), Believe(agent,prop)

The “periodic table” is the catalog + algebra + detectors, not a word list. Larger concepts (“cup”, “promise”, “traffic jam”) are **programs** (graphs) built from these primitives.

# Why this matters for advanced AI

1. **Compositional generalization.** Models can build new meanings by recombining a small set of tested atoms, instead of memorizing examples.
2. **Sample efficiency & transfer.** A primitive learned in images (“inside/edge/parallel”) helps with text diagrams, code ASTs, or robotics geometry.
3. **Interpretability by construction.** Each decision becomes a short primitive program you can read, test, and ablate.
4. **Robustness to distribution shift.** Symmetry-aware primitives (time-shift, rotation, scale) are stable under common perturbations.
5. **Continual learning without bloat.** You store **differences** (new primitive combos), not entire new networks; old atoms remain reusable.
6. **Safer planning & tool use.** Typed operators with explicit pre/post-conditions reduce illegal actions and hallucinated affordances.
7. **Efficient temporal reasoning.** Temporal primitives + a light reservoir/ESN let agents track events and causality without heavy retraining.

# How compression works from base primitives (MDL view)

We treat primitives as a **codebook of executable parts** and compress by explaining data as a short program of those parts.

**Define the score.**
Minimum Description Length (MDL):

* Total bits = **L(model)** + **L(data | model)**.
  Here, the “model” is your primitive catalog + algebra; “data|model” is the shortest program (composition of primitives + parameters) that reconstructs the sample.

**Encoding pipeline (conceptual):**

1. **Factorize** the input into a primitive graph
   `G = factor(input; detectors, algebra)`
2. **Parameterize** each primitive instance (e.g., angle θ, position x,y, lag Δt).
3. **Entropy-code** the sequence:

   * **Which primitives** (IDs) appear (use arithmetic coding with learned frequencies),
   * **How they compose** (edges in the program graph),
   * **Their parameters** (quantized & coded).
4. **Decode** by running the generator(s) via the algebra to reconstruct the input.
5. **MDL comparison**: if `L(G)` ≪ naive encodings (e.g., pixels/tokens), the primitives are earning their keep.

**Toy example (vision).**
64×64 grayscale image (4096 pixels). Raw 8-bit storage = 4096×8 = **32,768 bits**.
Suppose the image is a simple filled rectangle. With primitives:

* Primitive ID (“Rectangle”) → a few bits (e.g., 4–8)
* Position (x,y) each in 0..63 → **6 + 6 bits**
* Size (w,h) each in 0..64 → **6 + 6 bits**
* Fill intensity → **8 bits**
* A tiny overhead for “fill vs outline” → \~**2–4 bits**
  Even with rough overhead (say 16 bits), you’re in the ballpark of **\~52–56 bits** total—orders of magnitude smaller than 32,768 bits for this structure. Real images are more complex, so you’ll encode a **short program** of a few primitives (edges, corners, regions, textures), still beating raw or naive DCT when scenes are structured. (In practice you also add a small **L(model)** term amortized across a dataset.)

**Toy example (text).**
“The red ball is on the table.”

* `CategoryOf(ball, object)`, `HasProperty(ball, red)`, `On(ball, table)`
* Encode primitive IDs (CategoryOf, HasProperty, On), the **arguments** (entities), and a small lexicon map for surface words.
  The decoder reconstructs a canonical proposition set (and, if needed, a surface form). Across a corpus, these relations recur, so their IDs get very short codes.

**Key implementation notes (for the coder):**

* Start with **two baselines**: (a) gzip/brotli; (b) k-means/DCT (images) or n-gram (text).
* Use **arithmetic coding** (or range coding) on: primitive IDs, composition ops, parameters.
* Learn **parameter priors** (e.g., angles, ratios) to shrink codes over time.
* Keep a **delta mode**: only encode the **differences** between successive graphs in a sequence—huge gains for video, logs, dialogue.

# What “periodic” buys you in compression

* **Families** share priors (e.g., all edge angles share an angular prior; all temporal lags share a heavy-tail prior), which tightens codes.
* **Algebraic closure** means the search for programs is bounded and cached; common sub-programs become macros (further savings).
* **Symmetry** awareness lets you normalize away nuisance variation before coding (e.g., rotate to canonical orientation, then encode once).

# How this flows into model design

* Use primitives as the **middle layer API** between perception and planning: perception → primitive graph → planner operates over typed ops → action.
* Replace “giant opaque features” with **sparse activation of tested atoms**; train small adapters on top for each task.
* Make memory a **versioned store of primitive graphs** with delta-encoding; retrieval is symbolic-plus-neural (fast and cheap).
* For language models, bind a lightweight **semantic parser** to emit primitives alongside tokens; decode either to text or to actions.

---

**Bottom line:** The project should discover a **small, symmetry-aware, executable basis** of information. That basis lets you (1) compress and reconstruct multi-modal data, (2) reason and plan compositionally, (3) transfer knowledge efficiently, and (4) keep models interpretable and stable under shift. Compression isn’t a side effect—it’s the objective signal that your primitives are the right ones.
