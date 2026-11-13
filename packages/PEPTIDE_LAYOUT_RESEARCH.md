# General peptide depiction

Research and implementation notes, 2026-11-08.

## Reference implementations

| Source | Relevant behavior | Use in this project |
| --- | --- | --- |
| [RDKit depiction API](https://rdkit.org/docs/source/rdkit.Chem.rdDepictor.html) | Fixed coordinates, ring templates, and path-angle expansion | Local fragment coordinates, a complete-graph seed, and an independent baseline |
| [CoordGen](https://github.com/schrodinger/coordgenlibs) | Macrocycle construction and configurable ring-system templates | A separate baseline for complex cyclic graphs |
| [pyPept](https://github.com/Boehringer-Ingelheim/pyPept) | Monomer-based peptide assembly with BILN, HELM, and natural-sequence input | An independent notation and assembly comparison |

RDKit exposes coordinate constraints and ring templates as separate controls. Its constrained depiction API also separates regeneration from rigid alignment and includes handling for existing wedging. This supports keeping molecular identity, coordinate construction, and stereo rendering as distinct steps.

CoordGen publishes its macrocycle implementation and template mechanism. Its stated emphasis is drawing quality. A result from that implementation remains subject to this project's chemical and geometric validation.

pyPept accepts complex monomer notation and uses an internal monomer representation. Its FASTA interface is restricted to natural amino acids. This supports the existing choice of HELM/BILN with registered definitions for modified and crosslinked peptides.

## Model

A peptide is a graph of monomer instances and attachment endpoints. The graph can contain paths, branches, multiple chains, parallel connections, and cycles. A molecular ring basis cannot determine which bonds are peptide connections.

For sequence-derived structures, assembly supplies retained attachment atoms and atom ownership. Layout uses these exact boundaries. Terminal decorations are merged with their adjacent backbone fragment. Every retained inter-fragment bond becomes an equality constraint. Branch chains remain separate paths, and additional closures remain explicit edges.

For unannotated SMILES, backbone discovery operates on amide transitions. Strongly connected components restrict cycle search to candidate cyclic regions; acyclic paths are processed without recursion over residues. Mixed-length residues and urea links remain representable. A bounded search reports exhaustion explicitly. This structural decomposition does not establish a unique monomer sequence.

## Local geometry and motion

The fast solver keeps a rigid fragment's internal coordinates. If that model cannot satisfy a closure, atom refinement releases acyclic local geometry while retaining intrinsic ring shapes. A mixed coordinate chart represents intrinsic rings by translation and rotation and other atoms by Cartesian positions. Its analytic pullback supplies objective and constraint derivatives. Auxiliary-ring initialization also supports Cartesian refinement with independent ring-distance constraints; every accepted result passes complete pair-distance validation for those ring templates. Small backbone cycles use coordinates derived from the closed polygon, including the corresponding carbonyl geometry. Larger backbones use local residue geometry.

Internal rings are identified within each fragment. A bond that belongs to a crosslink-induced molecular cycle may still be a bridge in the fragment. Such a bond permits side-chain motion. Bonds inside an intrinsic ring remain part of the rigid fragment.

Pendant-subgraph rotations and reflections retain internal bond lengths. A motion that changes a connection endpoint triggers a joint projection of all fragment poses onto the bond-closure and interface-angle constraints. Candidate quality is evaluated after that projection. This distinction is necessary for disulfide and multiply closed structures.

## Initialization and optimization

A simple cyclic chain uses the existing cycle and port-direction initializers. A branched or crosslinked registered graph uses a complete molecular embedding as a reference. For acyclic registered connection graphs, retain the complete embedding within each fragment and normalize shortened bridge bonds. This avoids losing branch directions during rigid fitting. Crosslinked graphs use weighted rigid fitting to determine fragment translations, rotations, and reflection choices. The reference provides an initial state; the registered graph determines the bonds to close.

Connection restoration first uses damped minimum-norm Jacobian steps with a line search. A feasible result is accepted only if its defect rank improves; otherwise full constrained optimization remains available. Continuous optimization uses analytic derivatives for poses, bond equalities, angular bounds, and separation terms. Half-plane alternatives supply explicit separating choices for intersecting bonds. A separate shared-path initializer places the long path outside a short auxiliary ring instead of forcing both paths into the same face. Segment crossing costs use exact pairwise geometry during beam construction. Coupled residue reflections explore configurations that a sequence of individually improving reflections can miss.

Atom and segment spatial queries restrict numerical work to potentially active interactions. Candidate pruning includes soft contacts and angular dependencies. Motions that require a global closure projection remain eligible because their effect extends beyond their moved atoms.

## Validation and performance

Evaluate conversion, molecular identity, geometry, and final-image readability independently. Acceptance requires complete bond checks, closure residuals, angle bounds, contacts, crossings, ring intrusions, and stereo reconstruction. Registered atom ownership and connection endpoints are retained in the output report.

The fixed review matrix includes linear chains, terminal modifications, noncanonical residues, head-to-tail rings, disulfides, lactams, branches, and multiple closures. Every row reaches the solver; unsuccessful attempts remain in the report. Expansion tests include altered atom order, mixed backbone lengths, charged termini, and long backbone searches.

Measure initialization, optimization, projection, and drawing separately. Report acceptance alongside latency and include rejected inputs and timeouts. Source snapshots, chemical inputs, seeds, dependency versions, and image hashes accompany runs.

GPU work should target batches of independent solves with persistent device state. A distance-kernel port alone does not remove CPU optimizer synchronization. Compare any GPU solver with the same CPU derivatives, constraints, acceptance rules, and fixed inputs before selecting it. Model training is a separate two-GPU workload.

## Measured coverage

The fixed twelve-group matrix accepts 96/96 structures; a new-seed synthetic matrix accepts 24/24. The earlier eight long-ring diagnostics and the three reviewed patent references pass separately. Source snapshots and per-input failures from earlier versions remain in the experiment outputs. Passing these sets does not establish full chemical-space coverage or real-image recognition performance.

A head-to-side thioether reference exposed an important decomposition distinction: an acylated terminal nitrogen can participate in the macrocycle even when a conventional linear backbone is also identifiable. Such a ring uses the macrocycle path; an ordinary disulfide loop does not turn a free-terminal backbone into a head-to-tail chain.

## Remaining work

- Multiple registered ports on one atom and broader crosslink geometry coverage.
- Single-atom connector bodies and disconnected-component packing.
- Compact long-chain layouts evaluated at the actual model input size.
- Intrinsic glyph and stereo-mark clearance, patent-style calibration, and independent real-image evaluation.
- Controlled CPU/GPU and training comparisons with unchanged baseline checkpoints.
