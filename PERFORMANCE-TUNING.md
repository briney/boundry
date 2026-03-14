# Optimize Pipeline: Performance Bottleneck Analysis

Deep dive into the `optimize` pipeline to identify suboptimal algorithmic choices and GPU fallback issues. The optimize pipeline is: initial relax → (alanine scan → beam expansion with design+minimize → score) × N cycles × M campaigns. Scoring uses the ddG ensemble pipeline (35 MD ensemble members, each repacked+minimized+scored).

---

## Summary

| # | Finding | Verification | Category | Per-Campaign Cost (Default ddG Optimize) | Fix |
|---|---------|--------------|----------|------------------------------------------|-----|
| 1 | OpenMM system rebuild per `get_energy_breakdown` | Confirmed | Algorithmic | ~18,340 to ~65,590 rebuilds | Cache/reuse prepared System |
| 2 | GPU Context recreation per ensemble member | Confirmed | GPU | ~8,908 to ~31,858 redundant Context creates | Reuse one Simulation Context |
| 3 | Constant tensors recreated in denoising loop | Confirmed | GPU | ~200K to ~710K avoidable H2D sync points | Lazy-cache constants per device |
| 4 | O(N²) Python CA-CA distance loop | Confirmed | Algorithmic | ~90K Python pair loops (two sites) per ensemble build | Vectorize helper |
| 5 | O(N²) renumber deduplication | Confirmed | Algorithmic | ~450K list comparisons per renumber on ~300 aa proteins | Use set+list |
| 6 | `map_mpnn_to_af2_seq` CPU→GPU per call | Confirmed | GPU | Repeated transfer + unnecessary one_hot@matmul | Device cache + index lookup |
| 7 | Uncached CUDA probe in idealize | Confirmed | GPU | 1 extra OpenMM probe per idealize() | Shared cached probe utility |
| 8 | Cost model undercounts ddG scoring calls in optimize | New | Algorithmic | Existing estimates off by 1-2 orders of magnitude | Use expansion-aware formulas |
| 9 | `DdGConfig.workers` is effectively unused in ddG API path | New | Algorithmic | Requested parallelism not applied | Instantiate/pass WorkPool in ddG APIs |
| 10 | OpenMM GPU detection is CUDA-only | New | GPU fallback | CPU fallback even when OpenCL GPU exists | Probe/select CUDA then OpenCL |
| 11 | Repeated temp-file round-trips for design/repack | New | Algorithmic | High I/O churn in beam/ddG inner loops | Add in-memory PDB entry points |

Findings 1, 2, 3, and 11 are the highest-impact opportunities. The previous campaign-level counts in this document were directionally correct but materially undercounted because they did not include ddG scoring performed inside beam expansion tasks.

### Cost-model correction (important)

For default optimize settings (`design_cycles=10`, `beam_expansion=25`, `beam_width=4`, `interface_scoring_backend="ddg"`):

- Baseline scoring calls (initial + per-cycle + final): **12**
- Expansion scoring calls: **250 to 925** (depends on parent beam growth)
- Total `compute_interface_dg` calls per campaign: **262 to 937**

This changes the expected magnitude of bottlenecks from "hundreds" to often "**tens of thousands**" of heavy OpenMM operations per campaign.

---

## Finding 1: Full OpenMM System Rebuild Per `get_energy_breakdown` Call

**Severity: High** | **File: `relaxer.py:867-944`**

Every `get_energy_breakdown` call does the complete pipeline from scratch:
1. `filter_protein_only()` — string scan
2. `_prepare_structure()` — PDBFixer parse + find/add missing atoms
3. `_build_system_for_ddg()` — ForceField load + Modeller + addHydrogens + createSystem
4. `openmm_app.Simulation()` — Context creation on GPU
5. Energy evaluation
6. Context destruction

In `_process_ensemble_member` (ddG scoring), **each of the 35 ensemble members** calls `get_energy_breakdown` **twice** (bound + unbound WT), totaling **70 full system constructions per scoring call**. Each involves PDB parsing, PDBFixer, hydrogen placement, force field parameterization, and GPU context creation.

The bound and unbound structures share the same topology (just different coordinates). A single `System` could be parameterized once and reused by just updating positions via `context.setPositions()`.

**Impact in optimize (corrected):** each `compute_interface_dg` call performs `n_ensemble * 2` energy evaluations (`35 * 2 = 70` by default). With 262-937 scoring calls per campaign, that is **~18,340 to ~65,590** full system rebuilds. This is the dominant bottleneck.

### Proposed fix

Add a lower-level method that accepts a pre-built `(system, topology)` and only creates the `Simulation` + evaluates energy, skipping the PDBFixer/ForceField/hydrogen pipeline. Then in `_process_ensemble_member`, build the system once from the first member's PDB and reuse it for all subsequent energy evaluations (bound and unbound share the same atom set — unbound just translates coordinates).

A cache keyed by topology hash on the `Relaxer` instance is another option, but the ddG worker process pattern (one `Relaxer` per worker, many calls to `get_energy_breakdown`) makes explicit system passing cleaner.

Sketch:
```python
def _get_energy_from_prepared(self, system, topology, positions):
    """Evaluate energy using a pre-built System — no PDBFixer/ForceField."""
    for i in range(system.getNumForces()):
        system.getForce(i).setForceGroup(i)

    platform = openmm.Platform.getPlatformByName(
        "CUDA" if self._check_gpu_available() else "CPU"
    )
    simulation = openmm_app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    # ... evaluate force groups as before ...
```

Then `_process_ensemble_member` calls `_prepare_structure` + `_build_system_for_ddg` once and passes the system/topology to all subsequent energy evaluations.

---

## Finding 2: OpenMM Context Recreated Per Ensemble Member

**Severity: Medium-High** | **File: `relaxer.py:823-855`**

In `generate_local_md_ensemble`, a new `openmm_app.Simulation` (and thus a new GPU `Context`) is created **for every one of the 35 ensemble members**:

```python
for i in range(n_members):           # n_members = 35
    integrator = openmm.LangevinMiddleIntegrator(...)
    sim = openmm_app.Simulation(      # <-- new Context, uploads System to GPU
        modeller.topology, system, integrator, platform
    )
    sim.context.setPositions(minimised_positions)
    sim.step(md_equilibration_steps)
    sim.step(md_total_steps)
    sim.minimizeEnergy(maxIterations=100)
    # extract positions...
    del sim                           # <-- destroys GPU Context
```

OpenMM `Context` creation involves compiling CUDA kernels and uploading the entire force field parameterization to GPU memory. The `System` object is shared, but the context is rebuilt 35 times.

**Impact in optimize (corrected):** each ensemble build currently creates an init `Simulation` plus one per member (`n_members + 1`, default 36). Reusing one member Context would save **34** creates per scoring call. Across 262-937 scoring calls, that is **~8,908 to ~31,858** redundant Context constructions per campaign.

### Proposed fix

OpenMM's `LangevinMiddleIntegrator` supports `setRandomNumberSeed(seed)` which takes effect on the next `step()` call. Create one `Simulation` before the loop and reuse it:

```python
integrator = openmm.LangevinMiddleIntegrator(
    md_temperature * unit.kelvin,
    md_friction / unit.picosecond,
    TIMESTEP_PS * unit.picoseconds,
)
sim = openmm_app.Simulation(modeller.topology, system, integrator, platform)

for i in range(n_members):
    member_seed = base_seed * 100_000 + i
    integrator.setRandomNumberSeed(member_seed)
    sim.context.setPositions(minimised_positions)
    sim.context.setVelocitiesToTemperature(
        md_temperature * unit.kelvin, member_seed
    )
    sim.step(md_equilibration_steps)
    sim.step(md_total_steps)
    sim.minimizeEnergy(maxIterations=100)

    state = sim.context.getState(getPositions=True)
    # ... extract PDB ...
```

This eliminates 34 Context create/destroy cycles per ensemble generation.

---

## Finding 3: Constant Tensors Recreated From NumPy Every Denoising Step

**Severity: Medium-High** | **File: `LigandMPNN/sc_utils.py:129-142`**

Inside the `pack_side_chains` denoising loop (`for step in range(num_denoising_steps):`), **5 constant tensors are recreated from numpy arrays and copied to GPU on every iteration**:

```python
# Lines 133-141, INSIDE the denoising loop:
torch.tensor(restype_rigid_group_default_frame, device=device)  # [21,8,4,4] — TWICE
torch.tensor(restype_atom14_to_rigid_group, device=device)      # [21,14]
torch.tensor(restype_atom14_mask, device=device)                 # [21,14]
torch.tensor(restype_atom14_rigid_group_positions, device=device)# [21,14,3]
```

Plus `make_torsion_features` (called once before the loop) creates additional constant tensors, and one more conversion occurs after the loop. Total per `pack_side_chains` call is approximately:

```text
5 * num_denoising_steps + 6
```

With current default `sc_num_denoising_steps=3`, that's **~21 synchronous CPU→GPU constant copies** per call.

Each `torch.tensor(numpy_array, device="cuda")` does: numpy → CPU tensor → H2D copy with implicit CUDA stream sync. The tensors are small (~17 KB total), but **the synchronization cost dominates** — each is a pipeline barrier that forces the GPU to flush.

**Impact in optimize (corrected):** the dominant source is ddG member repacking (`_process_ensemble_member -> designer.repack`) in every ensemble member. At 35 repacks per scoring call and 262-937 scoring calls, this is **~9,170 to ~32,795** `pack_side_chains()` invocations per campaign. At ~21 constant-tensor copies per call (current defaults), that's roughly **~200K to ~710K** avoidable synchronization points.

### Proposed fix

Add a device-keyed cache at module level:

```python
_CONST_CACHE: Dict[torch.device, Dict[str, torch.Tensor]] = {}

def _get_cached_constants(device: torch.device) -> Dict[str, torch.Tensor]:
    if device not in _CONST_CACHE:
        _CONST_CACHE[device] = {
            "restype_rigid_group_default_frame": torch.tensor(
                restype_rigid_group_default_frame, device=device
            ),
            "restype_atom14_to_rigid_group": torch.tensor(
                restype_atom14_to_rigid_group, device=device
            ),
            "restype_atom14_mask": torch.tensor(
                restype_atom14_mask, device=device
            ),
            "restype_atom14_rigid_group_positions": torch.tensor(
                restype_atom14_rigid_group_positions, device=device
            ),
        }
    return _CONST_CACHE[device]
```

Then replace all `torch.tensor(restype_..., device=device)` calls in `pack_side_chains` and `make_torsion_features` with lookups from the cache. Constants are created once per device per process lifetime.

---

## Finding 4: CA-CA Pair Restraint Loop is O(N²) Python

**Severity: Low-Medium** | **Files: `relaxer.py:550-557` and `relaxer.py:787-795`**

Two separate methods (`minimize_with_pair_restraints` and `generate_local_md_ensemble`) contain identical O(N²) Python loops for CA-CA distance computation:

```python
for idx_a in range(len(ca_indices)):
    for idx_b in range(idx_a + 1, len(ca_indices)):
        diff = positions_nm[i] - positions_nm[j]
        dist = float(np.sqrt(np.dot(diff, diff)))
        if dist <= cutoff_nm:
            restraint_force.addBond(i, j, [dist])
```

For a 300-residue protein (300 CA atoms): 300×299/2 = 44,850 iterations in pure Python, computing distances one at a time. The code is also duplicated between the two methods.

### Proposed fix

Extract a shared helper and vectorize with `scipy.spatial.distance`:

```python
from scipy.spatial.distance import pdist, squareform

def _add_ca_pair_restraints(system, ca_indices, positions_nm, cutoff_nm, k_value):
    ca_positions = positions_nm[ca_indices]
    dists = pdist(ca_positions)           # all pairwise distances, vectorized
    pairs = np.array(list(zip(*np.triu_indices(len(ca_indices), k=1))))
    mask = dists <= cutoff_nm

    restraint_force = openmm.CustomBondForce("0.5 * k * (r - r0)^2")
    restraint_force.addGlobalParameter("k", k_value)
    restraint_force.addPerBondParameter("r0")

    for (a, b), d in zip(pairs[mask], dists[mask]):
        restraint_force.addBond(int(ca_indices[a]), int(ca_indices[b]), [float(d)])

    system.addForce(restraint_force)
    return int(mask.sum())
```

The `pdist` call replaces ~45K Python iterations with a single C-level vectorized operation. The remaining Python loop only iterates over the matching pairs (typically a small fraction of all pairs).

---

## Finding 5: `renumber_pdb` Has O(N²) Deduplication

**Severity: Low** | **File: `renumber.py:95`**

```python
if key not in chain_residues[chain_id]:    # O(N) list scan
    chain_residues[chain_id].append(key)
```

Inside an O(N_atoms) loop, this does O(N_residues) membership testing on a list. For 300 residues × ~10 atoms each = 3,000 atoms, with an average list length of ~150 for the membership check, that's ~450,000 comparisons.

### Proposed fix

Track seen keys in a parallel `set` for O(1) membership testing while preserving the ordered list for downstream use:

```python
chain_residues_seen = defaultdict(set)
# ...
if key not in chain_residues_seen[chain_id]:
    chain_residues_seen[chain_id].add(key)
    chain_residues[chain_id].append(key)
```

---

## Finding 6: `map_mpnn_to_af2_seq` CPU Constant Moved to GPU Per Call

**Severity: Low** | **File: `LigandMPNN/sc_utils.py:34-58, 188`**

```python
# Module level — hardcoded to CPU:
map_mpnn_to_af2_seq = torch.tensor([...], device="cpu")  # [21,21] permutation matrix

# Called every make_torsion_features invocation:
S_af2 = torch.argmax(
    torch.nn.functional.one_hot(feature_dict["S"], 21).float()
    @ map_mpnn_to_af2_seq.to(device).float(),  # <-- CPU→GPU transfer each call
    -1,
)
```

Similarly, `torch_pi` (line 31) is a CPU scalar used in GPU arithmetic at line 222-224, causing implicit device transfer in `2 * torch_pi * torch.rand(..., device=device)`.

### Proposed fix

Include both in the `_get_cached_constants` cache from Finding 3 and replace the one-hot matmul with direct index remapping:

```python
# In the cache builder:
"map_mpnn_to_af2_seq": map_mpnn_to_af2_seq.to(device).float(),
"torch_pi": torch.tensor(np.pi, device=device),
```

Then precompute an integer lookup once:

```python
mpnn_to_af2_index = torch.argmax(map_mpnn_to_af2_seq, dim=1).to(device)
S_af2 = mpnn_to_af2_index[feature_dict["S"].long()]
```

This removes both per-call transfer and the `one_hot @ permutation_matrix` compute.

---

## Finding 7: `idealize.py` Re-implements CUDA Probe Without Caching

**Severity: Low** | **File: `idealize.py:485-508`**

`idealize.py` has its own inline copy of the GPU detection logic (creating a throwaway OpenMM Context to probe CUDA availability), but unlike `Relaxer._check_gpu_available()`, it has **no caching**. Every `idealize()` call re-probes CUDA by creating and destroying a test GPU context.

### Proposed fix

Extract the CUDA probe into a shared module-level utility (e.g. in `relaxer.py` or a `_gpu.py` helper) with a module-level cache:

```python
_openmm_gpu_available: Optional[bool] = None

def check_openmm_gpu() -> bool:
    global _openmm_gpu_available
    if _openmm_gpu_available is not None:
        return _openmm_gpu_available
    # ... existing probe logic ...
    _openmm_gpu_available = result
    return result
```

Then both `idealize.py` and `Relaxer._check_gpu_available()` delegate to this shared function.

---

## Finding 8: Cost Model Misses Expansion-Path ddG Scoring

**Severity: High (analysis correctness)** | **Files: `optimize.py:295-309`, `optimize.py:1115-1117`, `optimize.py:1239-1241`**

The initial model in this document counted only mainline `_score_interface` calls, but each beam expansion task also scores via `compute_interface_dg` when backend is ddG (default in `OptimizeConfig`).

That means campaign cost scales primarily with:

```text
N_scoring_calls
  = (design_cycles + 2)                # baseline
  + expansion_calls_per_cycle_sum      # from pool.map(_execute_beam_expansion)
```

So default optimize campaigns can easily do hundreds of ddG scoring calls, not ~11.

### Proposed fix

Use expansion-aware formulas throughout this document and future profiling:

```text
system_rebuilds ~= N_scoring_calls * (2 * n_ensemble)
redundant_context_creates ~= N_scoring_calls * (n_ensemble - 1)
```

---

## Finding 9: `DdGConfig.workers` Is Not Applied in ddG API Path

**Severity: Medium** | **Files: `cli.py:1216-1258`, `operations.py:1180`, `operations.py:1187`, `ddg.py:810-1191`**

`--workers` is exposed on CLI and stored in `DdGConfig.workers`, but the ddG operation path calls `compute_ddg()` / `compute_interface_dg()` without constructing/passing a pool. In current code, those functions only parallelize when `pool` is explicitly provided.

Result: users request parallel workers but receive sequential per-member scoring in ddG commands.

### Proposed fix

In `operations.ddg()`, create a `WorkPool(config.workers)` when `config.workers > 1` and pass it into `compute_ddg()` / `compute_interface_dg()`. Keep current explicit `pool`-injection path for optimize integration.

---

## Finding 10: GPU Detection Is CUDA-Only (Misses OpenCL GPUs)

**Severity: Medium** | **Files: `relaxer.py:44-72`, `idealize.py:485-508`**

OpenMM platform selection is currently binary (`CUDA` else `CPU`). Systems with non-CUDA GPU acceleration (for example OpenCL-capable GPUs) will fall back to CPU even when hardware acceleration is available.

### Proposed fix

Centralize platform probing with priority order:

1. `CUDA`
2. `OpenCL`
3. `CPU`

Expose selected platform in logs/metadata to make fallback explicit and auditable.

---

## Finding 11: Temp-File Round-Trips in Hot Design/Repack Loops

**Severity: Medium** | **Files: `ddg.py:777-797`, `binding_energy.py:47-62`, `optimize.py:277-289`, `operations.py:218-223`**

Hot loops repeatedly convert PDB strings to temporary files so `Designer` can parse from path (`parse_PDB`), then read/write back to strings. This happens in:

- ddG member scoring repack path
- beam expansion design path
- interface scanning repack paths

The repeated filesystem round-trips add avoidable latency and serialization overhead, especially at high expansion counts.

### Proposed fix

Add in-memory `Designer` entry points (for example `design_pdb_string` / `repack_pdb_string`) using stream parsing (`parsePDBStream`) or pre-parsed/cached atom objects where safe. Keep file-based methods for compatibility.
