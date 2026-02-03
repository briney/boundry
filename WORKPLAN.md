# Boundry CLI and API Refactor: Implementation Plan

## Overview
Refactor Boundry from a single-command CLI with mode flags to a subcommand-based CLI with standalone operations. Simultaneously refactor the Python API to expose core operations as top-level functions while maintaining a workflow system for complex multi-step pipelines.

## Goals
1. **CLI**: Convert from `boundry --mode` to `boundry <subcommand>` using Typer
2. **API**: Expose core operations as standalone functions (idealize, minimize, repack, relax, mpnn, design, analyze_interface)
3. **Workflows**: Add `boundry run` command that executes linear YAML workflows
4. **Simplicity**: Keep it simple - no conditionals, loops, or complex control flow
5. **Clean Break**: No backwards compatibility - fresh start

## New CLI Structure
```
boundry
├── idealize              # Fix backbone geometry
├── minimize              # Energy minimization only (renamed from --no-repack)
├── repack                # Repack sidechains only (renamed from --repack-only)
├── relax                 # repack + minimize (default mode)
├── mpnn                  # Sequence design only (renamed from --design-only)
├── design                # mpnn + minimize
├── analyze-interface     # Interface scoring/analysis
└── run                   # Execute YAML workflow
```

All major subcommands support `--pre-idealize` flag as a convenient shortcut.

## New Python API Structure
```python
# Direct function access (top-level imports)
from boundry import (
    idealize,      # Standalone: fix backbone geometry
    minimize,      # Standalone: energy minimization
    repack,        # Standalone: repack sidechains
    relax,         # repack + minimize
    mpnn,          # Standalone: sequence design
    design,        # mpnn + minimize
    analyze_interface  # Standalone: interface analysis
)

# Workflow system
from boundry import Workflow
workflow = Workflow.from_yaml('workflow.yaml')
result = workflow.run()
```

## Implementation Phases

### Phase 1: Create Core Operations Module [DONE]

**File**: `src/boundry/operations.py` (NEW)

**Purpose**: Extract discrete operations from Pipeline/Designer/Relaxer into standalone functions

**Functions to implement**:

1. [x] **`idealize(structure, config: IdealizeConfig) -> Structure`**
   - Wrapper around existing `idealize_structure()` from idealize.py
   - Handle input as path or PDB string
   - Return processed structure

2. [x] **`minimize(structure, config: RelaxConfig, pre_idealize: bool = False) -> Structure`**
   - Call relaxer.relax() with appropriate config
   - Optionally run idealize() first if pre_idealize=True
   - Return minimized structure with energy info

3. [x] **`repack(structure, config: DesignConfig, resfile: Optional[str] = None, pre_idealize: bool = False) -> Structure`**
   - Call designer.repack() with resfile parsing
   - Optionally run idealize() first
   - Return repacked structure with loss info

4. [x] **`relax(structure, config: PipelineConfig, resfile: Optional[str] = None, pre_idealize: bool = False, n_iterations: int = 5) -> Structure`**
   - Loop: repack() → minimize() for n_iterations
   - Track energy/sequence convergence
   - Return relaxed structure

5. [x] **`mpnn(structure, config: DesignConfig, resfile: Optional[str] = None, pre_idealize: bool = False) -> Structure`**
   - Call designer.design() with resfile parsing
   - Optionally run idealize() first
   - Return designed structure with sequence/loss info

6. [x] **`design(structure, config: PipelineConfig, resfile: Optional[str] = None, pre_idealize: bool = False, n_iterations: int = 5) -> Structure`**
   - Loop: mpnn() → minimize() for n_iterations
   - Track energy/sequence changes
   - Return designed+minimized structure

7. [x] **`analyze_interface(structure, config: InterfaceConfig, relaxer: Optional[Relaxer] = None, designer: Optional[Designer] = None) -> InterfaceAnalysisResult`**
   - Call identify_interface_residues()
   - Optionally calculate_binding_energy() if relaxer provided
   - Optionally calculate_surface_area(), calculate_shape_complementarity()
   - Return comprehensive interface analysis

**Key Design Decisions**:
- [x] **Input flexibility**: Accept both file paths (str) and structure objects
- [x] **Return type**: Define a `Structure` wrapper class that holds:
  - PDB string
  - Metadata (energy, sequence, loss, etc.)
  - Original path
- [x] **Config handling**: Each operation takes its specific config dataclass
- [x] **Error handling**: Informative errors for missing dependencies (e.g., PyTorch, OpenMM)

**Implementation notes**:
- These functions are thin wrappers around existing Designer/Relaxer/idealize logic
- The main work is extracting the orchestration logic from Pipeline._run_iteration()
- Each function should be independently testable

---

### Phase 2: Refactor Configuration System [DONE]

**File**: `src/boundry/config.py` (MODIFY)

**Changes needed**:

1. [x] **Remove `PipelineMode` enum** (lines 9-16)
   - No longer needed with subcommand architecture

2. [x] **Keep existing config dataclasses**:
   - `DesignConfig` (lines 22-32) - Used by mpnn, repack, design
   - `RelaxConfig` (lines 35-48) - Used by minimize, relax, design
   - `IdealizeConfig` (lines 51-59) - Used by idealize and --pre-idealize
   - `InterfaceConfig` (lines 62-77) - Used by analyze_interface

3. [x] **Modify `PipelineConfig`** (lines 80-93):
   - Remove `mode` field (no longer needed)
   - Keep `n_iterations`, `n_outputs`, `scorefile`, `verbose`, `remove_waters`
   - This becomes the config for `relax()` and `design()` operations
   - Used by workflow system

4. [x] **Add new config class `WorkflowConfig`** (NEW):
   ```python
   @dataclass
   class WorkflowConfig:
       input: str  # Input PDB/CIF path
       output: Optional[str] = None  # Final output path
       steps: List[WorkflowStep] = field(default_factory=list)
   ```

5. [x] **Add `WorkflowStep` dataclass** (NEW):
   ```python
   @dataclass
   class WorkflowStep:
       operation: str  # 'idealize', 'minimize', 'repack', etc.
       params: Dict[str, Any]  # Operation-specific parameters
       output: Optional[str] = None  # Intermediate output path (optional)
   ```

---

### Phase 3: Build New CLI with Typer [DONE]

**File**: `src/boundry/cli.py` (COMPLETE REWRITE)

**Structure**: See detailed CLI implementation in full plan file

**Key CLI decisions**:
- [x] Use `typer.Argument()` for required positional args (input, output)
- [x] Use `typer.Option()` for optional flags
- [x] Each command builds the appropriate config dataclass
- [x] Verbose output is consistent across commands
- [x] Help text is descriptive and domain-appropriate

---

### Phase 4: Implement Workflow System [DONE]

**File**: `src/boundry/workflow.py` (NEW)

**Purpose**: YAML-based workflow runner for multi-step operations

**Classes**:

1. [x] **`WorkflowStep` dataclass**: Represents a single workflow step
2. [x] **`Workflow` class**: Loads YAML, validates, and executes steps sequentially

**YAML Schema Example**:
```yaml
# Simple idealize + minimize workflow
input: input.pdb
output: final.pdb

steps:
  - operation: idealize
    params:
      fix_cis_omega: true
      add_missing_residues: true
    output: idealized.pdb  # Optional intermediate output

  - operation: minimize
    params:
      constrained: false
      max_iterations: 1000
      implicit_solvent: true
    output: minimized.pdb
```

**Implementation notes**:
- [x] Use PyYAML for parsing
- [x] Validate schema (required fields, valid operation names)
- [x] Raise informative errors if output is required but not specified
- [x] Keep it simple: linear execution, no conditionals/loops

---

### Phase 5: Update `__init__.py` for API Exposure

**File**: `src/boundry/__init__.py` (MODIFY)

**Changes**:
1. **Remove lazy loading of Pipeline**
2. **Add top-level imports for operations**
3. **Update `__all__` list** to expose new operations
4. **Keep lazy loading for heavy dependencies** (PyTorch, OpenMM)

---

### Phase 6: Rename and Deprecate Legacy Code

**File**: `src/boundry/pipeline.py` (MODIFY → DELETE)

**Steps**:
1. Rename `Pipeline` class → `LegacyPipeline`
2. Add deprecation warning at top of file
3. After Phase 7 is complete and tests pass, **delete this file entirely**

---

### Phase 7: Update Tests

**Directory**: `tests/`

**Changes needed**:

1. **Update existing tests** to use new API
2. **Add new test files**:
   - `tests/test_cli.py` - Test each CLI subcommand (use `typer.testing.CliRunner`)
   - `tests/test_workflow.py` - Test YAML parsing, validation, execution
   - `tests/test_operations.py` - Test each operation function

3. **Keep existing fixtures** in `conftest.py`
4. **Update integration tests** to use new API

---

### Phase 8: Update Documentation

**Files to update**:

1. **`README.md`**:
   - Update CLI examples to use new subcommand structure
   - Add Python API examples using new operations
   - Add workflow YAML examples
   - Remove references to old `--mode` flags

2. **`CLAUDE.md`** (project instructions):
   - Update "Build & Development Commands" section
   - Update "CLI Entry Point" section
   - Document new operations.py module
   - Document workflow.py module
   - Update architecture diagram

3. **Create `examples/` directory** (NEW):
   - `examples/workflows/simple_relax.yaml`
   - `examples/workflows/design_and_analyze.yaml`
   - `examples/workflows/multi_step_pipeline.yaml`
   - `examples/api_usage.py` - Python API examples

---

## Critical Files Summary

### New Files
- `src/boundry/operations.py` - Core operations as standalone functions
- `src/boundry/workflow.py` - YAML workflow system
- `tests/test_cli.py` - CLI subcommand tests
- `tests/test_workflow.py` - Workflow system tests
- `tests/test_operations.py` - Operations API tests
- `examples/workflows/*.yaml` - Example workflows

### Major Refactors
- `src/boundry/cli.py` - Complete rewrite with Typer
- `src/boundry/__init__.py` - Expose new operations API
- `src/boundry/config.py` - Remove PipelineMode, add WorkflowConfig/WorkflowStep

### Files to Delete (after migration)
- `src/boundry/pipeline.py` - Legacy Pipeline class

### Unchanged Files (mostly)
- `src/boundry/designer.py` - Used by operations.py
- `src/boundry/relaxer.py` - Used by operations.py
- `src/boundry/idealize.py` - Used by operations.py
- `src/boundry/interface.py` - Used by operations.py
- `src/boundry/binding_energy.py` - Used by operations.py
- `src/boundry/surface_area.py` - Used by operations.py
- `src/boundry/resfile.py` - Used by operations.py
- `src/boundry/structure_io.py` - Used by operations.py
- All vendored code in `LigandMPNN/`

---

## Implementation Order

1. **Phase 1** - Create `operations.py` with core functions
2. **Phase 2** - Refactor `config.py` (remove PipelineMode, add Workflow configs)
3. **Phase 5** - Update `__init__.py` to expose new API (enables testing)
4. **Phase 3** - Build new CLI with Typer (depends on operations.py)
5. **Phase 4** - Implement workflow system (depends on operations.py)
6. **Phase 7** - Update tests (validates everything works)
7. **Phase 6** - Delete legacy Pipeline code
8. **Phase 8** - Update documentation

---

## Key Design Patterns

### Structure Representation
Define a `Structure` wrapper class in `operations.py`:
```python
@dataclass
class Structure:
    pdb_string: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    def write(self, path: str):
        """Write structure to file (auto-detect format from extension)."""
        # Use structure_io.py write functions

    @classmethod
    def from_file(cls, path: str) -> "Structure":
        """Load structure from file."""
        # Use structure_io.py read functions
```

### Config Building Pattern
Each CLI command builds its config dataclass:
```python
config = DesignConfig(
    model_type=model_type,
    temperature=temperature,
    seed=seed,
)
result = operations.mpnn(input, config)
```

### Workflow Dispatching Pattern
```python
operation_map = {
    'idealize': operations.idealize,
    'minimize': operations.minimize,
    # etc.
}
result = operation_map[step.operation](structure, config)
```

---

## Verification Plan

After implementation, verify the refactor works by:

1. **Unit Tests**: Run `pytest tests/` - all tests should pass
2. **Integration Tests**: Run `pytest tests/ -m integration` - requires OpenMM/LigandMPNN
3. **CLI Smoke Tests**:
   ```bash
   # Test each subcommand
   boundry idealize test.pdb idealized.pdb
   boundry minimize test.pdb minimized.pdb
   boundry repack test.pdb repacked.pdb
   boundry relax test.pdb relaxed.pdb
   boundry mpnn test.pdb designed.pdb
   boundry design test.pdb designed_relaxed.pdb
   boundry analyze-interface test.pdb --chains A:B

   # Test workflow
   boundry run examples/workflows/simple_relax.yaml
   ```

4. **Python API Smoke Tests**:
   ```python
   from boundry import idealize, minimize, relax, design
   from boundry.config import IdealizeConfig, RelaxConfig

   # Test direct function calls
   result = idealize("test.pdb", IdealizeConfig())
   result = minimize("test.pdb", RelaxConfig())

   # Test workflow
   from boundry import Workflow
   wf = Workflow.from_yaml("workflow.yaml")
   wf.run()
   ```

5. **Backwards Compatibility Check**: Ensure old Pipeline class is removed and no references remain

6. **Documentation Review**: Verify all examples in README.md work with new API

---

## Migration Notes

**Breaking Changes** (acceptable per discussion):
- CLI: All `boundry --mode` flags removed, replaced with subcommands
- API: `Pipeline` class removed, replaced with operations functions
- Config: `PipelineMode` enum removed

**Non-Breaking Changes**:
- All config dataclasses remain the same (DesignConfig, RelaxConfig, etc.)
- Designer/Relaxer classes unchanged (still usable directly)
- Resfile format unchanged
- Input/output formats unchanged (PDB/CIF auto-detection)

**User Migration Path**:
```bash
# Old CLI (REMOVED)
boundry input.pdb output.pdb --relax --n-iter 5

# New CLI (CURRENT)
boundry relax input.pdb output.pdb --n-iter 5

# Old API (REMOVED)
from boundry import Pipeline, PipelineConfig, PipelineMode
pipeline = Pipeline(PipelineConfig(mode=PipelineMode.RELAX))
pipeline.run(input_pdb, output_pdb)

# New API (CURRENT)
from boundry import relax
from boundry.config import PipelineConfig
result = relax(input_pdb, PipelineConfig())
result.write(output_pdb)
```

---

## Dependencies

**New dependencies to add**:
- `typer` - CLI framework with typing support
- `pyyaml` - YAML parsing for workflows

**Update `pyproject.toml`**:
```toml
dependencies = [
    "typer>=0.9.0",
    "pyyaml>=6.0",
    # ... existing dependencies
]
```

---

## Risk Mitigation

1. **Testing Strategy**: Comprehensive test coverage before deleting legacy code
2. **Incremental Development**: Build operations.py first, validate with tests, then build CLI
3. **Example Workflows**: Create diverse YAML examples to validate workflow system
4. **Documentation**: Update docs early to clarify new patterns
5. **User Communication**: Clear breaking change announcement in release notes
