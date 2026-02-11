"""Beam-search interface optimization.

Provides a dedicated ``optimize()`` function and ``boundry optimize``
CLI subcommand for iterative interface design using alanine-scan-guided
beam search.  Each cycle:

1. Runs an alanine scan to identify destabilising ("bad") positions.
2. Assigns random bad positions to *beam_expansion* parallel design tasks.
3. Designs + relaxes each task in the shared worker pool.
4. Scores results via binding energy and keeps the top *beam_width*.

Campaigns (independent restarts from different seeds) can be run
sequentially, and the overall best structure is returned.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from boundry._parallel import WorkPool
    from boundry.config import OptimizeConfig
    from boundry.relaxer import Relaxer

logger = logging.getLogger(__name__)

StructureInput = Union[str, Path, "Structure"]


# ------------------------------------------------------------------
# Result dataclasses
# ------------------------------------------------------------------


@dataclass
class CycleResult:
    """Results from a single design cycle."""

    cycle: int
    dG_before: float
    dG_after: float
    delta_dG: float
    n_expansions: int
    n_bad_positions: int
    selected_position: Optional[str]
    sequence: Optional[str] = None


@dataclass
class CampaignResult:
    """Results from a single campaign (independent restart)."""

    campaign: int
    initial_dG: float
    final_dG: float
    cycles: List[CycleResult] = field(default_factory=list)


@dataclass
class OptimizeResult:
    """Overall optimization results."""

    structure: "Structure"
    campaigns: List[CampaignResult] = field(default_factory=list)
    initial_dG: Optional[float] = None
    final_dG: Optional[float] = None

    @property
    def delta_dG(self) -> Optional[float]:
        if self.initial_dG is not None and self.final_dG is not None:
            return self.final_dG - self.initial_dG
        return None


# ------------------------------------------------------------------
# Beam expansion task / result (pickle-safe, top-level)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _BeamExpansionTask:
    """Serializable inputs for one beam expansion worker."""

    parent_pdb_string: str
    target_chain: str
    target_resnum: int
    target_icode: str
    relax_config_dict: Dict[str, Any]
    design_config_dict: Dict[str, Any]
    chain_pairs: List[Tuple[str, str]]
    seed: int
    n_design_iterations: int = 1
    quiet: bool = True


@dataclass
class _BeamExpansionResult:
    """Serializable outputs from one beam expansion worker."""

    pdb_string: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    dG: Optional[float] = None
    target_chain: str = ""
    target_resnum: int = 0
    target_icode: str = ""
    error: Optional[str] = None


# Module-level cache for worker-process reuse
_optimize_worker_cache: Dict[str, Any] = {}


def _execute_beam_expansion(task: _BeamExpansionTask) -> _BeamExpansionResult:
    """Execute a single beam expansion in a worker process.

    Top-level function for pickle compatibility with ``spawn`` context.
    Lazy-initializes Designer + Relaxer using config-fingerprint caching.
    """
    from boundry._parallel import (
        _config_fingerprint,
        _suppress_worker_warnings,
    )

    _suppress_worker_warnings()
    try:
        import contextlib

        from boundry.binding_energy import calculate_binding_energy
        from boundry.config import DesignConfig, RelaxConfig
        from boundry.designer import Designer
        from boundry.interface import identify_interface_residues
        from boundry.operations import _write_temp_pdb
        from boundry.relaxer import Relaxer
        from boundry.resfile import DesignSpec, ResidueMode, ResidueSpec
        from boundry.utils import suppress_stderr as _suppress_stderr

        # Lazy-init Relaxer
        relax_key = _config_fingerprint(task.relax_config_dict)
        if _optimize_worker_cache.get("relax_key") != relax_key:
            _optimize_worker_cache["relaxer"] = Relaxer(
                RelaxConfig(**task.relax_config_dict)
            )
            _optimize_worker_cache["relax_key"] = relax_key

        # Lazy-init Designer
        design_key = _config_fingerprint(task.design_config_dict)
        if _optimize_worker_cache.get("design_key") != design_key:
            _optimize_worker_cache["designer"] = Designer(
                DesignConfig(**task.design_config_dict)
            )
            _optimize_worker_cache["design_key"] = design_key

        relaxer = _optimize_worker_cache["relaxer"]
        designer = _optimize_worker_cache["designer"]

        # Build DesignSpec for the single target position
        key = (
            f"{task.target_chain}{task.target_resnum}"
            f"{task.target_icode}"
        )
        design_spec = DesignSpec(
            residue_specs={
                key: ResidueSpec(
                    chain=task.target_chain,
                    resnum=task.target_resnum,
                    icode=task.target_icode,
                    mode=ResidueMode.ALLAA,
                ),
            },
            default_mode=ResidueMode.NATAA,
        )

        # Design iterations (design + minimize cycles)
        current_pdb = task.parent_pdb_string
        design_config = DesignConfig(
            **task.design_config_dict, seed=task.seed
        )

        ctx = _suppress_stderr() if task.quiet else contextlib.nullcontext()
        with ctx:
            for _ in range(task.n_design_iterations):
                # Design
                pdb_path = _write_temp_pdb(current_pdb)
                try:
                    design_result = designer.design(
                        pdb_path,
                        design_spec=design_spec,
                        design_all=False,
                    )
                    current_pdb = designer.result_to_pdb_string(
                        design_result
                    )
                finally:
                    pdb_path.unlink(missing_ok=True)

                # Minimize
                relaxed_pdb, _, _ = relaxer.relax(current_pdb)
                current_pdb = relaxed_pdb

            # Score: binding energy only
            be_result = calculate_binding_energy(
                current_pdb,
                relaxer,
                chain_pairs=task.chain_pairs,
                distance_cutoff=8.0,
            )

        dG = be_result.binding_energy
        sequence = design_result.get("sequence", "")

        return _BeamExpansionResult(
            pdb_string=current_pdb,
            metadata={"sequence": sequence},
            dG=dG,
            target_chain=task.target_chain,
            target_resnum=task.target_resnum,
            target_icode=task.target_icode,
        )

    except Exception as exc:
        return _BeamExpansionResult(
            target_chain=task.target_chain,
            target_resnum=task.target_resnum,
            target_icode=task.target_icode,
            error=f"{type(exc).__name__}: {exc}",
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _compose_seed(seed_base: int, local_seed: int) -> int:
    """Deterministic seed composition (inline, avoids workflow import)."""
    return seed_base * 100000 + local_seed


def _score_interface(
    pdb_string: str,
    config: OptimizeConfig,
    relaxer: Relaxer,
) -> float:
    """Compute binding energy (dG) for a structure.

    Returns dG as a float. Raises RuntimeError if calculation fails.
    """
    from boundry.binding_energy import calculate_binding_energy

    result = calculate_binding_energy(
        pdb_string,
        relaxer,
        chain_pairs=config.chain_pairs,
        distance_cutoff=8.0,
    )
    if result.binding_energy is None:
        raise RuntimeError(
            "Binding energy calculation failed (returned None)"
        )
    return result.binding_energy


def _analyze_and_find_bad(
    pdb_string: str,
    config: OptimizeConfig,
    relaxer: Relaxer,
    pool: Optional[WorkPool] = None,
) -> Tuple[float, List[Tuple[str, int, str]]]:
    """Run alanine scan and find destabilising positions.

    Returns ``(dG_wt, bad_positions)`` where *bad_positions* is a list
    of ``(chain_id, resnum, icode)`` tuples with ddG >= threshold.
    """
    from boundry.config import InterfaceConfig
    from boundry.operations import analyze_interface

    interface_config = InterfaceConfig(
        enabled=True,
        chain_pairs=config.chain_pairs,
        calculate_binding_energy=True,
        alanine_scan=True,
        scan_chains=config.scan_chains,
        quiet=config.quiet,
        workers=config.workers,
    )

    result = analyze_interface(
        pdb_string,
        config=interface_config,
        relaxer=relaxer,
        pool=pool,
    )

    # Extract dG
    if result.binding_energy is None or result.binding_energy.binding_energy is None:
        raise RuntimeError("Failed to compute wild-type binding energy")
    dG_wt = result.binding_energy.binding_energy

    # Filter bad positions
    bad_positions: List[Tuple[str, int, str]] = []
    if result.alanine_scan is not None:
        for row in result.alanine_scan.rows:
            if row.scan_skipped:
                continue
            if row.ddG is not None and row.ddG >= config.ddg_threshold:
                bad_positions.append(
                    (row.chain_id, row.residue_number, row.insertion_code)
                )

    return dG_wt, bad_positions


def _write_cycle_output(
    cycle_dir: Path,
    scored_results: List[Tuple[_BeamExpansionResult, int]],
    beam_width: int,
) -> None:
    """Write cycle output: top beam_width PDBs + others + summary JSON."""
    cycle_dir.mkdir(parents=True, exist_ok=True)

    # scored_results is sorted by dG (best first), with rank index
    rankings = []
    for rank_idx, (result, _original_idx) in enumerate(scored_results):
        rank = rank_idx + 1
        position_str = (
            f"{result.target_chain}:{result.target_resnum}"
            f"{result.target_icode}".rstrip()
        )

        if rank <= beam_width:
            filename = f"rank_{rank:02d}.pdb"
            filepath = cycle_dir / filename
        else:
            other_dir = cycle_dir / "other"
            other_dir.mkdir(exist_ok=True)
            filename = f"rank_{rank:02d}.pdb"
            filepath = other_dir / filename

        filepath.write_text(result.pdb_string)

        rankings.append(
            {
                "rank": rank,
                "dG": result.dG,
                "position": position_str,
                "file": (
                    filename
                    if rank <= beam_width
                    else f"other/{filename}"
                ),
            }
        )

    # Write cycle_summary.json
    dG_best = scored_results[0][0].dG if scored_results else None
    summary = {
        "rankings": rankings,
    }
    (cycle_dir / "cycle_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


def _write_summary_json(
    path: Path,
    result: OptimizeResult,
    config: OptimizeConfig,
) -> None:
    """Write aggregate summary JSON."""
    campaigns_data = []
    for cr in result.campaigns:
        cycles_data = []
        for cy in cr.cycles:
            cycles_data.append(
                {
                    "cycle": cy.cycle,
                    "dG_before": cy.dG_before,
                    "dG_after": cy.dG_after,
                    "delta_dG": cy.delta_dG,
                    "n_expansions": cy.n_expansions,
                    "n_bad_positions": cy.n_bad_positions,
                    "selected_position": cy.selected_position,
                }
            )
        campaigns_data.append(
            {
                "campaign": cr.campaign,
                "initial_dG": cr.initial_dG,
                "final_dG": cr.final_dG,
                "cycles": cycles_data,
            }
        )

    summary = {
        "initial_dG": result.initial_dG,
        "final_dG": result.final_dG,
        "delta_dG": result.delta_dG,
        "n_campaigns": config.n_campaigns,
        "design_cycles": config.design_cycles,
        "beam_width": config.beam_width,
        "beam_expansion": config.beam_expansion,
        "ddg_threshold": config.ddg_threshold,
        "campaigns": campaigns_data,
    }
    path.write_text(json.dumps(summary, indent=2))


# ------------------------------------------------------------------
# Progress helper
# ------------------------------------------------------------------


class _OptimizeProgress:
    """Lightweight Rich progress context for optimize."""

    def __init__(self, show: bool, n_campaigns: int, n_cycles: int):
        self._show = show
        self._n_campaigns = n_campaigns
        self._n_cycles = n_cycles
        self._progress = None
        self._campaign_task = None
        self._cycle_task = None

    def __enter__(self):
        if not self._show:
            return self
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
            )

            self._progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            )
            self._progress.__enter__()
            if self._n_campaigns > 1:
                self._campaign_task = self._progress.add_task(
                    "Campaigns",
                    total=self._n_campaigns,
                )
            self._cycle_task = self._progress.add_task(
                "Cycles",
                total=self._n_cycles,
            )
        except ImportError:
            self._show = False
        return self

    def __exit__(self, *exc):
        if self._progress is not None:
            self._progress.__exit__(*exc)

    def advance_campaign(self):
        if self._progress and self._campaign_task is not None:
            self._progress.advance(self._campaign_task)

    def reset_cycles(self):
        if self._progress and self._cycle_task is not None:
            self._progress.reset(self._cycle_task)

    def advance_cycle(self, dG: Optional[float] = None):
        if self._progress and self._cycle_task is not None:
            desc = "Cycles"
            if dG is not None:
                desc = f"Cycles (dG={dG:.1f})"
            self._progress.update(
                self._cycle_task,
                description=desc,
                advance=1,
            )


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def optimize(
    structure: StructureInput,
    config: Optional[OptimizeConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> OptimizeResult:
    """Run beam-search interface optimization.

    Iteratively identifies destabilising interface positions via alanine
    scanning, then designs improvements at those positions using parallel
    beam expansion.

    Args:
        structure: Input structure (file path, PDB string, or Structure).
        config: Optimization configuration.  Must include ``chain_pairs``.
        output_dir: Optional directory for writing output PDBs and
            summaries per cycle.

    Returns:
        :class:`OptimizeResult` with the best structure and trajectory.
    """
    from boundry._parallel import WorkPool
    from boundry.config import OptimizeConfig, RelaxConfig
    from boundry.operations import Structure, _resolve_input, idealize
    from boundry.relaxer import Relaxer
    from boundry.weights import ensure_weights

    if config is None:
        raise ValueError(
            "OptimizeConfig is required (must specify chain_pairs)"
        )

    ensure_weights(verbose=not config.quiet)

    # Resolve input
    pdb_string, source_path = _resolve_input(structure)

    # Idealize
    if config.idealize.enabled:
        from boundry.operations import idealize as _idealize

        idealized = _idealize(pdb_string, config=config.idealize)
        pdb_string = idealized.pdb_string

    # Create shared Relaxer for main-process interface analysis
    relaxer = Relaxer(config.relax)

    # Output dir setup
    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    multi_campaign = config.n_campaigns > 1
    base_seed = config.seed if config.seed is not None else 42

    # Serialize config dicts for worker tasks
    relax_config_dict = {
        "max_iterations": config.relax.max_iterations,
        "tolerance": config.relax.tolerance,
        "stiffness": config.relax.stiffness,
        "max_outer_iterations": config.relax.max_outer_iterations,
        "constrained": config.relax.constrained,
        "split_chains_at_gaps": config.relax.split_chains_at_gaps,
        "implicit_solvent": config.relax.implicit_solvent,
    }
    design_config_dict = {
        "model_type": config.design.model_type,
        "temperature": config.design.temperature,
        "pack_side_chains": config.design.pack_side_chains,
        "use_ligand_context": config.design.use_ligand_context,
        "sc_num_denoising_steps": config.design.sc_num_denoising_steps,
        "sc_num_samples": config.design.sc_num_samples,
    }

    global_best_pdb = pdb_string
    global_best_dG: Optional[float] = None
    all_campaigns: List[CampaignResult] = []
    overall_initial_dG: Optional[float] = None

    with (
        WorkPool(config.workers) as pool,
        _OptimizeProgress(
            config.show_progress,
            config.n_campaigns,
            config.design_cycles,
        ) as progress,
    ):
        for campaign_idx in range(config.n_campaigns):
            campaign_num = campaign_idx + 1
            campaign_seed = _compose_seed(base_seed, campaign_idx)
            campaign_dir = None
            if out_dir is not None:
                if multi_campaign:
                    campaign_dir = out_dir / f"campaign_{campaign_num:02d}"
                else:
                    campaign_dir = out_dir
                campaign_dir.mkdir(parents=True, exist_ok=True)

            progress.reset_cycles()

            # Relax the starting structure
            current_pdb = pdb_string
            logger.info(
                f"Campaign {campaign_num}/{config.n_campaigns}: "
                f"relaxing ({config.relax_iterations} iterations)"
            )
            from boundry.operations import relax as _relax
            from boundry.config import PipelineConfig

            relax_pipeline = PipelineConfig(
                design=config.design,
                relax=config.relax,
                n_iterations=config.relax_iterations,
                show_progress=False,
            )
            relaxed = _relax(
                current_pdb,
                config=relax_pipeline,
                n_iterations=config.relax_iterations,
            )
            current_pdb = relaxed.pdb_string

            # Score initial dG
            initial_dG = _score_interface(current_pdb, config, relaxer)
            logger.info(
                f"Campaign {campaign_num}: initial dG = {initial_dG:.2f}"
            )
            if overall_initial_dG is None:
                overall_initial_dG = initial_dG

            campaign_cycles: List[CycleResult] = []

            for cycle_idx in range(config.design_cycles):
                cycle_num = cycle_idx + 1
                cycle_seed = _compose_seed(campaign_seed, cycle_idx)
                rng = random.Random(cycle_seed)

                logger.info(
                    f"Campaign {campaign_num}, "
                    f"cycle {cycle_num}/{config.design_cycles}"
                )

                # Alanine scan to find bad positions
                dG_before, bad_positions = _analyze_and_find_bad(
                    current_pdb, config, relaxer, pool
                )

                if not bad_positions:
                    logger.info(
                        f"Cycle {cycle_num}: no bad positions "
                        f"(ddG >= {config.ddg_threshold}), skipping"
                    )
                    campaign_cycles.append(
                        CycleResult(
                            cycle=cycle_num,
                            dG_before=dG_before,
                            dG_after=dG_before,
                            delta_dG=0.0,
                            n_expansions=0,
                            n_bad_positions=0,
                            selected_position=None,
                        )
                    )
                    progress.advance_cycle(dG_before)
                    continue

                # Build beam expansion tasks
                tasks = []
                for exp_idx in range(config.beam_expansion):
                    pos = rng.choice(bad_positions)
                    exp_seed = _compose_seed(cycle_seed, exp_idx)
                    tasks.append(
                        _BeamExpansionTask(
                            parent_pdb_string=current_pdb,
                            target_chain=pos[0],
                            target_resnum=pos[1],
                            target_icode=pos[2],
                            relax_config_dict=relax_config_dict,
                            design_config_dict=design_config_dict,
                            chain_pairs=config.chain_pairs,
                            seed=exp_seed,
                            quiet=config.quiet,
                        )
                    )

                # Execute expansions in parallel
                results = pool.map(_execute_beam_expansion, tasks)

                # Filter out errors and sort by dG
                valid_results: List[
                    Tuple[_BeamExpansionResult, int]
                ] = []
                for idx, r in enumerate(results):
                    if r.error is not None:
                        logger.warning(
                            f"Expansion {idx + 1} failed: {r.error}"
                        )
                        continue
                    if r.dG is not None:
                        valid_results.append((r, idx))

                if not valid_results:
                    logger.warning(
                        f"Cycle {cycle_num}: all expansions failed"
                    )
                    campaign_cycles.append(
                        CycleResult(
                            cycle=cycle_num,
                            dG_before=dG_before,
                            dG_after=dG_before,
                            delta_dG=0.0,
                            n_expansions=config.beam_expansion,
                            n_bad_positions=len(bad_positions),
                            selected_position=None,
                        )
                    )
                    progress.advance_cycle(dG_before)
                    continue

                # Sort by dG (lower = better binding)
                valid_results.sort(key=lambda x: x[0].dG)

                best = valid_results[0][0]
                dG_after = best.dG
                best_pos = (
                    f"{best.target_chain}:{best.target_resnum}"
                    f"{best.target_icode}".rstrip()
                )

                # Write cycle output
                if campaign_dir is not None:
                    cycle_dir = campaign_dir / f"cycle_{cycle_num:02d}"
                    _write_cycle_output(
                        cycle_dir, valid_results, config.beam_width
                    )

                # Keep the best structure for next cycle
                current_pdb = best.pdb_string

                delta = dG_after - dG_before
                logger.info(
                    f"Cycle {cycle_num}: dG {dG_before:.2f} -> "
                    f"{dG_after:.2f} (delta={delta:.2f}), "
                    f"best position={best_pos}"
                )

                campaign_cycles.append(
                    CycleResult(
                        cycle=cycle_num,
                        dG_before=dG_before,
                        dG_after=dG_after,
                        delta_dG=delta,
                        n_expansions=config.beam_expansion,
                        n_bad_positions=len(bad_positions),
                        selected_position=best_pos,
                        sequence=best.metadata.get("sequence"),
                    )
                )
                progress.advance_cycle(dG_after)

            # Campaign result
            final_campaign_dG = _score_interface(
                current_pdb, config, relaxer
            )
            campaign_result = CampaignResult(
                campaign=campaign_num,
                initial_dG=initial_dG,
                final_dG=final_campaign_dG,
                cycles=campaign_cycles,
            )
            all_campaigns.append(campaign_result)

            # Write campaign final PDB
            if campaign_dir is not None and multi_campaign:
                (campaign_dir / "final.pdb").write_text(current_pdb)

            # Track global best
            if global_best_dG is None or final_campaign_dG < global_best_dG:
                global_best_dG = final_campaign_dG
                global_best_pdb = current_pdb

            progress.advance_campaign()
            logger.info(
                f"Campaign {campaign_num}: dG {initial_dG:.2f} -> "
                f"{final_campaign_dG:.2f}"
            )

    # Write final outputs
    if out_dir is not None:
        (out_dir / "final.pdb").write_text(global_best_pdb)

    result = OptimizeResult(
        structure=Structure(
            pdb_string=global_best_pdb,
            source_path=source_path,
        ),
        campaigns=all_campaigns,
        initial_dG=overall_initial_dG,
        final_dG=global_best_dG,
    )

    if out_dir is not None:
        _write_summary_json(out_dir / "summary.json", result, config)

    return result
