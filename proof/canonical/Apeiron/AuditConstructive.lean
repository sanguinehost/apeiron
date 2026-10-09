import Apeiron.AuditClosed
import Apeiron.Combinator.BoundedRoutes
import Apeiron.Combinator.RelationalApplication
import Apeiron.Combinator.Linear.Core.Tracks
import Apeiron.Combinator.Linear.Core.Contraction
import Apeiron.Combinator.Linear.Composition
import Apeiron.Combinator.Linear.Compositor
import Apeiron.Combinator.Linear.Exchange
import Apeiron.Combinator.Linear.Substitution

/-!
# Constructive route audit

This gate rejects `Classical.choice` and every axiom outside the quotient/propositional foundation
used by the existing route definitions. It is narrower than the corpus-wide `#audit` command:
the declarations below are the constructive replacement surface for the old unbounded route
totalizers.
-/

open Lean Elab Command in
/-- The foundational cone accepted for constructive route declarations. -/
def Apeiron.constructiveRouteAxioms : Lean.NameSet :=
  ({} : Lean.NameSet) |>.insert `propext |>.insert `Quot.sound

open Lean Elab Command in
/-- Reject a route declaration when its transitive cone contains choice or another extra axiom. -/
elab "#audit_constructive " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !Apeiron.constructiveRouteAxioms.contains a)
    if bad.isEmpty then
      logInfo m!"'{c}' depends on constructive axioms: {axioms.toList}"
    else
      throwError m!"'{c}' depends on axioms outside the constructive route cone " ++
        m!"(propext, Quot.sound): {bad.toList}"

#audit_constructive Apeiron.Combinator.ApplicationWiring.plug
#audit_constructive Apeiron.Combinator.ApplicationWiring.bang
#audit_constructive Apeiron.Combinator.ApplicationWiring.iRoute
#audit_constructive Apeiron.Combinator.ApplicationWiring.kRoute
#audit_constructive Apeiron.Combinator.RelPlug
#audit_constructive Apeiron.Combinator.RelApp
#audit_constructive Apeiron.Combinator.relPlug_routeGraph
#audit_constructive Apeiron.Combinator.relApp_routeGraph
#audit_constructive Apeiron.Combinator.RApp
#audit_constructive Apeiron.Combinator.CApp
#audit_constructive Apeiron.Combinator.rApp_unique
#audit_constructive Apeiron.Combinator.cApp_unique
#audit_constructive Apeiron.Combinator.rApp_iff_traceWithin
#audit_constructive Apeiron.Combinator.cApp_iff_traceWithin
#audit_constructive Apeiron.Combinator.rApp_iRoute
#audit_constructive Apeiron.Combinator.rApp_kRoute_first
#audit_constructive Apeiron.Combinator.rApp_kRoute_second
#audit_constructive Apeiron.Combinator.rApp_kRoute_twoStage
#audit_constructive Apeiron.Combinator.BoundedRoutes.TraceBudget
#audit_constructive Apeiron.Combinator.BoundedRoutes.traceRouteOfBudget
#audit_constructive Apeiron.Combinator.BoundedRoutes.traces_iff_traceRouteOfBudget
#audit_constructive Apeiron.Combinator.BoundedRoutes.traceRouteOfBudget_sound
#audit_constructive Apeiron.Combinator.BoundedRoutes.RunsBudget
#audit_constructive Apeiron.Combinator.BoundedRoutes.runsRouteOfBudget
#audit_constructive Apeiron.Combinator.BoundedRoutes.runs_iff_runsRouteOfBudget
#audit_constructive Apeiron.Combinator.BoundedRoutes.runsRouteOfBudget_sound
#audit_constructive Apeiron.Combinator.RelationalPaths.oddPos
#audit_constructive Apeiron.Combinator.RelationalPaths.traces_iff_relRuns
#audit_constructive Apeiron.Combinator.RelationalPaths.nonUnit_loop_entry_exit
#audit_constructive Apeiron.Combinator.RelationalPaths.relRunsN_embed_iff
#audit_constructive Apeiron.Combinator.RelationalPaths.relRuns_embed_iff
#audit_constructive Apeiron.Combinator.RelationalPaths.relRuns_embed_exists_iff
#audit_constructive Apeiron.Combinator.RelationalPaths.relConj
#audit_constructive Apeiron.Combinator.RelationalPaths.relRuns_conj_iff
#audit_constructive Apeiron.Combinator.RelationalPaths.relRuns_embed_succ_nonUnit
#audit_constructive Apeiron.Combinator.extPos
#audit_constructive Apeiron.Combinator.outPos
#audit_constructive Apeiron.Combinator.innPos
#audit_constructive Apeiron.Combinator.outerSet
#audit_constructive Apeiron.Combinator.innerSet
#audit_constructive Apeiron.Combinator.conjRoute
#audit_constructive Apeiron.Combinator.relTraces_conj
#audit_constructive Apeiron.Combinator.relRuns_flatten
#audit_constructive Apeiron.Combinator.relTraces_flatten
#audit_constructive Apeiron.Combinator.relTraces_flatten_nonUnit
#audit_constructive Apeiron.Combinator.relRuns_vanishing_three
#audit_constructive Apeiron.Combinator.rcomp_eq_some
#audit_constructive Apeiron.Combinator.runs_end_not_mem
#audit_constructive Apeiron.Combinator.runsN_rcomp_id_fwd
#audit_constructive Apeiron.Combinator.runsN_rcomp_id_bwd
#audit_constructive Apeiron.Combinator.runs_rcomp_id_on
#audit_constructive Apeiron.Combinator.rtensor_ext
#audit_constructive Apeiron.Combinator.rtensor_arg
#audit_constructive Apeiron.Combinator.rtensor_ext_id
#audit_constructive Apeiron.Combinator.dupArg
#audit_constructive Apeiron.Combinator.dupArg_ext
#audit_constructive Apeiron.Combinator.dupArg_arg
#audit_constructive Apeiron.Combinator.dupArg_inn
#audit_constructive Apeiron.Combinator.outerArg
#audit_constructive Apeiron.Combinator.outerArg_ext
#audit_constructive Apeiron.Combinator.outerArg_arg
#audit_constructive Apeiron.Combinator.dupArg_eq
#audit_constructive Apeiron.Combinator.outerArg_id_on_oddPos
#audit_constructive Apeiron.Combinator.outerArg_not_enter
#audit_constructive Apeiron.Combinator.stored_eq
#audit_constructive Apeiron.Combinator.relCollapse_step
#audit_constructive Apeiron.Combinator.appliedStep
#audit_constructive Apeiron.Combinator.traces_appliedStep
#audit_constructive Apeiron.Combinator.relRuns_eq_self
#audit_constructive Apeiron.Combinator.relRuns_rcomp_id
#audit_constructive Apeiron.Combinator.relRuns_route_embed
#audit_constructive Apeiron.Combinator.relRuns_oddPos_iff_traces
#audit_constructive Apeiron.Combinator.rcomp_embed
#audit_constructive Apeiron.Combinator.swapRoute
#audit_constructive Apeiron.Combinator.swapRoute_id_zSet
#audit_constructive Apeiron.Combinator.swapRoute_not_enter_zSet
#audit_constructive Apeiron.Combinator.pairRoute
#audit_constructive Apeiron.Combinator.pairRoute_ee
#audit_constructive Apeiron.Combinator.pairRoute_ea
#audit_constructive Apeiron.Combinator.pairRoute_ae
#audit_constructive Apeiron.Combinator.pairRoute_aa
#audit_constructive Apeiron.Combinator.pairRoute_x
#audit_constructive Apeiron.Combinator.pairRoute_y
#audit_constructive Apeiron.Combinator.bStore
#audit_constructive Apeiron.Combinator.bArg
#audit_constructive Apeiron.Combinator.bArg_ext
#audit_constructive Apeiron.Combinator.bArg_arg
#audit_constructive Apeiron.Combinator.bArg_xPos
#audit_constructive Apeiron.Combinator.bArg_yPos
#audit_constructive Apeiron.Combinator.bArg_id_oddPos
#audit_constructive Apeiron.Combinator.bArg_not_enter_oddPos
#audit_constructive Apeiron.Combinator.bPlug
#audit_constructive Apeiron.Combinator.bBig
#audit_constructive Apeiron.Combinator.swap_arg_comm
#audit_constructive Apeiron.Combinator.bBig_eq
#audit_constructive Apeiron.Combinator.bPlug_x
#audit_constructive Apeiron.Combinator.bPlug_y
#audit_constructive Apeiron.Combinator.BCompose
#audit_constructive Apeiron.Combinator.bCompose_iff_traceWithin
#audit_constructive Apeiron.Combinator.bCombOfBudget
#audit_constructive Apeiron.Combinator.bCompose_iff_bCombOfBudget
#audit_constructive Apeiron.Combinator.FedRel
#audit_constructive Apeiron.Combinator.fedRel_iff
#audit_constructive Apeiron.Combinator.relBPlug_collapse_x
#audit_constructive Apeiron.Combinator.relBPlug_collapse_y
#audit_constructive Apeiron.Combinator.relFed_x
#audit_constructive Apeiron.Combinator.relFed_inn
#audit_constructive Apeiron.Combinator.relApplied_step
#audit_constructive Apeiron.Combinator.bracket_sets
#audit_constructive Apeiron.Combinator.relApplied_reading
#audit_constructive Apeiron.Combinator.relOuter_reading
#audit_constructive Apeiron.Combinator.relLeft_reading
#audit_constructive Apeiron.Combinator.relRight_reading
#audit_constructive Apeiron.Combinator.relBracket
#audit_constructive Apeiron.Combinator.bArg_rtensor_comm
#audit_constructive Apeiron.Combinator.cStore
#audit_constructive Apeiron.Combinator.cMach
#audit_constructive Apeiron.Combinator.cMach_eq
#audit_constructive Apeiron.Combinator.relCRight_reading
#audit_constructive Apeiron.Combinator.conjRoute_rcomp
#audit_constructive Apeiron.Combinator.relRuns_conj
#audit_constructive Apeiron.Combinator.cSwap_image
#audit_constructive Apeiron.Combinator.bArg_conj
#audit_constructive Apeiron.Combinator.CExchange
#audit_constructive Apeiron.Combinator.relCLeft_reading
#audit_constructive Apeiron.Combinator.relCommutator
#audit_constructive Apeiron.Combinator.sxPos
#audit_constructive Apeiron.Combinator.syPos
#audit_constructive Apeiron.Combinator.sxPos_ext
#audit_constructive Apeiron.Combinator.sxPos_arg
#audit_constructive Apeiron.Combinator.syPos_ext
#audit_constructive Apeiron.Combinator.syPos_arg
#audit_constructive Apeiron.Combinator.sxPos_injective
#audit_constructive Apeiron.Combinator.syPos_injective
#audit_constructive Apeiron.Combinator.sPair
#audit_constructive Apeiron.Combinator.sPair_ee
#audit_constructive Apeiron.Combinator.sPair_eae
#audit_constructive Apeiron.Combinator.sPair_eaa
#audit_constructive Apeiron.Combinator.sPair_ae
#audit_constructive Apeiron.Combinator.sPair_aa
#audit_constructive Apeiron.Combinator.sPair_x
#audit_constructive Apeiron.Combinator.sPair_y
#audit_constructive Apeiron.Combinator.bArg_zPos
#audit_constructive Apeiron.Combinator.bArg_sxPos
#audit_constructive Apeiron.Combinator.bArg_syPos
#audit_constructive Apeiron.Combinator.sPlug
#audit_constructive Apeiron.Combinator.sPlug_x
#audit_constructive Apeiron.Combinator.sPlug_y
#audit_constructive Apeiron.Combinator.relSPlug_collapse_x
#audit_constructive Apeiron.Combinator.relSPlug_collapse_y
#audit_constructive Apeiron.Combinator.sStore
#audit_constructive Apeiron.Combinator.SSubstitute
#audit_constructive Apeiron.Combinator.sBig
#audit_constructive Apeiron.Combinator.sBig_eq
#audit_constructive Apeiron.Combinator.relSLeft_reading
#audit_constructive Apeiron.Combinator.relSRight_reading
#audit_constructive Apeiron.Combinator.relSubstitution

#guard
  Apeiron.Combinator.sPair (R := ℤ)
      (fun _ => some (0, (2, 0)))
      (fun _ => some (1, (3, 0)))
      (Apeiron.Combinator.extPos
        (Apeiron.Combinator.argPos (Apeiron.Combinator.argPos 0))) =
    some (Apeiron.Combinator.syPos 1, (3, 0))
