"""Combined IT+OT seeding, run end to end on one unified hit table.

Every other study here treats the IT and the OT as separate systems joined only
at the fit. This one builds ONE hit table -- IT clusters as layers 1-4, OT barrel
stubs as layers 11-16 -- and drives the same seeding machinery across both, so a
seed may take its two pair hits and its projection target from either subsystem.

TWO THINGS AN OT STUB DOES NOT HAVE, and how they are handled:

  ANGLES. SmartPixels clusters carry alpha and beta; stubs carry neither. Their
  angle sigmas are set huge, so every per-hit and per-pair consistency gate
  passes trivially. The joint (z0, phi) pairing then classes them as wildcards
  and searches them on phi alone, which is the right answer rather than a
  workaround: a stub has no per-hit z0 estimate to contribute.

  A POSITION SIGMA COMPARABLE TO A PIXEL. The z search needs the target hit's own
  resolution, so it is MEASURED here per OT layer against each TP's L1TTP helix
  z = z0 + r*tanL rather than assumed: ~0.8-1.9 mm for the PS layers (L1-L3)
  and ~1.8 cm for the 2S layers (L4-L6), against 12.5 um for an IT cluster.

WHAT IT SHOWS. Mixed IT+OT seeds reach 0.902-0.924 efficiency against 0.860-0.862
for IT-only, because a long lever arm measures curvature far better and fewer
tracks are lost at the kappa cut. They also cost 7-22x more and fake 3-11x worse
-- and that is the projection study confirming itself, because this machinery
searches Z-FIRST and a mixed pair's sigma(z0) is 550 um against an IT pair's
73 um. The same arithmetic says a PHI-FIRST search inverts the cost ordering and
makes the mixed seeds the cheapest of all. That search is not implemented here,
so the efficiency advantage measured below is real but not yet reachable.
"""
import sys; sys.path.insert(0,"eval_refitq/combinatorics")
import numpy as np, json
import tracklet_topology_cost as M

# argv: [n_events] [input spec, default the regenerated PU200 ttbar set]
NEV=int(sys.argv[1]) if len(sys.argv)>1 else None
src=sys.argv[2] if len(sys.argv)>2 else M.TP_REGEN; PT=2.0

I,nev,_=M.load_flat(src, M.IT_TABLE, list(M.IT_COLS), NEV)
O,onev,_=M.load_ot(src, NEV, tp=("tp_z0","tp_tanL"))
bar=(O["isBarrel"]>0)&(O["eta"]<=M.ETA_MATCHED)
nI=len(I["layer"]); nO=int(bar.sum())

# per-OT-layer z resolution, measured (used as the target sigma in the z search)
def rs(x):
    q=np.percentile(x,[15.865,84.135]); return float(0.5*(q[1]-q[0]))
otsig={}
for L in range(1,7):
    g=bar&(O["layer"]==L)&(O["tpIdx"]>=0)&(O["tpPt"]>=PT)&np.isfinite(O["tp_z0"])
    if g.sum()<200: continue
    otsig[L]=rs(O["z"][g]-(O["tp_z0"][g]+O["r"][g]*O["tp_tanL"][g]))

# ---- unified hit table: IT layers 1-4, OT barrel layers 11-16 --------------
sigY_ot=np.array([otsig.get(int(L),0.2) for L in O["layer"][bar]])
U={"layer":np.r_[I["layer"], O["layer"][bar]+10],
   "globalR":np.r_[I["globalR"], O["r"][bar]],
   "globalZ":np.r_[I["globalZ"], O["z"][bar]],
   "globalPhi":np.r_[I["globalPhi"], O["phi"][bar]],
   "sigY":np.r_[I["sigY"], sigY_ot],
   "tpIdx":np.r_[I["tpIdx"], O["tpIdx"][bar]],
   "tpPt":np.r_[I["tpPt"], O["tpPt"][bar]],
   "event":np.r_[I["event"], O["event"][bar]]}
# angles: IT from the model; OT stubs carry NONE, so their sigmas are made huge
# and every consistency gate passes trivially. The joint (z0,phi) pairing then
# classes them as wildcards and searches them on phi alone, which is correct --
# an OT stub has no per-hit z0 estimate to contribute.
QI=M.it_prepare({k:I[k] for k in M.IT_COLS}, None)
Q={"kap_a":np.r_[QI["kap_a"], np.zeros(nO)],
   "s_kap":np.r_[QI["s_kap"], np.full(nO,1e9)],
   "z0":np.r_[QI["z0"], np.zeros(nO)],
   "s_z0":np.r_[QI["s_z0"], np.full(nO,1e9)],
   "ovf_a":np.r_[QI["ovf_a"], np.zeros(nO,bool)],
   "ovf_z":np.r_[QI["ovf_z"], np.zeros(nO,bool)]}
M.set_limits(M.triplets_for_budget(1.5), 6.0)
NM={1:"IT L1",2:"IT L2",3:"IT L3",4:"IT L4",11:"OT L1",12:"OT L2",13:"OT L3",
    14:"OT L4",15:"OT L5",16:"OT L6"}
MODES=[("IT L1L2 -> IT L3",1,2,3),("IT L1L2 -> OT L1",1,2,11),
       ("IT L2L3 -> IT L1",2,3,1),("IT L3+OT L1 -> IT L2",3,11,2),
       ("IT L4+OT L1 -> OT L2",4,11,12),("IT L4+OT L2 -> OT L1",4,12,11),
       ("OT L1L2 -> OT L3",11,12,13)]
print(f"{nev} events, {nI/nev:.0f} IT clusters + {nO/nev:.0f} OT barrel stubs per event, pT>{PT}")
print(f"OT z resolutions used as target sigma: "
      + ", ".join(f"L{k}={v*1e4:.0f}um" for k,v in sorted(otsig.items())) + "\n")
print(f"{'seed mode':<24}{'findable':>10}{'found':>9}{'eff':>7}{'fake':>7}"
      f"{'pairs/ev':>10}{'cand/ev':>10}{'fit/ev':>9}")
res={}
for nm,la,lb,lc in MODES:
    fk=M.findable_keys(U,la,lb,lc,PT)
    try:
        oo=M.it_pair_seed(U,Q,np.arange(len(U["layer"])),la,lb,lc,PT,True,False,0.0)
    except M.TooWide as e:
        print(f"{nm:<24}{len(fk):>10,d}  INFEASIBLE {e.n:.2g} candidate triplets"); continue
    rk=M.recovered_keys(U,*oo["_trip"]) if "_trip" in oo else np.empty(0,np.int64)
    nc,nt=M.cand_purity(U,*oo["_trip"]) if "_trip" in oo else (0,0)
    eff=float(np.isin(fk,rk).mean()) if len(fk) else 0.0
    fake=1.0-nt/max(nc,1)
    res[nm]=dict(n_findable=int(len(fk)),found=int(len(rk)),efficiency=eff,fake=fake,
                 pairs=oo.get("tracklets",0)/nev,cand=oo.get("match_cand",0)/nev,
                 fit=oo.get("tracks_to_fit",0)/nev)
    print(f"{nm:<24}{len(fk):>10,d}{len(rk):>9,d}{eff:>7.3f}{fake:>7.3f}"
          f"{res[nm]['pairs']:>10,.0f}{res[nm]['cand']:>10,.0f}{res[nm]['fit']:>9,.0f}")
json.dump({"n_events":nev,"ot_sigma_z_cm":otsig,"modes":res},
          open("eval_refitq/combinatorics/results/ttbar_pu200_combined_e2e.json","w"),indent=1)
