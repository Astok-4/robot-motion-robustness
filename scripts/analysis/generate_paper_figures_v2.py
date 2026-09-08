#!/usr/bin/env python3
"""Deterministic plotting from frozen formal analysis CSVs only."""
from pathlib import Path
import csv, math
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT/'results_analysis/formal_analysis_v1'; OUT=ROOT/'results_analysis/paper_figures_v2_2'; OUT.mkdir(parents=True,exist_ok=True)
LABEL={'formal_baseline_v1':'Scene A','multiscene_candidate_001':'Scene B','multiscene_candidate_005':'Scene C','multiscene_candidate_006':'Scene D'}
def read(name):
 with (SRC/name).open(newline='') as f:return list(csv.DictReader(f))
plt.rcParams.update({'font.size':10,'axes.labelsize':11,'legend.fontsize':9,'figure.dpi':160})
def save(fig,n): fig.tight_layout(); fig.savefig(OUT/(n+'.png'),dpi=220); fig.savefig(OUT/(n+'.svg')); plt.close(fig)
def f(x): return float(x)

def fig1():
 rows=read('formal_condition_table.csv'); fig,ax=plt.subplots(figsize=(6.6,4.8))
 for sid,g in __import__('itertools').groupby(rows,key=lambda r:r['scene_id']):
  g=list(g); x=[f(r['c_curobo_m'])*1000 for r in g]; y=[]; cens=[]
  for r in g:
   y.append(f(r['minimum_refined_physx_tolerance_midpoint_estimate_mm'])); cens.append(f(r['right_censored_fraction'])>0.99)
  ax.plot(x,y,'o-',label=LABEL[sid],ms=4)
  for xx,yy,cc in zip(x,y,cens):
   if cc: ax.plot(xx,60,'v',mfc='white',mec='black',ms=6)
 ax.set(xlabel='Nominal cuRobo clearance (mm)',ylabel='Minimum sampled PhysX tolerance midpoint (mm)',ylim=(5,36)); ax.legend(ncol=2); save(fig,'figure1_clearance_vs_physx_tolerance_v2_1')
def fig2():
 rows=read('rq1_scene_trends.csv'); fig,axs=plt.subplots(2,2,figsize=(7,5.5),sharex=True,sharey=True)
 for ax,(sid,g) in zip(axs.flat,__import__('itertools').groupby(rows,key=lambda r:r['scene_id'])):
  g=list(g); mags=[10,20,30,40,50,60]
  for m in mags: ax.plot([f(r['inflation_mm']) for r in g],[f(r[f'collision_rate_at_{m}mm']) for r in g],'-o',ms=3,label=f'{m} mm')
  ax.set_title(LABEL[sid]); ax.grid(alpha=.2)
 axs[0,0].set_ylabel('Empirical collision fraction'); axs[1,0].set_ylabel('Empirical collision fraction'); axs[1,0].set_xlabel('Planning inflation (mm)'); axs[1,1].set_xlabel('Planning inflation (mm)'); axs[0,1].legend(title='Perturbation magnitude',fontsize=7,ncol=2); save(fig,'figure2_physx_robustness_curves_v2_2')
def fig3():
 rows=read('rq2_tradeoff_summary.csv'); fig,axs=plt.subplots(1,3,figsize=(11,3.5))
 for sid,g in __import__('itertools').groupby(rows,key=lambda r:r['scene_id']):
  g=list(g); g.sort(key=lambda r:int(r['inflation_mm'])); base=f(g[0]['trajectory_length_rad']); xs=[int(r['inflation_mm']) for r in g]; rel=[(f(r['trajectory_length_rad'])-base)/base*100 for r in g]; axs[0].plot(xs,rel,'-o',label=LABEL[sid],ms=3); axs[1].errorbar(xs,[f(r['planner_median_s'])*1000 for r in g],yerr=[f(r['planner_iqr_s'])*500 for r in g],fmt='-o',ms=3,label=LABEL[sid]); axs[2].plot(xs,[f(r['minimum_tolerance_midpoint_estimate_mm']) for r in g],'-o',ms=3,label=LABEL[sid])
 axs[0].set(xlabel='Planning inflation (mm)',ylabel='Relative trajectory-length change (%)'); axs[1].set(xlabel='Planning inflation (mm)',ylabel='Planner solve time (ms)'); axs[2].set(xlabel='Planning inflation (mm)',ylabel='Minimum sampled PhysX tolerance midpoint (mm)'); axs[0].legend(fontsize=7); save(fig,'figure3_safety_margin_benefit_cost_v2_2')
def fig4():
 rows=read('rq3_disagreement_distances.csv'); vals=np.array([max(f(r['absolute_curobo_clearance_mm']),f(r['absolute_physx_separation_mm'])) for r in rows]); fig,axs=plt.subplots(1,2,figsize=(8,3.6)); bars=axs[0].bar(['cuRobo-safe /\nPhysX-collision','cuRobo-collision /\nPhysX-safe'],[1009,69],color=['#4C78A8','#F58518']); axs[0].bar_label(bars,labels=['1009','69'],padding=3); axs[0].set_ylabel('Disagreement count'); axs[1].plot(np.sort(vals),np.arange(1,len(vals)+1)/len(vals),color='#4C78A8'); axs[1].set(xlabel='Distance to opposite-engine boundary (mm)',ylabel='ECDF'); [axs[1].axvline(v,color='0.7',ls='--',lw=.8) for v in (1,2,3)]; axs[1].set_xlim(left=0); save(fig,'figure4_cross_engine_disagreement_v2_2')
if __name__=='__main__': fig1(); fig2(); fig3(); fig4(); print(OUT)
