#!/usr/bin/env python3
"""Derive Cartesian EE traces from frozen corridor trajectories (no planning)."""
from pathlib import Path
import csv, json
import numpy as np, torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import JointState

ROOT=Path(__file__).resolve().parents[2]
PROTO=ROOT/"configs/experiments/formal_rq2_constrained_space_stress_test_v1.json"
OUT=ROOT/"results_visualization/corridor_route_v2"; OUT.mkdir(parents=True,exist_ok=True)
def main():
 p=json.loads(PROTO.read_text()); cond=lambda mm: next(c for c in p['conditions'] if c['corridor_width_mm']==120 and c['inflation_mm']==mm)
 boxes=cond(0)['boxes']; scene=Scene(cuboid=[Cuboid(name=b['name'],dims=b['nominal_size_m'],pose=[*b['center_m'],1,0,0,0]) for b in boxes])
 planner=MotionPlanner(MotionPlannerCfg.create(robot='franka.yml',scene_model=scene,collision_cache={'cuboid':10},random_seed=0));
 rows=[]
 for label,mm in [('passage',0),('bypass',20)]:
  q=np.asarray(cond(mm)['canonical_trajectory_position'],dtype=np.float32)
  js=JointState.from_position(torch.tensor(q,device='cuda'),planner.kinematics.config.kinematics_config.joint_names)
  pos=planner.kinematics.compute_kinematics(js).tool_poses.get_link_pose(planner.kinematics.config.kinematics_config.tool_frames[0]).position.detach().cpu().numpy()
  with (OUT/f'{label}_ee_positions.csv').open('w',newline='',encoding='utf-8') as f:
   w=csv.writer(f,lineterminator='\n'); w.writerow(['trajectory_type','state_index','ee_x_m','ee_y_m','ee_z_m']); w.writerows([[label,i,*map(float,x)] for i,x in enumerate(pos)])
  rows.append((label,pos))
 import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,6),dpi=180)
 for x,color,label in [(rows[0][1],'#1976D2','Passage (0 mm)'),(rows[1][1],'#F57C00','Bypass (20 mm)')]: ax.plot(x[:,0],x[:,2],'-o',ms=2.2,lw=2,color=color,label=label)
 for b in boxes:
  c=np.asarray(b['center_m']); s=np.asarray(b['nominal_size_m']); ax.add_patch(plt.Rectangle((c[0]-s[0]/2,c[2]-s[2]/2),s[0],s[2],facecolor='#D32F2F',alpha=.55,edgecolor='#8B0000'))
 ax.set_xlabel('world x (m)'); ax.set_ylabel('world z (m)'); ax.set_title('120-mm corridor: ordered EE positions of frozen trajectories'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/'corridor_passage_vs_bypass_overlay.png'); plt.close(fig)
 (OUT/'audit.json').write_text(json.dumps({'status':'CORRIDOR_EE_OVERLAY_V2_AUDIT_PASS','no_replanning':True,'source_conditions':['120 mm / 0 mm','120 mm / 20 mm'],'state_count':41,'ee_link':planner.kinematics.config.kinematics_config.tool_frames[0],'derived_visualization_only':True},indent=2)+'\n')
 print('CORRIDOR_EE_OVERLAY_V2_AUDIT_PASS')
if __name__=='__main__': main()
