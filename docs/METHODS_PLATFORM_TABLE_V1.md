# Methods Platform Table V1

| Component | Verified value |
| --- | --- |
| Robot | Franka Panda |
| cuRobo | Public source, commit `8e734f3ced1df898990bcd92de40abce475907db` |
| Isaac Sim | 6.0.1.0 |
| cuRobo Python | 3.12.3 |
| Isaac Sim Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| GPU | NVIDIA GeForce RTX 4080 |
| Frozen trajectory representation | 41 states × 7 Panda arm joints |
| Main geometric evaluator | cuRobo collision spheres with analytical sphere–AABB signed clearance |
| Independent validation backend | Isaac Sim / PhysX contact reports |

The machine-readable source is `results_paper/methods/platform_versions.csv`. Values are copied from canonical repository metadata or directly queried from the installed project environment. On 2026-08-17, `.venv-isaacsim/bin/python` resolved to the project Isaac Sim virtual environment and reported CPython 3.12.3; its `pyvenv.cfg` independently recorded `version_info = 3.12.3`. This is the interpreter used to launch the repository's Isaac Sim validation scripts.
