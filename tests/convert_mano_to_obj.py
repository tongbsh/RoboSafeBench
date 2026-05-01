"""Generate a semi-grasping MANO left hand mesh for pushing a cylindrical object."""
import sys, types, pickle, struct
import numpy as np
from scipy.sparse import issparse

# --- Fake chumpy ---
_mods = {}
def _mk(name):
    m = types.ModuleType(name)
    _mods[name] = m
    sys.modules[name] = m
    return m
_root = _mk("chumpy")
for sub in ["ch","utils","reordering","optimization","logic","ch_ops","linalg","testing_functions"]:
    child = _mk(f"chumpy.{sub}")
    setattr(_root, sub, child)
class _Ch:
    def __init__(self, *a, **kw): pass
    def __setstate__(self, state):
        if isinstance(state, dict): self.__dict__.update(state)
        elif isinstance(state, (tuple,list)) and len(state)>=1 and isinstance(state[-1], dict):
            self.__dict__.update(state[-1])
    def __array__(self):
        return np.asarray(self.x) if hasattr(self,'x') else np.empty(0)
for mod in _mods.values():
    for attr in ["Ch","MatVecMult","array","zeros","concatenate","vstack","hstack","Select","Rodrigues"]:
        setattr(mod, attr, _Ch)

# --- Load MANO ---
with open("assets/mano_v1_2/models/MANO_LEFT.pkl","rb") as f:
    mano = pickle.load(f, encoding="latin1")

def to_np(x):
    if issparse(x): return np.array(x.todense())
    if isinstance(x, np.ndarray): return x
    return np.array(x)

v_template = to_np(mano['v_template'])       # (778, 3)
faces      = to_np(mano['f']).astype(np.int32) # (1538, 3)
J_regressor= to_np(mano['J_regressor'])       # (16, 778)
weights    = to_np(mano['weights'])            # (778, 16)
posedirs   = to_np(mano['posedirs'])           # (778, 3, 135)
hands_comp = to_np(mano['hands_components'])   # (ncomps, 45)
hands_mean = to_np(mano['hands_mean'])         # (45,)
kintree    = to_np(mano['kintree_table']).astype(int)  # (2, 16)

print(f"v_template: {v_template.shape}, faces: {faces.shape}")
print(f"hands_comp: {hands_comp.shape}, hands_mean: {hands_mean.shape}")

# --- Rodrigues: axis-angle → 3x3 rotation ---
def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta < 1e-8:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)

# --- Set PCA coefficients for semi-grasp pose ---
# Positive first component = finger curl (closing fist)
n_comps = hands_comp.shape[0]
pca_coefs = np.zeros(n_comps)
pca_coefs[0] = 2.0   # strong finger curl
pca_coefs[1] = 0.5   # slight spread adjustment

hand_pose = hands_mean + hands_comp.T @ pca_coefs  # (45,)
# Global rotation: identity (will be set in MuJoCo)
full_pose = np.concatenate([np.zeros(3), hand_pose])  # (48,) = 16 joints * 3

# --- Forward kinematics with LBS ---
n_joints = 16
J = J_regressor @ v_template  # (16, 3)

# Pose blend shapes
pose_rots = np.array([rodrigues(full_pose[i*3:i*3+3]) for i in range(n_joints)])
pose_feature = (pose_rots[1:] - np.eye(3)).flatten()  # (135,)
pose_bs = posedirs.reshape(778*3, -1) @ pose_feature  # (2334,)
v_posed = v_template + pose_bs.reshape(778, 3)

# Build global transforms
local_transforms = np.zeros((n_joints, 4, 4))
for i in range(n_joints):
    local_transforms[i, :3, :3] = pose_rots[i]
    local_transforms[i, :3, 3] = J[i]
    local_transforms[i, 3, 3] = 1.0

global_transforms = np.zeros((n_joints, 4, 4))
for i in range(n_joints):
    parent = kintree[0, i]
    if i == 0:
        global_transforms[i] = local_transforms[i]
    else:
        # Relative transform
        rel = local_transforms[i].copy()
        rel[:3, 3] = J[i] - J[parent]
        global_transforms[i] = global_transforms[parent] @ rel

# Remove rest-pose joint positions
for i in range(n_joints):
    j_homo = np.array([J[i,0], J[i,1], J[i,2], 1.0])
    global_transforms[i, :3, 3] -= (global_transforms[i] @ j_homo)[:3]

# Skinning
v_skinned = np.zeros_like(v_posed)
for i in range(n_joints):
    T = global_transforms[i]
    w = weights[:, i:i+1]  # (778, 1)
    v_homo = np.hstack([v_posed, np.ones((len(v_posed),1))])  # (778, 4)
    v_transformed = (T @ v_homo.T).T[:, :3]  # (778, 3)
    v_skinned += w * v_transformed

verts = v_skinned.astype(np.float32)
print(f"Skinned verts bounds: min={verts.min(0)}, max={verts.max(0)}")
print(f"Size: {verts.max(0) - verts.min(0)}")

# --- Save OBJ ---
obj_path = "assets/mano_v1_2/mano_left_hand_grasp.obj"
with open(obj_path, "w") as f:
    for v in verts:
        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for face in faces + 1:
        f.write(f"f {face[0]} {face[1]} {face[2]}\n")
print(f"OBJ saved: {obj_path}")

# --- Save STL ---
stl_path = "assets/mano_v1_2/mano_left_hand_grasp.stl"
with open(stl_path, "wb") as f:
    f.write(b'\0'*80)
    f.write(struct.pack('<I', len(faces)))
    for face in faces:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        n = np.cross(v1-v0, v2-v0)
        nl = np.linalg.norm(n)
        if nl > 0: n /= nl
        f.write(struct.pack('<fff',*n))
        for vv in [v0,v1,v2]: f.write(struct.pack('<fff',*vv))
        f.write(struct.pack('<H',0))
print(f"STL saved: {stl_path}")

# Copy to assets
import shutil
dst = "simulator_env/dyniSafety_libero/libero/libero/assets/hand_objects/hand_pusher/visual/hand_pusher_vis.stl"
shutil.copy2(stl_path, dst)
print(f"Copied to: {dst}")
