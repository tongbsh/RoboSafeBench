# Third-Party Notices

DyniSafety redistributes or depends on the following **core** third-party
software and documents one **restricted** asset class that is **not** shipped
with this repository.

**Coverage:** SimplerEnv-OpenVLA, ManiSkill2_real2sim, RoboTwin,
LIBERO / dyniSafety_libero, robosuite, RoboCasa, and MANO/SMPL+H. Nested licenses under RoboTwin
`policy/**` or `curobo` are **out of scope** for this file; consult those trees
if you vendor or redistribute them.

Operational pins, patches, and install commands:
[`docs/third_party.md`](docs/third_party.md).

---

## 1. SimplerEnv-OpenVLA

- **Location:** `third_party/SimplerEnv-OpenVLA` (git submodule)
- **Upstream:** `git@github.com:DelinQu/SimplerEnv-OpenVLA.git`
- **Pin:** `ccfe3809766839a2fcfb7a3d3c9abff585189188`
- **License file:** `third_party/SimplerEnv-OpenVLA/LICENSE`
- **License:** MIT License  
  Copyright (c) 2024 simpler-env

```
MIT License

Copyright (c) 2024 simpler-env

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. ManiSkill2_real2sim

- **Location:** `third_party/SimplerEnv-OpenVLA/ManiSkill2_real2sim`
- **Upstream:** `https://github.com/allenzren/ManiSkill2_real2sim.git`
- **Pin:** `91d154bfd864577f8d2e80f3fc2f8b4d9df9ae5c`
- **License file:** `third_party/SimplerEnv-OpenVLA/ManiSkill2_real2sim/LICENSE`
- **License:** Apache License, Version 2.0

You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The full Apache-2.0 text ships with the ManiSkill2 tree; DyniSafety does not
restate the entire license body here.

---

## 3. RoboTwin (DyniSafety private mirror)

- **Location:** `third_party/RoboTwin` (git submodule)
- **Lineage:** [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin);
  pin is a **private DyniSafety mirror** (tag `dynisafety-pin-20260806`), not a
  claim of bit-identity with a public upstream SHA.
- **Pin:** `afd5f1bbc5759ac1512dc42b3cdaf8a218fb7c00`
- **License file:** `third_party/RoboTwin/LICENSE`
- **License:** MIT License  
  Copyright (c) 2025 Tianxing Chen (陈天行)

```
MIT License

Copyright (c) 2025 Tianxing Chen (陈天行)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 4. LIBERO / dyniSafety_libero (SafeLIBERO)

- **Location (install):** `third_party/dyniSafety_libero` (created by
  `scripts/install/install_libero.sh`; **not** a git submodule)
- **Upstream:** `https://github.com/Lifelong-Robot-Learning/LIBERO.git`
- **Reference commit:** `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- **License file:** `LICENSE` in the installed / upstream tree
- **License:** MIT License  
  Copyright (c) 2023 Lifelong Robot Learning

```
MIT License

Copyright (c) 2023 Lifelong Robot Learning

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

DyniSafety’s evaluation fork may add BDDL/assets on top of LIBERO; those
additions remain under the same MIT notice unless a nested file states otherwise.

---

## 5. MANO / SMPL+H — **not redistributed**

RoboCasa L3 **default** visuals are original clothed meshes in
`src/dynisafety/backends/robocasa/assets/pedestrian/` (see §5b). They do
**not** require SMPL-family weights. The optional exporter in
`src/dynisafety/backends/robocasa/smpl_pedestrian.py` is only for a local
rebuild if you already hold MPI models.

- **Owner:** Max Planck Institute for Intelligent Systems /
  Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPI / MPG)
- **Status in DyniSafety:** **Do not ship** model binaries, meshes, or derived
  redistributable archives in this repository or in DyniSafety releases.
- **User action:** Obtain the model from the official MANO distribution after
  accepting MPI/MPG terms. Place files under a local gitignored path such as
  `<workspace>/assets/mano_v1_2/`. Print guidance with:

  ```bash
  python scripts/install/download_assets.py --mano-notice
  ```

### License grant (summary)

Max-Planck grants a non-exclusive, non-transferable, free of charge right to
download and use the Model on computers you control **solely** for
non-commercial scientific research, non-commercial education, or non-commercial
artistic projects.

**Prohibited (non-exhaustive):** commercial products/services; use as training
data for commercial systems; commercial ergonomic analysis; redistribution or
making the Model available to third parties without prior written permission;
pornographic use; reverse engineering.

The Model is provided **AS IS** without warranties. Contact for commercial
licensing: `ps-license@tue.mpg.de`.

Full terms accompany the official MANO download (`LICENSE.txt`). If a copy
exists only on a developer machine under `assets/mano_v1_2/LICENSE.txt`, that
file remains **local** and must not be committed.

---

## 6. robosuite

- **Location:** `third_party/robosuite` (install-script clone; **not** a git submodule)
- **Upstream:** `https://github.com/ARISE-Initiative/robosuite.git`
- **Pin:** `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` (git master, package 1.5.2)
- **License file:** `third_party/robosuite/LICENSE`
- **License:** MIT License  
  Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab

```
MIT License

Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Do **not** substitute PyPI `robosuite` 1.4.x for this tree.

---

## 7. RoboCasa

- **Location:** `third_party/robocasa` (install-script clone; **not** a git submodule)
- **Upstream:** `https://github.com/robocasa/robocasa.git`
- **Pin:** `921c9a5736a8d0ea5589657898aadcfa55a6a195` (package 1.0.1)
- **License file:** `third_party/robocasa/LICENSE`
- **License (code):** MIT License  
  Copyright (c) 2026 the RoboCasa Team
- **License (kitchen assets and datasets):** Creative Commons Attribution 4.0
  International (CC BY 4.0) — https://creativecommons.org/licenses/by/4.0/
- **Default asset download:** lightweight (`tex`, `tex_generative`,
  `fixtures_lw`, `objs_lw`). Full ~10GB registry only when
  `DYNISAFETY_ROBOCASA_FULL_ASSETS=1`.

```
MIT License

Copyright (c) 2026 the RoboCasa Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The CC BY 4.0 text is not restated in full here; obtain it from Creative Commons.
Kitchen assets downloaded by `robocasa.scripts.download_kitchen_assets` remain
under CC BY 4.0 and are **not** committed to this repository.

---

## 5b. DyniSafety original RoboCasa visuals (in-tree)

- **Location:** `src/dynisafety/backends/robocasa/assets/obstacles/*.obj` and
  `src/dynisafety/backends/robocasa/assets/pedestrian/*.obj`
- **Generator:** `src/dynisafety/backends/robocasa/assets/generate.py`
- **Status:** Original low-poly meshes authored for DyniSafety. They are
  **not** third-party scans (not OpenUSD / Reallusion / Mixamo / SMPL).
- **Use:** L1/L2 visual pool and L3 clothed pedestrian. Collision half-sizes
  stay in YAML; these files are appearance only.
- **License:** same as this repository (see `LICENSE`). Redistributable with
  the open-source tree.

---

## 5c. Google Scanned Objects overlay (LIBERO obstacle pack)

- **Location:** `src/dynisafety/backends/libero/assets/obstacle_pack_gso/`
- **Upstream meshes:** [kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects)
  (Google Scanned Objects, CC BY 4.0)
- **Generator:** `scripts/libero/materialize_obstacle_pack_gso.py`
- **Use:** Optional remesh of **unused** SafeLIBERO obstacle slots onto
  similar-sized GSO scans. Official on-stage types keep stock meshes.
  Off by default (`extra.libero.obstacle_pack.enabled=false`). Does not
  add joints or rewrite `pruned_init`.
- **Shipped scans:** Weston 12 fl oz tonics (wine-like); ACE / Threshold mugs
  (mug-like). See `manifest.yaml` in the pack directory.

The CC BY 4.0 text is not restated in full here; obtain it from Creative Commons.

---

## Disclaimer

Nothing in this file grants rights beyond those in the cited upstream licenses.
If you redistribute DyniSafety with bundled third-party source trees, you must
preserve the corresponding `LICENSE` files and any required notices.
