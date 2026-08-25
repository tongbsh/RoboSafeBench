# LIBERO GSO obstacle pack

Adds similar-sized extras on **unused** SafeLIBERO obstacle slots.
Official types keep their stock visual and collision. Does not change
BDDL or `pruned_init`. Tasks that already use all 6 slots are unchanged.

- Enable: `--set extra.libero.obstacle_pack.enabled=true`
- Disable (official types only): `enabled=false` (default)

`official_identities.json` lists each task's BDDL slots and the Level-I
instance that is on-stage in each of the 50 init states. Pack planning
steals surplus episodes from those official types so extras also appear.

Meshes are Google Scanned Objects via
[kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects)
(CC BY 4.0). See `manifest.yaml` and `THIRD_PARTY_NOTICES.md`.

Regenerate meshes:

```
python scripts/extra/libero/materialize_obstacle_pack_gso.py \
  --gso-root /tmp/gso_full/mujoco_scanned_objects
```
