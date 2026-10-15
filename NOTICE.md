# Source and model notice

The Waffle Pi, tyre and LiDAR visual meshes and original robot XML were supplied
in the user's ROBOTIS MuJoCo model packages. Their Apache License 2.0 text is
retained at `models/robotis_tb3/LICENSE`.

The integrated robot XML is modified: wheel collision primitives, chassis
collision geometry, wheel-joint friction/armature, rear-support contact priority,
camera and sensor definitions. The visual STL files are unchanged. Compare the
original robot XML in `reference_sources/Part-A/Models/robotis_tb3/`.

The user's section code is preserved in `reference_sources/` and adapted as
described in `docs/INTEGRATION.md`. No new blanket licence is imposed on the
user's code. `docs/SOURCE_MANIFEST.json` records supplied archive hashes and
source provenance.
