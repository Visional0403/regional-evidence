# Datasets

- [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset). Use the official 6,471-image training split and 548-image validation split for domain-expert training, adaptation and the final expert evaluations.
- [UA-DETRAC](https://sites.google.com/view/daweidu/projects/ua-detrac). Obtain the detection images and corresponding XML annotations from the benchmark authors. The mechanism replication uses samples from both official splits with frozen pretrained weights.

The `splits/` directory gives the selected image and region identities, including mask coordinates in the resized image where applicable. Each region is indexed in row-major order from zero. Raw images and official annotations are obtained from the original sources.

[`visdrone_full_image_controls.json`](../splits/visdrone_full_image_controls.json) contains the 164 natural zero-target regions and 40 same-image positive controls used with the full-image interface. Mask boxes use half-open pixel coordinates `[left, top, right, bottom]` on the resized canvas; the fill is RGB `(127, 127, 127)`. Each record includes the reference count and the final prediction under each of the three conditions.

Pass an extracted VisDrone split directory containing `images/` and `annotations/` to `prepare-visdrone`. The five questions per image cover object statistics, vehicle composition, pedestrian statistics, spatial distribution and spatial comparison. The original annotation order and a fixed wording seed determine the generated files. Use `--language zh` for the Chinese five-task protocol. All expert regional mechanism evaluations used English prompts.
