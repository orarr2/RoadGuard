# Webapp YOLO model

The webapp loads `yolov8n-roadguard.onnx` from this directory at runtime
(`webapp/assets/detect.js`). If the file is missing the detector refuses
to run and throws `detector unavailable`; there is no mock fallback, so
the agent service never receives a synthesised photo.

## What is committed today

`yolov8n-roadguard.onnx` (12 MB, team-trained): the `yolov8_n_lr0003_coloraug`
checkpoint from the Shai modelling repository, exported to ONNX by the
team (see the hand-off contract and the model card in that repository).
Research-only, every verdict needs a human.

## Hand-off contract

The ONNX graph is `[1, 3, 640, 640]` float32 RGB NCHW, pixels divided by 255,
aspect-preserving letterbox with pad value 114. Output is `[1, 8, 8400]`
with channel order:

```
cx, cy, w, h, D00, D10, D20, D40
```

No separate objectness channel, no extra sigmoid, class-aware NMS applied
externally with IoU 0.7. `handoff_contract.json` in the modelling repo
holds the full specification.

## Class order

The ONNX graph reorders the native checkpoint channels so slot `i` is the
class id the webapp expects. The native checkpoint uses a different order
(`0=D20, 1=D00, 2=D40, 3=D10`); do not add a second permutation in the
browser.

```js
// webapp/assets/config.js
YOLO_CLASS_NAMES: ["D00", "D10", "D20", "D40"]
//                  0      1      2      3
```

## Replace with a new version

Drop the new `.onnx` at this path and confirm the metadata still carries
`roadguard_class_order: D00,D10,D20,D40`. The webapp picks the new file up
on the next page load.
