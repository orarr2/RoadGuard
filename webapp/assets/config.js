// Runtime configuration. GitHub Pages serves this verbatim; the webapp
// reads tunnel-url.json first and falls back to API_ORIGIN below.

export const CONFIG = Object.freeze({
  // Default API origin when tunnel-url.json is missing or unreachable.
  // Point at a local uvicorn during development.
  API_ORIGIN: "http://127.0.0.1:8000",

  // Set this to the OAuth 2.0 Client ID published in the Google Cloud
  // console. The webapp mounts the Google button only when it is set.
  GOOGLE_CLIENT_ID: "",

  // Local URL for the YOLO ONNX weights. Dropped into assets/models/
  // when Shai exports the trained model; the webapp falls through to a
  // mock detector when the file is missing (so the UI is testable
  // without a trained model).
  YOLO_MODEL_URL: "assets/models/yolov8n-roadguard.onnx",
  YOLO_INPUT_SIZE: 640,

  // YOLO_CLASS_NAMES is indexed by the model's class-id output. The
  // agent API enum is fixed at D00/D10/D20/D40, so this list must map
  // every slot to one of those four strings.
  //
  // Order confirmed against the actual data.yaml of xjsj/road-damage-10k
  // v1 downloaded to the VM (see /opt/roadguard/datasets/road-damage-10k/
  // manifest.json). All four classes are named - the dataset is the full
  // RDD2022 four-class set, just under friendly labels:
  //   id 0 = AlligatorCrack      -> D20
  //   id 1 = LongitudinalCracks  -> D00
  //   id 2 = Potholes            -> D40
  //   id 3 = TransverseCrack     -> D10
  YOLO_CLASS_NAMES: ["D20", "D00", "D40", "D10"],
  YOLO_CONFIDENCE_THRESHOLD: 0.35,

  // Path to the tunnel JSON file. On the VM, deploy/update-tunnel-url.sh
  // writes this file whenever cloudflared mints a new tunnel URL.
  TUNNEL_URL_JSON: "tunnel-url.json",

  // Local-dev defaults for the camera.
  CAMERA_DEFAULT_FACING_MODE: "environment",
});
