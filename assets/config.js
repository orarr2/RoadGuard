// Runtime configuration. GitHub Pages serves this verbatim; the webapp
// reads tunnel-url.json first and falls back to API_ORIGIN below.

export const CONFIG = Object.freeze({
  // Default API origin when tunnel-url.json is missing or unreachable.
  // Point at a local uvicorn during development.
  API_ORIGIN: "http://127.0.0.1:8000",

  // Set this to the OAuth 2.0 Client ID published in the Google Cloud
  // console. The webapp mounts the Google button only when it is set.
  GOOGLE_CLIENT_ID: "",

  // Local URL for the YOLO ONNX weights. The committed file is the
  // team-trained yolov8n (dataset owner: Shai), 4-class. Replace it at
  // the same path when a new version ships. The webapp refuses to run
  // without this file: no mock detector, so the agent service never
  // receives a synthesised photo.
  YOLO_MODEL_URL: "assets/models/yolov8n-roadguard.onnx",
  YOLO_INPUT_SIZE: 640,

  // YOLO_CLASS_NAMES is indexed by the model's class-id output. The
  // agent API enum is fixed at D00/D10/D20/D40, so this list must map
  // every slot to one of those four strings. The order matches the
  // team-trained yolov8n ONNX metadata (verified
  // against m.names on export):
  //   id 0 = longitudinal_crack -> D00 (handoff_contract.json: onnx_class_ids)
  //   id 1 = transverse_crack   -> D10
  //   id 2 = alligator_crack    -> D20
  //   id 3 = pothole            -> D40
  YOLO_CLASS_NAMES: ["D00", "D10", "D20", "D40"],
  // Model confidence threshold (default 0.35). The team's frozen
  // research threshold is 0.01, which pulls false positives into
  // triage; 0.35 keeps the webapp aligned with the policy bands
  // (0.6+ auto triage, 0.4-0.6 evidence, below 0.4 human review).
  YOLO_CONFIDENCE_THRESHOLD: 0.35,
  // Non-maximum-suppression IoU threshold, per the team hand-off.
  YOLO_NMS_IOU: 0.7,

  // Path to the tunnel JSON file. Holds the fixed named-tunnel hostname
  // (see deploy/cloudflared.md); it is committed by hand once and only
  // changes if the hostname changes.
  TUNNEL_URL_JSON: "tunnel-url.json",

  // Local-dev defaults for the camera.
  CAMERA_DEFAULT_FACING_MODE: "environment",
});
