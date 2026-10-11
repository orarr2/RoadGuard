"""Dataset backup script helpers: YAML class parser + class-map suggester."""

from tools.download_dataset import _parse_data_yaml_classes, _suggest_class_map


def test_parse_inline_list_names():
    yaml = """
names: ['LongitudinalCracks', 'Potholes', 'TransverseCrack']
nc: 3
train: ../train/images
""".strip()
    classes, nc = _parse_data_yaml_classes(yaml)
    assert classes == ["LongitudinalCracks", "Potholes", "TransverseCrack"]
    assert nc == 3


def test_parse_multiline_sequence_names():
    # The exact shape roboflow emits for xjsj/road-damage-10k v1.
    yaml = """
names:
- LongitudinalCracks
- Potholes
- TransverseCrack
nc: 4
test: ../test/images
train: ../train/images
val: ../valid/images
""".strip()
    classes, nc = _parse_data_yaml_classes(yaml)
    assert classes == ["LongitudinalCracks", "Potholes", "TransverseCrack"]
    assert nc == 4


def test_parse_dict_names():
    yaml = """
names:
  0: LongitudinalCracks
  1: Potholes
  2: TransverseCrack
  3: AlligatorCrack
nc: 4
""".strip()
    classes, nc = _parse_data_yaml_classes(yaml)
    assert classes == [
        "LongitudinalCracks",
        "Potholes",
        "TransverseCrack",
        "AlligatorCrack",
    ]
    assert nc == 4


def test_parse_empty_when_yaml_missing_names():
    yaml = "nc: 4\ntrain: ../train/images\n"
    classes, nc = _parse_data_yaml_classes(yaml)
    assert classes == []
    assert nc == 4


def test_class_map_covers_all_rdd_synonyms():
    classes = ["LongitudinalCracks", "Potholes", "TransverseCrack", "AlligatorCrack"]
    out = _suggest_class_map(classes)
    assert out["LongitudinalCracks"] == "D00"
    assert out["Potholes"] == "D40"
    assert out["TransverseCrack"] == "D10"
    assert out["AlligatorCrack"] == "D20"


def test_class_map_unknown_returns_empty_string():
    out = _suggest_class_map(["WeirdNewClass", "Potholes"])
    assert out["WeirdNewClass"] == ""
    assert out["Potholes"] == "D40"


def test_class_map_is_case_and_punctuation_insensitive():
    out = _suggest_class_map(["longitudinal-crack", "POTHOLE", "D40"])
    assert out["longitudinal-crack"] == "D00"
    assert out["POTHOLE"] == "D40"
    assert out["D40"] == "D40"
