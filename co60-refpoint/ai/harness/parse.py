"""Extract the final JSON object from a model reply and validate it against the frozen schema.

The answer format asks for one JSON object in a ```json fenced block at the end of the reply. The parser takes
the last such block that parses; with no fenced block, the last balanced {...} object in the text. Validation
covers the subset of JSON Schema the study's schemas use (object, array with min/max items, number, string with
enum, null).
"""
import json
import re

FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S)


def extract_json(text):
    blocks = FENCE.findall(text or "")
    for b in reversed(blocks):
        try:
            return json.loads(b), None
        except json.JSONDecodeError as e:
            last_err = f"fenced block is not valid JSON: {e}"
    if blocks:          # a fenced answer that does not parse: report the JSON error, not a nested object (D034)
        return None, last_err
    # Fallback: last balanced object
    s = text or ""
    end = s.rfind("}")
    while end != -1:
        depth = 0
        for i in range(end, -1, -1):
            if s[i] == "}":
                depth += 1
            elif s[i] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[i:end + 1]), None
                    except json.JSONDecodeError:
                        break
        end = s.rfind("}", 0, end)
    return None, (last_err if blocks else "no JSON object found in the reply")


def _check(v, sch, path, errs):
    t = sch.get("type")
    types = t if isinstance(t, list) else [t]
    ok = False
    for ty in types:
        if ty == "number" and isinstance(v, (int, float)) and not isinstance(v, bool):
            ok = v == v and abs(v) != float("inf")
        elif ty == "string" and isinstance(v, str):
            ok = "enum" not in sch or v in sch["enum"]
        elif ty == "null" and v is None:
            ok = True
        elif ty == "array" and isinstance(v, list):
            ok = True
            if len(v) < sch.get("minItems", 0) or len(v) > sch.get("maxItems", 10**9):
                errs.append(f"{path}: {len(v)} items, expected {sch.get('minItems')}-{sch.get('maxItems')}")
            for i, x in enumerate(v):
                _check(x, sch.get("items", {}), f"{path}[{i}]", errs)
        elif ty == "object" and isinstance(v, dict):
            ok = True
            for k in sch.get("required", []):
                if k not in v:
                    errs.append(f"{path}.{k}: missing")
            if sch.get("additionalProperties") is False:
                for k in v:
                    if k not in sch.get("properties", {}):
                        errs.append(f"{path}.{k}: not in schema")
            for k, x in v.items():
                if k in sch.get("properties", {}):
                    _check(x, sch["properties"][k], f"{path}.{k}", errs)
        if ok:
            break
    if not ok:
        errs.append(f"{path}: expected {t}, got {type(v).__name__}")


def validate(obj, schema, max_errors=20):
    errs = []
    _check(obj, schema, "$", errs)
    return errs[:max_errors]


def parse_reply(text, schema):
    obj, err = extract_json(text)
    if obj is None:
        return None, [err]
    errs = validate(obj, schema)
    return obj, errs
