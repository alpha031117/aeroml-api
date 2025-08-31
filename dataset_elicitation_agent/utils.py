
import json, re, logging

def _extract_json(text: str):
    """
    Try hard to get a JSON array/object out of an LLM reply.
    1) Strip code fences
    2) Look for the first top-level JSON array/object using a bracket counter
    """
    if not text:
        return None

    # strip code fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    # fast path
    try:
        return json.loads(text)
    except Exception:
        pass

    # bracket-balanced extraction (arrays or objects)
    for opener, closer in [('[', ']'), ('{', '}')]:
        start = text.find(opener)
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            continue
    return None

def _soft_repair_json(text: str):
    if not text:
        return None
    # Strip code fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    # Already valid?
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to complete top-level array/object if brackets are unbalanced
    def try_complete(opener, closer):
        start = text.find(opener)
        if start == -1:
            return None
        depth = 0
        last_idx = None
        for i, ch in enumerate(text[start:], start):
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    last_idx = i
        if last_idx is None:
            # Close with as many closers as needed
            needed = text[start:].count(opener) - text[start:].count(closer)
            candidate = text + (closer * max(0, needed))
        else:
            candidate = text[:last_idx+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return try_complete('[', ']') or try_complete('{', '}')