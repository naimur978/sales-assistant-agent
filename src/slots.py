"""Safety net against guessed tool arguments: a slot value must be traceable to what the user said.
The model is told never to guess; this enforces it in code."""
import difflib
import itertools
import re
import unicodedata

from src.verifier import numbers

NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())} | {"dozen": 12, "couple": 2}

EQUIPMENT_HINTS = {
    "larox_pf": ["pf"], "larox_pf_ds": ["pf-ds", "pfds", "pf ds", "tower", "double-sided", "double sided"],
    "columncell": ["columncell", "column"], "concorde_cell": ["concorde"], "cellstation": ["cellstation", "cell station"],
    "filter_optimizer": ["optimizer", "optimiser"], "md_pump": ["md", "mill discharge"], "orion_pump": ["orion"],
    "sala_vsp750": ["sala", "vsp", "sump"],
}
# Saying only "Larox" does not tell the PF pressure filter from the PF-DS tower press, so it is not enough for either.
# And a message that names the PF-DS must not be accepted for plain larox_pf.
EQUIPMENT_EXCLUDES = {"larox_pf": ["pf-ds", "pfds", "pf ds", "tower", "double-sided", "double sided"]}
JOB_HINTS = {
    "cloth_replacement": ["cloth"],
    "wear_part_replacement": ["replac", "impeller", "rotor", "seal", "plate", "wear", "swap", "change"],
    "inspection_or_preventive_maintenance": ["inspect", "preventive", "maintenance", "check", "pm"],
    "software_update_or_calibration": ["software", "update", "upgrade", "calibrat", "algorithm", "recalibr"],
    "commissioning_or_installation": ["commission", "install"],
    "emergency_repair": ["emergency", "repair", "breakdown", "shutdown", "fault", "failure"],
    "cleaning": ["clean"],
}


def _has_hint(hint: str, low: str) -> bool:
    """Word-start match, and a whole-word match for short hints, so 'md' does not match 'command'."""
    tail = r"(?![a-z])" if len(hint) <= 3 else ""
    return re.search(rf"(?<![a-z]){re.escape(hint)}{tail}", low) is not None


# Names that mean the same country, so "UK", "Britain" or "Chilean" count as the user having said it.
COUNTRY_GROUPS = [
    {"chile", "chilean"}, {"peru", "peruvian"}, {"sweden", "swedish"}, {"russia", "russian"}, {"finland", "finnish"},
    {"zambia", "zambian"}, {"australia", "australian"}, {"canada", "canadian"}, {"south africa", "south african", "rsa"},
    {"united states", "usa", "us", "u.s.", "america", "american"}, {"united kingdom", "uk", "u.k.", "britain", "british", "england"},
]


def _plain(text: str) -> str:
    """Lower-case and strip accents ('Perú' -> 'peru')."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def canonical_country(name: str, known: list[str]) -> str:
    """Map what was typed ('chille', 'Chilean') onto a country spelled the way the data spells it. Unknown names stay as typed."""
    n = _plain(name).strip()
    by_plain = {_plain(k): k for k in known}
    if n in by_plain:
        return by_plain[n]
    for group in COUNTRY_GROUPS:
        if n in group:
            for member in sorted(group):
                if member in by_plain:
                    return by_plain[member]
    close = difflib.get_close_matches(n, list(by_plain), n=1, cutoff=0.85) if len(n) >= 4 else []
    return by_plain[close[0]] if close else name


def _country_in(country: str, text: str) -> bool:
    c, t = _plain(country).strip(), _plain(text)
    names = {c}
    for group in COUNTRY_GROUPS:
        if c in group:
            names |= group
    for n in names:
        tail = r"(?![a-z])" if len(n) <= 3 else ""
        if re.search(rf"(?<![a-z]){re.escape(n)}{tail}", t):
            return True
    # a small typo ("Chille") still counts, but only for names long enough to be unambiguous
    return len(c) >= 4 and bool(difflib.get_close_matches(c, re.findall(r"[a-z]+", t), n=1, cutoff=0.85))


LABELS = {"silica": ["silica", "sio2"], "iron": ["iron", "fe"]}
NEAR = 25  # a reading must sit this close to its label


def _pair_readings(value: float, low: str, spots: list, name: str) -> bool:
    """Both labels are present. Pair the labels with the numbers jointly (smallest total gap), so a long gap such as
    '3.1 percent silica, 64.5 percent Fe' cannot make a number drift to the wrong label."""
    gap = lambda n, lab: max(n[0] - lab[1], lab[0] - n[1], 0)
    nums = [(m.start(), m.end(), float(m.group().replace(",", "."))) for m in re.finditer(r"\d+(?:[.,]\d+)?", low)]
    first = {slot: min((sp for sp in spots if sp[2] == slot), key=lambda sp: sp[0]) for slot in LABELS}
    best, best_cost = None, None
    for a, b in itertools.permutations(range(len(nums)), 2):
        cost = gap(nums[a], first["silica"]) + gap(nums[b], first["iron"])
        if best_cost is None or cost < best_cost:
            best, best_cost = {"silica": nums[a][2], "iron": nums[b][2]}, cost
    return best is not None and abs(best[name] - value) < 1e-9


def _number_near_label(value: float, text: str, name: str) -> bool:
    """A reading counts for a slot only if that slot's label is the closest label to the number."""
    low = text.lower()
    spots = [(m.start(), m.end(), slot) for slot, labs in LABELS.items() for lab in labs
             for m in re.finditer(rf"(?<![a-z]){lab}(?![a-z])", low)]
    if not any(slot == name for *_, slot in spots):
        return _number_in(value, text)  # the user never labelled it: any mention of the number counts
    if {slot for *_, slot in spots} >= set(LABELS):
        return _pair_readings(value, low, spots, name)
    for m in re.finditer(r"\d+(?:[.,]\d+)?", low):
        if abs(float(m.group().replace(",", ".")) - value) > 1e-9:
            continue
        dist = lambda a, b, m=m: max(a - m.end(), m.start() - b, 0)
        best = min(spots, key=lambda sp: dist(sp[0], sp[1]))
        if best[2] == name and dist(best[0], best[1]) <= NEAR:
            return True
    return False


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _number_in(value: float, text: str) -> bool:
    nums = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))}
    nums |= {float(n.replace(",", ".")) for n in numbers(text) if re.fullmatch(r"\d+(?:[.,]\d+)?", n)}
    nums |= {float(NUMBER_WORDS[w]) for w in _words(text) if w in NUMBER_WORDS}
    return any(abs(value - n) < 1e-9 for n in nums)


def ungrounded(args: dict, user_text: str) -> list[str]:
    """Names of slots whose value the user never said. Empty list = all grounded."""
    text, low = user_text, user_text.lower()
    bad = []
    for name in ("quantity", "silica", "iron", "silica_max", "iron_min"):
        if args.get(name) is None:
            continue
        v = float(args[name])
        if not (_number_near_label(v, text, name) if name in LABELS else _number_in(v, text)):
            bad.append(name)
    country = args.get("country")
    if country and not _country_in(country, text):
        bad.append("country")
    eq = args.get("equipment")
    if eq and eq != "other":
        said = any(_has_hint(h, low) for h in EQUIPMENT_HINTS.get(eq, []))
        contradicted = any(_has_hint(x, low) for x in EQUIPMENT_EXCLUDES.get(eq, []))
        if not said or contradicted:
            bad.append("equipment")
    job = args.get("job")
    if job and not any(_has_hint(h, low) for h in JOB_HINTS.get(job, [])):
        bad.append("job")
    return bad
