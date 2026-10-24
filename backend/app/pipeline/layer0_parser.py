"""Layer 0 — parse a clinical note into chief-complaint flags.

Two-mode implementation:
  1. Ollama / Llama 3 with structured JSON output (when Ollama is reachable)
  2. Deterministic keyword matcher (always available; used as fallback)

Both modes emit ONLY real chief-complaint columns of the trained models
(`cc_*`, 199 Yale/Epic complaints, names after `sanitize_name`). The keyword
matcher covers the most common PL+EN triage signals; the LLM can pick any of
the 199 codes (Polish abbreviations, paraphrases, embedded vitals etc.).
Code combinations follow the training data: one code per keyword rule (the most
frequent Yale synonym) and a generic complaint is dropped when a specific one
fired (REDUNDANT_GENERIC) — in both modes.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from app.inference.ollama_client import OllamaClient, OllamaError

log = logging.getLogger(__name__)


# ── Chief-complaint vocabulary of the trained models ─────────────────
# Mirrors `groups.chief_complaints` in training/data/processed/esi/feature_sets.json
# (without the `cc_` prefix). The LLM is told to pick only from this list and
# every code emitted by this module is guaranteed to be a model feature
# (tests/test_layer0_parser.py checks it against feature_sets.json and against
# the list training's prepare builds). Yale's two-tier trauma activation
# (fulltrauma / modifiedtrauma) is merged in training into "trauma_activation".
KNOWN_CC_CODES: list[str] = [
    "abdominalcramping", "abdominaldistention", "abdominalpain", "abdominalpainpregnant",
    "abnormallab", "abscess", "addictionproblem", "agitation", "alcoholintoxication",
    "alcoholproblem", "allergicreaction", "alteredmentalstatus", "animalbite",
    "ankleinjury", "anklepain", "anxiety", "arminjury", "armpain", "armswelling",
    "assaultvictim", "asthma", "backpain", "bleeding_bruising", "blurredvision",
    "bodyfluidexposure", "breastpain", "breathingdifficulty", "breathingproblem", "burn",
    "cardiacarrest", "cellulitis", "chestpain", "chesttightness", "chills",
    "coldlikesymptoms", "confusion", "conjunctivitis", "constipation", "cough", "cyst",
    "decreasedbloodsugar_symptomatic", "dehydration", "dentalpain", "depression",
    "detoxevaluation", "diarrhea", "dizziness", "drug_alcoholassessment", "drugproblem",
    "dyspnea", "dysuria", "earpain", "earproblem", "edema", "elbowpain",
    "elevatedbloodsugar_nosymptoms", "elevatedbloodsugar_symptomatic", "emesis",
    "epigastricpain", "epistaxis", "exposuretostd", "extremitylaceration",
    "extremityweakness", "eyeinjury", "eyepain", "eyeproblem", "eyeredness", "facialinjury",
    "faciallaceration", "facialpain", "facialswelling", "fall", "fall_65", "fatigue",
    "femaleguproblem", "fever", "fever_75yearsorolder", "fever_9weeksto74years",
    "feverimmunocompromised", "fingerinjury", "fingerpain", "fingerswelling", "flankpain",
    "follow_upcellulitis", "footinjury", "footpain", "footswelling", "foreignbodyineye",
    "generalizedbodyaches", "gibleeding", "giproblem", "groinpain",
    "hallucinations", "handinjury", "handpain", "headache",
    "headache_newonsetornewsymptoms", "headache_recurrentorknowndxmigraines",
    "headachere_evaluation", "headinjury", "headlaceration", "hematuria", "hemoptysis",
    "hippain", "homicidal", "hyperglycemia", "hypertension", "hypotension", "influenza",
    "ingestion", "insectbite", "irregularheartbeat", "jawpain", "jointswelling",
    "kneeinjury", "kneepain", "laceration", "leginjury", "legpain", "legswelling",
    "lethargy", "lossofconsciousness", "maleguproblem", "mass", "medicalproblem",
    "medicalscreening", "medicationproblem", "medicationrefill", "migraine",
    "motorcyclecrash", "motorvehiclecrash", "multiplefalls",
    "nasalcongestion", "nausea", "nearsyncope", "neckpain", "neurologicproblem", "numbness",
    "oralswelling", "otalgia", "other", "overdose_accidental", "overdose_intentional",
    "pain", "palpitations", "panicattack", "pelvicpain", "poisoning", "post_opproblem",
    "psychiatricevaluation", "psychoticsymptoms", "rapidheartrate", "rash",
    "rectalbleeding", "rectalpain", "respiratorydistress", "ribinjury", "ribpain",
    "seizure_newonset", "seizure_priorhxof", "seizures", "shortnessofbreath",
    "shoulderinjury", "shoulderpain", "sicklecellpain", "sinusproblem", "skinirritation",
    "skinproblem", "sorethroat", "stdcheck", "strokealert", "suicidal",
    "suture_stapleremoval", "swallowedforeignbody", "syncope", "tachycardia",
    "testiclepain", "thumbinjury", "tickremoval", "toeinjury", "toepain", "trauma",
    "trauma_activation", "unresponsive", "uri", "urinaryfrequency", "urinaryretention", "urinarytractinfection",
    "vaginalbleeding", "vaginaldischarge", "vaginalpain", "weakness", "wheezing",
    "withdrawal_alcohol", "woundcheck", "woundinfection", "woundre_evaluation",
    "wristinjury", "wristpain",
]
_KNOWN_CC_SET: frozenset[str] = frozenset(KNOWN_CC_CODES)

# Codes removed from the training data → their current column (LLM / older clients)
LEGACY_CC_CODES: dict[str, str] = {
    "fulltrauma": "trauma_activation",
    "modifiedtrauma": "trauma_activation",
}


# ── Pattern helpers ───────────────────────────────────────────────────

def _near(a: str, b: str, gap: int = 30) -> str:
    """Both stems within one phrase (≤ gap chars, no , . ; or newline between), any order."""
    sep = rf"[^.,;\n]{{0,{gap}}}?"
    return rf"\b(?:{a}){sep}\b(?:{b})|\b(?:{b}){sep}\b(?:{a})"


def _cooccur(a: str, b: str, unless: str | None = None) -> str:
    """Both stems anywhere in the note (zero-width match) and — if given — no phrase
    of `unless` (negation, e.g. "test ciążowy ujemny") anywhere in the note."""
    guard = rf"(?![\s\S]*?\b(?:{unless}))" if unless else ""
    return rf"^{guard}(?=[\s\S]*?\b(?:{a}))(?=[\s\S]*?\b(?:{b}))"


def _without(a: str, b: str) -> str:
    """Stem `a` present and stem `b` absent anywhere in the note (zero-width match)."""
    return rf"^(?![\s\S]*?\b(?:{b}))(?=[\s\S]*?\b(?:{a}))"


_INJURY = r"złama|uraz|skręc|zwichn|stłucz|fractur|injur|sprain|broken"
_SWELLING = r"obrzęk|opuchl|opuchn|swelling|swollen|edema|oedema"
_FEVER = r"gorącz|fever|febr|sepsa|sepsy|sepsis|septic|posocznic"
_OVERDOSE = r"przedawkow|overdose|zatruci\w* lek|intoksykac\w* lek|połkn\w* \w* ?tabletk"
_SUICIDE = r"samobój|suicid|celow\w* przedawk|intentional"
# Negated mentions — a co-occurrence rule must not fire on "test ciążowy ujemny"
# or "bez immunosupresji" (the generic complaint would be dropped for a wrong code)
_NEG_PREGNANCY = (
    r"ciąż\w*[\s:-]+(?:wykluczon|ujemn|negatywn)|wyklucz\w* ciąż|nie (?:jest )?w ciąży"
    r"|bez ciąży|brak ciąży|neguje ciąż|not pregnant|denies pregnan|negative (?:urine )?pregnan"
    r"|pregnancy test[\s:-]+negative|pregnancy ruled out"
)
_NEG_IMMUNOSUPPRESSION = (
    r"(?:bez|brak|neguje|nie(?: jest| był\w*)?(?: w trakcie| po)?)\s+"
    r"(?:immunosupres|chemioterapi|neutropeni|przeszczep)"
    r"|not immunocompromis|non-immunocompromis|no (?:neutropeni|transplant)"
)


# Keyword → cc_* mapping (PL + EN). Patterns use leading \b but no trailing \b
# because Polish stems (drgawk-, gorączk-, samobójcz-) need to match across
# different word endings (drgawki, gorączki, samobójcza).
# Every target MUST be a real chief-complaint column (see KNOWN_CC_CODES).
# ONE code per rule: Yale codes a visit with one of several synonymous complaints,
# practically never with two of them (e.g. Shortness of Breath + Dyspnea: 39 of
# 446k train visits) — the rule emits the synonym most frequent in train.parquet,
# so the models see a combination they were trained on.
KEYWORD_MAP: dict[str, str] = {
    # ── Critical (ESI 1 signals) ──
    r"\b(zatrzymani\w* krążenia|nzk\b|cardiac arrest|resuscytac|reanimac|rko\b|cpr\b)":
        "cc_cardiacarrest",
    # Trauma-team activation / pre-notification ("powiadomienie o pacjencie urazowym")
    (r"\b(politraum|polytrauma|uraz\w* wielonarządow|uraz\w* wielomiejscow|major trauma"
     r"|zespół urazow|zespołu urazow|centrum urazow|trauma team|trauma activation)"):
        "cc_trauma_activation",
    r"\b(nieprzytomn|unresponsive|unconscious|nie reaguje)":
        "cc_unresponsive",
    r"\b(niewydoln\w* oddechow|respiratory distress|ciężk\w* duszno|skrajn\w* duszno)":
        "cc_respiratorydistress",

    # ── Cardiovascular ──
    r"\b(ból w klatce|bólu w klatce|ból klatki|bólu klatki|ból zamostkow|chest pain|zawał|heart attack|stemi|nstemi)":
        "cc_chestpain",
    r"\b(ucisk w klatce|ucisk za mostkiem|ściskani\w* w klatce|gniecenie w klatce|chest tightness|chest pressure)":
        "cc_chesttightness",
    r"\b(kołatani|palpitation)":
        "cc_palpitations",
    r"\b(arytmi|niemiarow|migotani\w* przedsionk|irregular heart|atrial fibrillation)":
        "cc_irregularheartbeat",
    r"\b(tachykardi|tachycardi)":
        "cc_tachycardia",
    r"\b(szybkie bicie serca|serce wali|rapid heart)":
        "cc_rapidheartrate",
    r"\b(omdleni|omdlał|zasłab|syncope|fainted)":
        "cc_syncope",
    r"\b(przedomdl|presyncope|near.syncope)":
        "cc_nearsyncope",
    r"\b(utrat\w* przytomności|stracił\w* przytomność|loss of consciousness)":
        "cc_lossofconsciousness",

    # ── Respiratory ──
    # Yale synonyms: Shortness of Breath 19 702 train visits vs Dyspnea 2 479 (together: 39)
    r"\b(duszn|dyspn|brak tchu|krótki oddech|shortness of breath|short of breath)":
        "cc_shortnessofbreath",
    r"\b(trudnoś\w* w oddychaniu|breathing difficult|difficulty breathing)":
        "cc_breathingdifficulty",
    r"\b(kaszel|kaszl|cough)":
        "cc_cough",
    r"\b(krwioplu|hemoptysis)":
        "cc_hemoptysis",
    r"\b(astm|asthma)":
        "cc_asthma",
    r"\b(świsty|świszcz|wheez)":
        "cc_wheezing",

    # ── Neurological ──
    r"\b(udar|stroke|porażeni\w* połowicz|opadnięt\w* kącik|asymetri\w* twarzy|afazj|bełkotliw|facial droop|slurred speech)":
        "cc_strokealert",
    r"\b(niedowład|paraliż|hemipare|hemipleg)":
        "cc_extremityweakness",
    # Yale synonyms: Altered Mental Status 6 859 train visits vs Confusion 640 (together: 6)
    r"\b(splątan|zdezorientowan|dezorientac|confus|altered mental)":
        "cc_alteredmentalstatus",
    r"\b(drgaw|konwulsj|padaczk|epilep|seizure)":
        "cc_seizures",
    r"\b(ból głowy|bólu głowy|bóle głowy|headache)":
        "cc_headache",
    r"\b(migren|migraine)":
        "cc_migraine",
    r"\b(najsilniejszy ból głowy|najgorszy ból głowy|piorunujący ból głowy|thunderclap|worst headache)":
        "cc_headache_newonsetornewsymptoms",
    r"\b(zawroty głowy|zawrotów głowy|dizz|vertigo)":
        "cc_dizziness",
    r"\b(drętwieni|mrowieni|numbness|tingling)":
        "cc_numbness",
    r"\b(osłabieni|weakness)":
        "cc_weakness",
    r"\b(zmęczeni|wyczerpani|fatigue)":
        "cc_fatigue",

    # ── Abdominal / GI ──
    r"\b(ból brzucha|bólu brzucha|bóle brzucha|brzuch|abdominal pain)":
        "cc_abdominalpain",
    _cooccur(r"ciąż|pregnan", r"brzuch|podbrzusz|abdominal", unless=_NEG_PREGNANCY):
        "cc_abdominalpainpregnant",
    r"\b(nadbrzusz|epigastr|zgag|heartburn)":
        "cc_epigastricpain",
    r"\b(kolk\w* nerkow|okolicy lędźwiowej|flank pain|renal colic)":
        "cc_flankpain",
    r"\b(wymiot|vomit|emesis)":
        "cc_emesis",
    r"\b(nudno|mdłoś|nausea)":
        "cc_nausea",
    r"\b(biegunk|diarrh)":
        "cc_diarrhea",
    r"\b(zaparci|constipat)":
        "cc_constipation",
    r"\b(krwawieni\w* z przewodu|krwotok\w* z przewodu|krew w stolcu|fusowat|smolist|hematemesis|melena|gi bleed)":
        "cc_gibleeding",
    r"\b(krwawieni\w* z odbytu|rectal bleed|hematochez)":
        "cc_rectalbleeding",

    # ── Trauma ──
    # Generic injury / fracture → "Trauma" (no generic fracture complaint in Yale);
    # dropped when a specific injury / fall / crash code fired (REDUNDANT_GENERIC)
    r"\b(uraz|trauma|wypadek|wypadku|accident|złama|fractur)":
        "cc_trauma",
    r"\b(upad|przewróci|fall)":
        "cc_fall",
    r"\b(wielokrotne upadki|częste upadki|multiple falls|recurrent falls)":
        "cc_multiplefalls",
    r"\b(wypad\w* komunikacyjn|wypad\w* samochodow|kolizj|potrąc|motor vehicle|car accident|car crash|mvc\b|mva\b)":
        "cc_motorvehiclecrash",
    r"\b(motocykl|motorcycle)":
        "cc_motorcyclecrash",
    r"\b(pobic|pobit|napaś|assault)":
        "cc_assaultvictim",
    _near(_INJURY + r"|uderz", r"głow|head\b"):            "cc_headinjury",
    _near(_INJURY, r"nog|nodz|nóg|podudzi|goleni|leg\b"):   "cc_leginjury",
    _near(_INJURY, r"ręk|ręce|ramieni|przedrami|arm\b|forearm"): "cc_arminjury",
    _near(_INJURY, r"nadgarst|wrist"):                      "cc_wristinjury",
    _near(_INJURY, r"kostk|staw\w* skokow|ankle"):          "cc_ankleinjury",
    _near(_INJURY, r"kolan|knee"):                          "cc_kneeinjury",
    _near(_INJURY, r"bark|obojczyk|shoulder|clavic"):       "cc_shoulderinjury",
    _near(_INJURY, r"dłoń|dłoni|hand\b"):                   "cc_handinjury",
    _near(_INJURY, r"palc|finger"):                         "cc_fingerinjury",
    _near(_INJURY, r"stop(?:a|y|ie|ę)\b|foot"):             "cc_footinjury",
    _near(_INJURY, r"żeb(?:r|er)|ribs?\b"):                 "cc_ribinjury",
    _near(_INJURY, r"twarz|face|facial"):                   "cc_facialinjury",
    _near(_INJURY, r"ok[aou]\b|oczu|eye"):                  "cc_eyeinjury",
    # "od rana" / "do rana" / "z rana" = "since/until/in the morning", not a wound
    r"\b((?<!od )(?<!do )(?<!z )rana\b|ran[ęy]\b|skaleczeni|rozcięci|laceration|cut\b)":
        "cc_laceration",
    r"\b(oparzeni|burn)":
        "cc_burn",
    r"\b(pogryzi|ugryzi|pogryzł|ugryzł|animal bite|dog bite|bitten)":
        "cc_animalbite",
    r"\b(użądl|ukąszeni\w* owad|insect bite|bee sting)":
        "cc_insectbite",
    r"\b(kleszcz|tick\b)":
        "cc_tickremoval",

    # ── Bleeding ──
    (r"\b(krwawieni\w*\b(?!\s+z\s+(?:przewodu|odbytu|nosa|dróg rodnych|pochwy))"
     r"|krwiak|siniak|zasinieni|bruis|(?<!rectal )(?<!vaginal )(?<!gi )(?<!nose )bleeding)"):
        "cc_bleeding_bruising",
    r"\b(krwawieni\w* z nosa|krwotok\w* z nosa|epistaxis|nosebleed|nose bleed)":
        "cc_epistaxis",
    r"\b(krwawieni\w* z dróg rodnych|krwawieni\w* z pochwy|plamieni|vaginal bleed)":
        "cc_vaginalbleeding",
    r"\b(krwiomocz|krew w moczu|hematuria)":
        "cc_hematuria",

    # ── Infection / general ──
    # Sepsis has no own complaint in Yale — closest real code is "Fever"
    rf"\b({_FEVER}|wysoka temperatura|stan podgorączk)":
        "cc_fever",
    _cooccur(_FEVER, r"immunosupres|chemioterapi|neutropeni|immunocompromis|przeszczep|transplant",
             unless=_NEG_IMMUNOSUPPRESSION):
        "cc_feverimmunocompromised",
    r"\b(dreszcz|chills|rigors)":
        "cc_chills",
    r"\b(odwodni|dehydrat)":
        "cc_dehydration",
    r"\b(gryp|influenza)":
        "cc_influenza",
    r"\b(przeziębi|common cold|cold symptoms)":
        "cc_coldlikesymptoms",
    r"\b(ból gardła|bólu gardła|sore throat)":
        "cc_sorethroat",
    r"\b(ból ucha|bólu ucha|ear pain|earache)":
        "cc_earpain",
    r"\b(ból zęba|bólu zęba|ból zębów|toothache|dental pain)":
        "cc_dentalpain",
    r"\b(ropień|ropnia|abscess)":
        "cc_abscess",
    r"\b(zapaleni\w* tkanki łącznej|cellulitis)":
        "cc_cellulitis",

    # ── Psychiatric / toxicology ──
    r"\b(samobój|suicid)":
        "cc_suicidal",
    r"\b(homicid|chce zabić|grozi zabiciem|myśli zabójcz)":
        "cc_homicidal",
    r"\b(psychoz|psychotyczn|psychosis|psychotic)":
        "cc_psychoticsymptoms",
    r"\b(halucynac|omam|hallucinat)":
        "cc_hallucinations",
    r"\b(psychiatr)":
        "cc_psychiatricevaluation",
    r"\b(pobudzon|pobudzeni|agresywn|agresj|agitat)":
        "cc_agitation",
    r"\b(lęk|niepokój|anxiety|anxious)":
        "cc_anxiety",
    r"\b(atak paniki|napad paniki|panic attack)":
        "cc_panicattack",
    r"\b(depresj|depress)":
        "cc_depression",
    r"\b(nietrzeźw|upojeni|pijan|alkohol|alcohol|etanol|drunk)":
        "cc_alcoholintoxication",
    r"\b(odstawieni\w* alkohol|zespół abstynencyjn|zespołu abstynencyjn|delirium tremens|alcohol withdrawal)":
        "cc_withdrawal_alcohol",
    _cooccur(_OVERDOSE, _SUICIDE):
        "cc_overdose_intentional",
    _without(_OVERDOSE, _SUICIDE):
        "cc_overdose_accidental",
    r"\b(narkoty|dopalacz|amfetamin|metamfetamin|kokain|heroin|substancj\w* psychoaktywn)":
        "cc_drugproblem",
    r"\b(zatruci|zatrut|poison|tlenek węgla|czad)":
        "cc_poisoning",

    # ── Skin / allergy ──
    r"\b(wysypk|pokrzywk|rash|hives|urticari)":
        "cc_rash",
    r"\b(reakcj\w* alergiczn|odczyn\w* alergiczn|allergic reaction|anafilak|anaphyla|obrzęk\w* naczynioruchow|angioedema)":
        "cc_allergicreaction",
    rf"\b({_SWELLING})":
        "cc_edema",
    _near(_SWELLING, r"nog|nóg|podudzi|goleni|kończyn\w* doln|leg"):  "cc_legswelling",
    _near(_SWELLING, r"twarz|powiek|face|facial"):                      "cc_facialswelling",
    _near(_SWELLING, r"język|warg|gardł|tongue|lip|throat"):            "cc_oralswelling",

    # ── Metabolic ──
    r"\b(hipoglikemi|niski cukier|hypoglyc)":
        "cc_decreasedbloodsugar_symptomatic",
    r"\b(hiperglikemi|wysoki cukier|hyperglyc|kwasic\w* ketonow|ketoacidosis|dka\b)":
        "cc_hyperglycemia",

    # ── Genitourinary ──
    r"\b(infekcj\w* (?:dróg|układu) moczow|zakażeni\w* (?:dróg|układu) moczow|zum\b|uti\b|urinary tract infection)":
        "cc_urinarytractinfection",
    r"\b(pieczeni\w* przy (?:oddawaniu|mikcji)|bolesne oddawanie moczu|dyzuri|dysuri)":
        "cc_dysuria",
    r"\b(zatrzymani\w* moczu|urinary retention)":
        "cc_urinaryretention",
    r"\b(ból jąd|bólu jąd|testicular pain|testicle pain)":
        "cc_testiclepain",

    # ── Pain by location ──
    r"\b(ból plec|bólu plec|ból kręgosłup|bólu kręgosłup|ból lędźw|rwa kulszow|back pain|lumbago)":
        "cc_backpain",
    r"\b(ból szyi|bólu szyi|ból karku|bólu karku|neck pain)":
        "cc_neckpain",

    # ── Low-acuity visits (ESI 4–5) ──
    r"\b(zdjęci\w* szwów|usunięci\w* szwów|suture removal|staple removal)":
        "cc_suture_stapleremoval",
    r"\b(kontrol\w* rany|zmian\w* opatrunku|wound check)":
        "cc_woundcheck",
}

_FLAGS = re.IGNORECASE | re.UNICODE
_COMPILED_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, _FLAGS), cc) for pattern, cc in KEYWORD_MAP.items()
]


# ── Generic complaint dropped when a specific one fired ──────────────
# "złamanie nogi" matches the generic Trauma rule AND Leg Injury; "obrzęk nóg"
# Edema AND Leg Swelling, "wielokrotne upadki" Fall AND Multiple Falls. In Yale a
# visit with the specific complaint practically never also carries the generic
# one (< 1% of the rarer code in train.parquet — checked by
# tests/test_layer0_parser.py), so emitting both would feed the models a
# combination they never saw. generic → codes that make it redundant; applied to
# the keyword AND the LLM output (the LLM may also return both synonyms).
_TRAUMA_SPECIFIC: frozenset[str] = frozenset(
    f"cc_{c}" for c in KNOWN_CC_CODES
    if c.endswith(("injury", "laceration"))
    or c in {"fall", "fall_65", "multiplefalls", "motorvehiclecrash", "motorcyclecrash",
             "assaultvictim", "trauma_activation"}
)
REDUNDANT_GENERIC: dict[str, frozenset[str]] = {
    "cc_trauma": _TRAUMA_SPECIFIC,
    "cc_fall": frozenset({"cc_fall_65", "cc_multiplefalls"}),
    "cc_edema": frozenset(f"cc_{c}" for c in KNOWN_CC_CODES if c.endswith("swelling")),
    "cc_fever": frozenset(f"cc_{c}" for c in KNOWN_CC_CODES
                          if c.startswith("fever") and c != "fever"),
    "cc_headache": frozenset(f"cc_{c}" for c in KNOWN_CC_CODES
                             if c.startswith("headache") and c != "headache") | {"cc_migraine"},
    "cc_abdominalpain": frozenset({"cc_abdominalpainpregnant"}),
    "cc_bleeding_bruising": frozenset({"cc_epistaxis", "cc_gibleeding", "cc_rectalbleeding",
                                       "cc_vaginalbleeding"}),
    "cc_poisoning": frozenset({"cc_overdose_accidental", "cc_overdose_intentional"}),
    "cc_seizures": frozenset({"cc_seizure_newonset", "cc_seizure_priorhxof"}),
    # "ciężka duszność" also matches the shortness-of-breath rule (together: 0.7% of
    # Respiratory Distress visits); "near syncope" contains "syncope" (together: 0)
    "cc_shortnessofbreath": frozenset({"cc_respiratorydistress"}),
    "cc_syncope": frozenset({"cc_nearsyncope"}),
    # Synonyms (the keyword rules emit only the more frequent one; the LLM may return both)
    "cc_dyspnea": frozenset({"cc_shortnessofbreath"}),
    "cc_confusion": frozenset({"cc_alteredmentalstatus"}),
}


def drop_redundant_generic(codes: Iterable[str]) -> set[str]:
    """Remove generic complaints made redundant by a specific one (REDUNDANT_GENERIC)."""
    found = set(codes)
    return {c for c in found if not (REDUNDANT_GENERIC.get(c, frozenset()) & found)}


# Pain score: extract first 0-10 number near "ból" or "pain"
# (informational only — pain is not a feature of the current triage models)
_PAIN_RE = re.compile(r"(?:ból|pain)\s+(\d{1,2})\s*/?\s*10?", re.IGNORECASE)


# ── LLM prompt + schema ───────────────────────────────────────────────

_PARSER_SYSTEM = """Jesteś precyzyjnym parserem notatek triażowych SOR.
Czytasz polski (lub angielski) tekst pielęgniarki i wyodrębniasz skargi główne
pacjenta (chief complaints) do triażu ESI.

Reguły:
- chief_complaints: lista kodów ANGIELSKICH WYŁĄCZNIE z dostarczonej listy (np. "chestpain", "dyspnea")
- pain_score: 0-10 jeśli wspomniana, w przeciwnym razie null
- altered_mental_status: true/false
- urgency_signals: krótkie hasła (np. "hipotensja", "tachykardia")

Zwróć WYŁĄCZNIE poprawny JSON zgodny z schematem. Bez komentarzy.
"""

_PARSER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "chief_complaints": {
            "type": "array",
            "items": {"type": "string", "enum": KNOWN_CC_CODES},
        },
        "pain_score":           {"type": ["integer", "null"], "minimum": 0, "maximum": 10},
        "altered_mental_status":{"type": "boolean"},
        "urgency_signals":      {"type": "array", "items": {"type": "string"}},
    },
    "required": ["chief_complaints", "altered_mental_status"],
}


# ── Public API ────────────────────────────────────────────────────────

def normalize_cc_code(code: str) -> str | None:
    """'Chest Pain' / 'cc_chestpain' → 'cc_chestpain'; None if not a model feature."""
    key = str(code).strip().lower().replace(" ", "")
    if key.startswith("cc_"):
        key = key[3:]
    key = LEGACY_CC_CODES.get(key, key)
    return f"cc_{key}" if key in _KNOWN_CC_SET else None


def parse_with_keywords(note: str) -> dict:
    """Deterministic fallback parser — no LLM required."""
    cc_features: dict[str, int] = {}
    keywords: list[str] = []
    extra: dict[str, float] = {}

    if note and note.strip():
        note_lc = note.lower()
        matched: set[str] = set()
        for pattern, cc in _COMPILED_KEYWORDS:
            hit = pattern.search(note_lc)
            if hit:
                matched.add(cc)
                # Zero-width rules (co-occurrence) have no matched text → use the code
                text = hit.group(0).strip()
                keywords.append(text if text else cc[3:])
        cc_features = {cc: 1 for cc in sorted(drop_redundant_generic(matched))}

        pain_match = _PAIN_RE.search(note_lc)
        if pain_match:
            try:
                extra["triage_pain"] = float(min(10, int(pain_match.group(1))))
            except ValueError:
                pass

    log.debug("[keyword] parsed %d CC flags", len(cc_features))
    return {"cc": cc_features, "extra": extra, "keywords": keywords, "source": "keyword"}


async def parse_with_llm(
    note: str,
    client: OllamaClient,
    model: str,
) -> dict:
    """Use Ollama / Llama to parse the note. Raises OllamaError on failure."""
    if not note or not note.strip():
        return {"cc": {}, "extra": {}, "keywords": [], "source": "llm-empty"}

    parsed = await client.chat_json(
        model=model,
        system=_PARSER_SYSTEM,
        user=f"Notatka triażowa: {note}",
        schema=_PARSER_SCHEMA,
    )

    complaints = list(parsed.get("chief_complaints") or [])
    if parsed.get("altered_mental_status"):
        complaints.append("alteredmentalstatus")

    # The LLM may ignore the enum — keep only real model features
    codes: set[str] = set()
    for raw in complaints:
        code = normalize_cc_code(raw)
        if code is None:
            log.debug("[llm] dropping unknown chief complaint %r", raw)
            continue
        codes.add(code)
    cc_features = {cc: 1 for cc in sorted(drop_redundant_generic(codes))}

    extra: dict[str, float] = {}
    pain = parsed.get("pain_score")
    if pain is not None:
        try:
            extra["triage_pain"] = float(max(0, min(10, int(pain))))
        except (TypeError, ValueError):
            pass

    log.info("[llm] parsed %d CC flags via %s", len(cc_features), model)
    return {
        "cc": cc_features,
        "extra": extra,
        "keywords": parsed.get("urgency_signals", []),
        "source": "llm",
    }


async def parse_clinical_note_async(
    note: str,
    client: OllamaClient | None = None,
    model: str | None = None,
) -> dict:
    """Async entry point — uses Ollama if available, else falls back to keywords."""
    if client is not None and model:
        try:
            if await client.health():
                return await parse_with_llm(note, client, model)
        except OllamaError as exc:
            log.warning("LLM parser failed (%s) — falling back to keyword matcher", exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM parser unexpected error: %s — fallback to keyword", exc)
    return parse_with_keywords(note)


# Backwards-compatible sync API used by older callers
def parse_clinical_note(note: str) -> dict:
    return parse_with_keywords(note)
