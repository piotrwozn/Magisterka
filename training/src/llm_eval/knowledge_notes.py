"""
Mała, kuratorowana baza wiedzy dla części "tools" (zastępstwo, dopóki nie powstanie właściwy RAG).

    ESI_DECISION_POINTS_PL — własna, skrócona parafraza punktów decyzyjnych A–D algorytmu ESI v4
    ESI_LEVELS_PL          — krótki opis poziomów ESI 1–5 (kolory TOPSOR tylko do wyświetlania)
    esi_criteria(level)    — tekst dla narzędzia esi_criteria
    KNOWLEDGE_NOTES        — 15 krótkich notatek (id W01–W15), własny tekst, bez cytatów
    search_knowledge(q)    — proste wyszukiwanie po rdzeniach słów (bez zależności zewnętrznych)

Treści są autorskim streszczeniem powszechnie znanych zasad triażu i skal wczesnego ostrzegania —
NIE są kopią podręcznika ESI ani wytycznych. Służą wyłącznie do testu korzystania z narzędzi.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.llm_eval.cc_labels_pl import fold_text
from src.utils.config import TOPSOR_COLORS, TOPSOR_COLORS_PL, TOPSOR_MAX_WAIT_MINUTES

# ─────────────────────────────────────────
# Punkty decyzyjne ESI (parafraza własna)
# ─────────────────────────────────────────
ESI_DECISION_POINTS_PL: dict[str, str] = {
    "A": ("Punkt A — czy pacjent potrzebuje natychmiastowej interwencji ratującej życie (np. udrożnienia "
          "dróg oddechowych, wspomagania oddechu, defibrylacji, pilnych leków lub płynów w stanie wstrząsu)? "
          "Jeśli tak → ESI 1. Typowe sytuacje: zatrzymanie krążenia, bezdech lub oddech niewydolny, brak "
          "reakcji na bodźce, objawy wstrząsu."),
    "B": ("Punkt B — czy pacjent nie powinien czekać: sytuacja wysokiego ryzyka, nowe zaburzenia "
          "świadomości (splątanie, senność, dezorientacja) albo silny ból lub cierpienie? Jeśli tak → ESI 2. "
          "Przykłady: ból w klatce piersiowej podejrzany o zespół wieńcowy, objawy udaru, myśli samobójcze, "
          "hipotensja."),
    "C": ("Punkt C — ile różnych rodzajów zasobów będzie prawdopodobnie potrzebnych (badania laboratoryjne, "
          "obrazowe, leki lub płyny dożylne, konsultacja specjalistyczna, prosty zabieg)? Dwa lub więcej → "
          "sprawdź punkt D, a potem ESI 3; jeden → ESI 4; żaden → ESI 5. Badanie fizykalne, leki doustne, "
          "recepta czy prosty opatrunek nie są liczone jako zasoby."),
    "D": ("Punkt D — dotyczy pacjenta kwalifikowanego do ESI 3: czy parametry życiowe są w strefie zagrożenia "
          "(dorośli: tętno > 100/min, oddechy > 20/min, SpO2 < 92%)? Jeśli tak → rozważ podniesienie do ESI 2; "
          "ostateczna decyzja należy do osoby triażującej."),
}

_LEVEL_DESC: dict[int, str] = {
    1: "stan bezpośredniego zagrożenia życia — potrzebna natychmiastowa interwencja ratująca życie (punkt A)",
    2: "sytuacja wysokiego ryzyka, nowe zaburzenia świadomości albo silny ból / cierpienie; także "
       "pacjent z ESI 3 z parametrami w strefie zagrożenia, jeśli triażysta tak zdecyduje (punkty B i D)",
    3: "stan stabilny, ale przewidywane co najmniej dwa rodzaje zasobów (punkt C)",
    4: "stan stabilny, przewidywany jeden rodzaj zasobu (punkt C)",
    5: "stan stabilny, nie są przewidywane żadne zasoby poza badaniem i prostą pomocą (punkt C)",
}
_LEVEL_POINTS: dict[int, tuple[str, ...]] = {1: ("A",), 2: ("B", "D"), 3: ("C", "D"), 4: ("C",), 5: ("C",)}


def _level_line(level: int) -> str:
    color = TOPSOR_COLORS[level - 1]
    return (f"ESI {level} (kolor {TOPSOR_COLORS_PL[level - 1]}, kontakt z lekarzem do "
            f"{TOPSOR_MAX_WAIT_MINUTES[color]} min wg TOPSOR): {_LEVEL_DESC[level]}.")


ESI_LEVELS_PL: dict[int, str] = {level: _level_line(level) for level in range(1, 6)}


def esi_criteria(level: int) -> dict[str, object]:
    """Kryteria poziomu ESI 1–5: opis poziomu + właściwe punkty decyzyjne (parafraza własna)."""
    level = int(level)
    if level not in ESI_LEVELS_PL:
        raise ValueError(f"Poziom ESI musi być liczbą 1–5, podano {level}")
    return {
        "poziom_esi": level,
        "opis": ESI_LEVELS_PL[level],
        "punkty_decyzyjne": {p: ESI_DECISION_POINTS_PL[p] for p in _LEVEL_POINTS[level]},
        "uwaga": "Kolory TOPSOR służą wyłącznie do wyświetlania; etykietą jest poziom ESI.",
    }


# ─────────────────────────────────────────
# Notatki
# ─────────────────────────────────────────
@dataclass(frozen=True)
class Note:
    id: str
    title: str
    tags: tuple[str, ...]
    text: str


KNOWLEDGE_NOTES: tuple[Note, ...] = (
    Note("W01", "Strefa zagrożenia parametrów życiowych u dorosłych",
         ("strefa zagrożenia", "tętno", "oddechy", "saturacja", "punkt D", "ESI 2"),
         "U dorosłego pacjenta tętno powyżej 100/min, częstość oddechów powyżej 20/min lub SpO2 poniżej 92% "
         "to sygnał, by pacjenta kwalifikowanego do ESI 3 rozważyć jako ESI 2. To wskazówka, nie automat: "
         "liczy się obraz całości (np. ból, gorączka, lęk przyspieszają tętno)."),
    Note("W02", "Wskaźnik wstrząsowy",
         ("wskaźnik wstrząsowy", "shock index", "tętno", "ciśnienie", "wstrząs", "krwawienie"),
         "Wskaźnik wstrząsowy to iloraz tętna i ciśnienia skurczowego. Wartości około 0,5–0,7 są typowe; "
         "wartość 1,0 lub wyższa sugeruje istotną niewydolność krążenia (np. krwawienie, sepsę) nawet wtedy, "
         "gdy samo ciśnienie wygląda jeszcze akceptowalnie."),
    Note("W03", "Skala NEWS2",
         ("NEWS2", "skala wczesnego ostrzegania", "punktacja", "pogorszenie"),
         "NEWS2 sumuje punkty za częstość oddechów, SpO2, tlenoterapię, ciśnienie skurczowe, tętno, "
         "świadomość i temperaturę. Wynik 5–6 wymaga pilnej oceny lekarskiej, 7 i więcej — oceny natychmiastowej. "
         "W naszych danych brakuje oceny świadomości, więc wynik jest zaniżony o ten składnik."),
    Note("W04", "qSOFA i podejrzenie sepsy",
         ("qSOFA", "sepsa", "zakażenie", "oddechy", "ciśnienie", "świadomość"),
         "qSOFA przyznaje po punkcie za częstość oddechów co najmniej 22/min, ciśnienie skurczowe najwyżej "
         "100 mmHg i zaburzenia świadomości. Dwa punkty u pacjenta z podejrzeniem zakażenia oznaczają wyższe "
         "ryzyko złego przebiegu; niski wynik nie wyklucza sepsy."),
    Note("W05", "Kryteria SIRS",
         ("SIRS", "gorączka", "tętno", "oddechy", "zakażenie"),
         "SIRS opiera się na temperaturze poza zakresem 36–38 °C, tętnie powyżej 90/min, oddechach powyżej "
         "20/min oraz leukocytozie. Bez badań krwi liczymy tylko trzy pierwsze składowe. Dwa kryteria to "
         "uogólniona odpowiedź zapalna, mało swoista, ale przydatna jako sygnał ostrzegawczy."),
    Note("W06", "Saturacja i tlenoterapia",
         ("saturacja", "SpO2", "tlen", "tlenoterapia", "hipoksemia", "duszność"),
         "SpO2 poniżej 94% oznacza hipoksemię, a poniżej 90% — ciężką. Pomiar wykonany u pacjenta już "
         "otrzymującego tlen zaniża obraz problemu: prawidłowa saturacja na tlenie nie znaczy, że pacjent "
         "oddycha wydolnie. U chorych z przewlekłą niewydolnością oddechową docelowe wartości bywają niższe."),
    Note("W07", "Ból w klatce piersiowej",
         ("ból w klatce piersiowej", "zawał", "zespół wieńcowy", "EKG", "wysokie ryzyko"),
         "Ból w klatce piersiowej o charakterze mogącym wskazywać na ostry zespół wieńcowy, rozwarstwienie "
         "aorty lub zatorowość płucną to sytuacja wysokiego ryzyka (zwykle ESI 2). EKG powinno być wykonane "
         "i ocenione w ciągu kilku minut od przybycia."),
    Note("W08", "Objawy udaru",
         ("udar", "alert udarowy", "niedowład", "zaburzenia mowy", "neurologia", "czas"),
         "Nagły niedowład, zaburzenia mowy, asymetria twarzy czy zaburzenia widzenia sugerują udar. Liczy się "
         "czas od początku objawów — pacjent z podejrzeniem udaru w oknie terapeutycznym wymaga natychmiastowej "
         "ścieżki udarowej (co najmniej ESI 2)."),
    Note("W09", "Pacjent w podeszłym wieku",
         ("wiek", "starszy", "upadek", "nietypowe objawy", "hipotensja"),
         "U osób starszych ciężkie choroby często przebiegają nietypowo: bez gorączki, z osłabieniem, "
         "splątaniem lub upadkiem. Ciśnienie skurczowe w normie może być względną hipotensją u chorego "
         "z nadciśnieniem. Upadek u seniora wymaga oceny przyczyny i urazów (zwłaszcza głowy przy lekach "
         "przeciwkrzepliwych)."),
    Note("W10", "Myśli samobójcze i zagrożenie dla innych",
         ("myśli samobójcze", "samobójstwo", "psychiatria", "bezpieczeństwo", "przedawkowanie"),
         "Pacjent z myślami samobójczymi, po celowym przedawkowaniu lub stanowiący zagrożenie dla innych to "
         "sytuacja wysokiego ryzyka (ESI 2): wymaga bezpiecznego miejsca, nadzoru i szybkiej oceny, nawet "
         "przy prawidłowych parametrach życiowych."),
    Note("W11", "Gorączka u pacjenta z obniżoną odpornością",
         ("gorączka", "neutropenia", "chemioterapia", "odporność", "zakażenie"),
         "Gorączka u chorego po chemioterapii, po przeszczepie lub przyjmującego leki immunosupresyjne może "
         "oznaczać szybko postępujące zakażenie. Taki pacjent nie powinien czekać — zwykle traktowany jest "
         "jako sytuacja wysokiego ryzyka."),
    Note("W12", "Zasoby w ESI",
         ("zasoby", "ESI 3", "ESI 4", "ESI 5", "badania", "punkt C"),
         "Zasoby w ESI to m.in. badania laboratoryjne, EKG i obrazowe, leki lub płyny dożylne, konsultacja "
         "specjalisty i proste zabiegi (np. szycie rany). Nie są nimi: badanie fizykalne, lek doustny, "
         "recepta, prosty opatrunek czy temblak. Liczy się liczba RODZAJÓW zasobów, nie liczba badań."),
    Note("W13", "Kolory TOPSOR",
         ("TOPSOR", "kolory", "czas oczekiwania", "czerwony", "pomarańczowy"),
         "W polskim systemie TOPSOR poziomom ESI odpowiadają kolory: ESI 1 czerwony (natychmiast), ESI 2 "
         "pomarańczowy (do 10 min), ESI 3 żółty (do 60 min), ESI 4 zielony (do 120 min), ESI 5 niebieski "
         "(do 240 min). Kolor służy do wyświetlania — etykietą jest poziom ESI."),
    Note("W14", "Ograniczenia systemu wspomagania triażu",
         ("ograniczenia", "model", "dane", "niedotriażowanie", "odpowiedzialność", "pomocniczy"),
         "Model uczono na danych z oddziałów ratunkowych w USA, a decyzja to wynik statystyczny — narzędzie "
         "pomocnicze, nie zastępuje oceny osoby triażującej. Najgroźniejszym błędem jest niedotriażowanie "
         "pacjentów ESI 1–2, dlatego wątpliwości rozstrzyga się na korzyść wyższego priorytetu."),
    Note("W15", "Brak pomiaru to nie wynik prawidłowy",
         ("brak pomiaru", "braki", "parametry", "niekompletne dane"),
         "Jeśli parametru nie zmierzono, nie wolno zakładać, że jest prawidłowy. U najciężej chorych pomiary "
         "często są pomijane, bo pacjent od razu trafia na salę resuscytacyjną. Brak danych należy nazwać "
         "wprost i w razie wątpliwości uzupełnić pomiar."),
)
NOTES_BY_ID: dict[str, Note] = {n.id: n for n in KNOWLEDGE_NOTES}


def _stems(text: str) -> list[str]:
    """Rdzenie słów (≥ 3 znaki, po `fold_text`, maks. 5 znaków) do dopasowania odmian."""
    return [w[:5] for w in re.findall(r"[a-z0-9]+", fold_text(text)) if len(w) >= 3]


def search_knowledge(query: str, top_k: int = 3) -> list[dict[str, str]]:
    """
    Notatki pasujące do zapytania: punkty za wspólne rdzenie słów (tytuł × 3, tagi × 2, treść × 1).
    Zwraca do `top_k` notatek z wynikiem > 0 (malejąco, przy remisie wg id).
    """
    wanted = set(_stems(query))
    if not wanted:
        return []
    scored = []
    for note in KNOWLEDGE_NOTES:
        title, tags, text = set(_stems(note.title)), set(_stems(" ".join(note.tags))), set(_stems(note.text))
        score = 3 * len(wanted & title) + 2 * len(wanted & tags) + len(wanted & text)
        if score > 0:
            scored.append((-score, note.id, note))
    scored.sort()
    return [{"id": n.id, "tytul": n.title, "tresc": n.text} for _, _, n in scored[:top_k]]


__all__ = [
    "ESI_DECISION_POINTS_PL",
    "ESI_LEVELS_PL",
    "KNOWLEDGE_NOTES",
    "NOTES_BY_ID",
    "Note",
    "esi_criteria",
    "search_knowledge",
]
