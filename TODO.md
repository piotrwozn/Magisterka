# SOR-AI: Kierunki dalszego rozwoju i tematyka badawcza (Ph.D.)

Poniższe punkty stanowią propozycje fundamentalnej rozbudowy obecnego systemu, przekształcającej projekt inżynieryjny (praca magisterska) w zaawansowane środowisko badawcze o potencjale doktoranckim.

## 1. Rozszerzenie źródeł wiedzy (Integracja NLP)
Obecny system, choć wysoce precyzyjny w analizie parametrów liczbowych i z góry ustalonych dolegliwości, traci niuanse zawarte w opisie "wolnego tekstu" z wywiadu z pacjentem. Naturalnym rozwinięciem będzie integracja małych, lokalnych medycznych modeli językowych (np. BioClinicalBERT) potrafiących wyłapywać tzw. "czerwone flagi" bezpośrednio z surowych notatek pielęgniarskich. Wyekstrahowane cechy semantyczne zostaną zintegrowane wewnątrz głównego modelu klasyfikacyjnego (NLP-Tabular Fusion).

## 2. Multimodalna ocena stanu pacjenta
Obecny system analizuje pacjenta w 100% cyfrowo (na podstawie wklepanych tabel). Dalsze badania włączą ocenę sygnałów wprost z ciała poprzez:
* Dodanie konwolucyjnych sieci wizyjnych (CNN) analizujących kamery termowizyjne w poczekalni SOR.
* Rozpoznawanie na żywo asymetrii twarzy (podejrzenie udaru) czy estymację tętna na podstawie fotopletyzmografii twarzy (rPPG).
Będzie to stanowiło krok w stronę Triażu Multimodalnego, dopełniając tradycyjną analizę numeryczną.

## 3. Proaktywna redukcja niepewności (Active Learning)
Aktualnie system pasywnie przyjmuje wszystkie wprowadzone dane i zwraca ocenę punktową. W przyszłości SOR-AI będzie działać jako system dialogowy (Human-in-the-Loop). Jeśli algorytm kwantyfikacji niepewności (Uncertainty Quantification) zauważy, że predykcja waha się na niebezpiecznej granicy pilności (np. między MTS 2 a 3) i brakuje mu pewności diagnozy, system samodzielnie zasugeruje pielęgniarce: *"System wymaga ponownego pomiaru saturacji po podaniu tlenu, aby bezpiecznie sfinalizować priorytet"*. Zmniejszy to ryzyko fałszywych klasyfikacji w przypadkach OOD (Out-of-Distribution).

## 4. Kliniczna walidacja w czasie rzeczywistym ("Shadow Mode Deployment")
Ostatecznym sprawdzianem jakości modeli medycznych jest weryfikacja poza eksperymentami in-silico na wyczyszczonych plikach historycznych (RData/Parquet). Kluczowym krokiem badawczym przed jakimkolwiek wdrożeniem produkcyjnym będzie instalacja obecnego systemu na serwerach w prawdziwym szpitalu w tzw. trybie "Shadow mode". System będzie przetwarzał na bieżąco napływające parametry pacjentów (zbierane w ułamku sekundy i często obarczone ludzkimi pomyłkami pod wpływem ogromnego stresu). Badanie pozwoli wyliczyć rzeczywistą skuteczność, redukcję czasu decyzji triażowych w stosunku do personelu medycznego oraz ocenę zjawiska przesunięcia danych (Concept Drift).
## 5. Moduł Behawioralny i Detekcja Nadużyć Systemowych ("Frequent Flyers")
Oddziały ratunkowe na całym świecie borykają się ze zjawiskiem pacjentów nadużywających opieki nagłej (tzw. "Frequent Flyers" lub pacjentów z zaburzeniami lękowymi o zdrowie). Tacy pacjenci często uczą się zgłaszać wyuczone "czerwone flagi" słowne (np. "duszność", "ból w klatce"), wymuszając na personelu wyższy priorytet triażowy (Overtriage). W przyszłości system zostanie wzbogacony o moduł analizy szeregów czasowych z historycznej dokumentacji medycznej (EHR). Integracja modeli detekcji anomalii (np. Isolation Forest) pozwoli na identyfikację "profilu behawioralnego" pacjenta. W połączeniu ze wzorcowymi, prawidłowymi parametrami życiowymi (Vital Signs), system uchroni przed nieuzasadnionym podnoszeniem priorytetów i zatykaniem kolejki krytycznej przez osoby niewymagające opieki nagłej.

## 6. Automatyzacja Bazy Danych EHR (Integracja z Systemami Państwowymi po numerze PESEL)
Obecne modele klasy Triażowej często opierają się wyłącznie na danych wprowadzonych ręcznie przez pacjenta na wejściu (co rodzi problem fałszowania lub zapominania historii chorób, tzw. Information Bias). W perspektywie wdrożeniowej (np. na terytorium Polski), system zostanie zintegrowany poprzez bezpieczne bramki API z P1 (Systemem e-Zdrowie) i platformą IKP (Internetowe Konto Pacjenta). Po odczytaniu numeru PESEL przy okienku rejestracyjnym, system SOR-AI w czasie rzeczywistym i w sposób automatyczny pobierze pełną historię medyczną: listę aktywnych e-recept (ryzyko interakcji lekowych), ostatnie wizyty specjalistyczne oraz historię chorób przewlekłych (kody ICD-10). Dane te zasilą równolegle wejścia tabularyczne (Historical Features) oraz moduł NLP, tworząc w pełni automatyczny, bezobsługowy dla pielęgniarki mechanizm natychmiastowego budowania wielowymiarowego profilu ratunkowego pacjenta.
