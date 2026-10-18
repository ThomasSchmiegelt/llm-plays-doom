# LLM plays Doom

Demonstrator: ein Vision-Language-Modell sieht das Spielbild, erkennt darin per
Textprompt Monster **und** Menschen, und ein Agent zielt nur auf die Monster.
Steht ein Mensch in der Schusslinie, wird das Feuer eingestellt.

* **Sehen** — [NVIDIA LocateAnything-3B](https://research.nvidia.com/labs/lpr/locate-anything/):
  Open-Vocabulary-Detection, Eingabe Bild + Text, Ausgabe Bounding Boxes.
  Ein einziger Prompt enthaelt beide Gruppen; die Trennung passiert allein
  ueber das Label, das das Modell zurueckgibt.
* **Spielen** — [ViZDoom](https://vizdoom.farama.org/) mit den mitgelieferten
  Freedoom-WADs. Das Spiel laeuft nur weiter, wenn der Agent handelt, steht
  also waehrend der Inferenz still.
* **Handeln** — deterministische Geometrie: Box-Mitte gegen Fadenkreuz,
  Drehwinkel aus dem Blickfeld, Schuss ab 14 px Abweichung.

## Videos

**[Aussehen lügt — wie der Agent Unbekanntes einordnet](https://youtube.com/shorts/H2lsb3mzzgw)** *(Short, 52 s)*
Ein unbekanntes Wesen taucht auf. Der Detektor rät nach dem Aussehen — und wird
vom Verhalten überstimmt. Im Panel steht jeweils, woher das Urteil kommt:
*GEFÄHRLICH (Aussehen)* ist eine Vermutung, *(Verhalten)* ist endgültig.

**[Derselbe Lauf, ungeschnitten](https://youtu.be/FkpW8xI2Nhs)** *(46 s)*
Ohne Vor- und Abspann.

**[Drei Regelwerke, ein Doom-Agent: was Vorsicht kostet](https://youtu.be/WlqqHI5g-y4)** *(21 s, Breitbild)*
Drei Agenten nebeneinander, derselbe Seed, dieselbe Arena — und alle drei sehen
exakt dasselbe. Nur die Regel unterscheidet sie: *wild* schießt auf alles,
*streng* hält an, bis Unbekanntes eingeordnet ist, *gewichtet* beobachtet
nebenher. Nach 160 Schritten: 57 Schuss und elfmal auf einen Menschen gefeuert
gegen 4 Schuss und kein einziges Mal — der eine handlungsunfähig vor Vorsicht,
der andere blind.

> Einschränkung, die man dem Bild nicht ansieht: bei allen dreien steht am Ende
> „Monster am Leben: 0", aber die gesicherten haben **null eigene Treffer**.
> Geräumt hat die Arena ein befreundeter Marine mit BFG. Der Vergleich misst
> Sicherheit sauber und Wirksamkeit gar nicht.

Farbcode in beiden: **grün** Monster erkannt · **rot** aktuelles Ziel ·
**blau** Mensch, nicht schießen · **violett** unbekannt, wird beobachtet.

Die Videodateien liegen nicht im Repository (`out/` ist ausgeschlossen), sind
aber jederzeit reproduzierbar:

```bash
src/main.py --unknowns 7 --smooth --record-vertical out/lauf.mp4
src/make_social.py --source out/lauf.mp4 --out out/kurz.mp4 --duration 52 --max-hold 4
src/compare_ammo.py          # drei Regelwerke nebeneinander, auch im Breitbild
```

## Start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python src/main.py
```

Beim ersten Start laedt LocateAnything-3B (~7 GB) aus dem HuggingFace-Cache.

## Das Szenario

`arena_mixed` stellt in einer offenen Doom-Arena einen Ring aus Monstern und
Menschen um den Agenten:

| | Sprites | Rolle |
|---|---|---|
| Monster | Imp, Demon, Lost Soul, Cacodemon | duerfen beschossen werden |
| Menschen | Marines (Helm, sichtbares Gesicht) | duerfen **nicht** beschossen werden |

Die Marines werden als `summonfriend` gesetzt: sie greifen den Agenten nicht
an. Wer schiesst, entscheidet also allein die Wahrnehmung.

## Was man sieht

Ein Fenster **Agent Vision** zeigt links das Spielbild mit den Erkennungen und
rechts den Zustand des Agenten:

* **gruen** — als feindlich erkannt
* **rot** — das aktuell anvisierte Ziel, mit Linie zum Fadenkreuz
* **blau** — als Mensch erkannt, "nicht schiessen"
* **blau gestrichelt** — Mensch aus dem Kurzzeitgedaechtnis (in diesem Frame
  nicht erkannt, gilt aber weiter als geschuetzt)
* **blauer Streifen** — der gesperrte Schusskorridor
* **cyan gestrichelt** (Taste `g`) — Ground Truth aus der Doom-Engine

Mit `--game-window` kommt zusaetzlich das rohe ViZDoom-Fenster dazu.

| Taste | Wirkung |
|---|---|
| `q` | beenden |
| `g` | Engine-Ground-Truth ein/aus |
| `space` | Pause |
| `r` | Episode neu starten |

## Wichtige Optionen

```bash
src/main.py --monsters 8 --humans 5 --waves 5   # Nachschub fuer lange Laeufe
src/main.py --god --steps 700 --record out/lauf.mp4   # Aufzeichnung
src/main.py --classes ... --protected ...       # Prompt und Schutzklassen
src/main.py --scenario defend_the_center        # klassische Arena ohne Menschen
src/main.py --fake-detector                     # ohne GPU, Boxen aus der Engine
src/main.py --decoding hybrid                   # Dekodierung des Modellcodes statt guided
src/main.py --detector models/distill/weights/best.pt   # YOLO-Schueler, Echtzeit
```

## Echtzeit: Dekodierung und Lehrer-Schueler

Ausfuehrlich mit Bildern in [`posts/artikel_echtzeit.md`](posts/artikel_echtzeit.md).

* **Gefuehrte Dekodierung** (`guided`, jetzt Default). Der Hybridmodus von
  LocateAnything wiederholt auf Doom-Frames Boxen bis zum Token-Limit; in 8
  von 24 Testbildern kam er nie bis zur Menschen-Klasse. `guided` gibt die
  Klassen vor und entscheidet ueber einen Schiedsrichter aus derselben
  Vorwaertsrechnung, ob noch eine Box folgt: 1575 -> 486 ms je Bild,
  Menschen erkannt 57 % -> 86 %.
* **LocateAnything als Lehrer.** `src/distill.py` laesst LocateAnything 2.000
  Bilder beschriften und trainiert darauf ein YOLO11s - ohne Engine-Labels,
  ohne die Seeds der Messung. 9 ms je Bild; auf 100 Testbildern 64 % Monster
  und 79 % Menschen erkannt, gegenueber 41 % und 72 % beim Lehrer.
* **YOLO11 statt YOLO26.** Auf denselben Daten ist YOLO26 auf Testbildern
  vorsichtiger, verschiesst im Spiel aber ein Drittel mehr Munition fuer
  dieselben Treffer (Trefferquote 55 % gegen 71 %).

`src/compare_decoding.py` vergleicht Detektoren nebeneinander in echter
Echtzeit: Waehrend der Inferenz laeuft das Spiel weiter. Treffer zaehlen ab
Episodenstart, der Weckschuss aus `env.wake()` ist herausgerechnet.

## Dritte Stufe: Wesen, die der Agent nicht kennt

Im Szenario tauchen Aktoren auf, für die es keine Beschreibung im Prompt gibt —
Hell Knights, Fässer, Säulen, Bäume. Der Agent soll merken, dass da etwas ist,
es beobachten und einordnen.

**Erkennen.** Der Prompt enthält eine absichtlich weite Klasse
(`any creature or figure`). Eine Box dieser Klasse, die keine Box einer
spezifischen Klasse überlappt, ist ein Unbekanntes. Die Engine-Labels dürfen
dafür nicht herangezogen werden — sonst misst sich die Demo selbst.

**Beobachten.** LocateAnything beschriftet nicht, es lokalisiert. Als
Klassifikator lässt es sich trotzdem einsetzen: Man bietet ihm mehrere
Beschreibungen an und schaut, auf welche es eine Box an die Stelle des
verfolgten Wesens legt. Ein einzelnes Bild reicht nicht, deshalb wird über ein
Beobachtungsfenster abgestimmt.

Entscheidend ist, die Vergleiche zu **planen** statt zu würfeln. Treten alle
Beschreibungen auf einmal an, gewinnen einzelne unabhängig vom Bildinhalt — eine
Steinsäule wurde sechsmal von sechs als `silver armored mechanical soldier`
gemeldet. Deshalb tritt pro Bild nur ein ausgewogenes Quartett an, und der
Spielplan (`classifier.py:_schedule`) setzt jede Paarung genau einmal an:
vollständige Abdeckung nach 4 statt 12,3 Bildern.

**Lernen.** Die siegreiche Beschreibung wandert in den Prompt. Ab da wird
dieselbe Art sofort erkannt. Der Nebeneffekt ist bezeichnend: danach wird nichts
mehr beobachtet — einmal `man in dark uniform holding a gun` gelernt, ist keine
gepanzerte Figur mehr unbekannt. `--no-learn` schaltet das für Aufnahmen ab.

```bash
src/main.py --unknowns 7 --unknown-every 20 --window 5 --no-learn
```

### Die Grenze

Zwei Einträge im Testfeld sind mit `summonfriend` gesetzt — ein freundlicher
Hell Knight sieht **pixelgleich** aus wie sein feindlicher Zwilling. Rein
visuelle Überwachung kann die beiden nicht trennen und liegt dort in jedem
Hypothesensatz falsch. Umgekehrt sieht `EvilEye` bedrohlich aus und ist
Dekoration. Beides steht so im Ergebnis.

## Wie der Agent Menschen verschont

Die Klassifikation des Modells ist nicht perfekt, deshalb haengt die Sicherheit
nicht an einem einzelnen Frame. Drei Stufen greifen ineinander:

1. **Label** — Ziel wird nur, was das Modell einer feindlichen Klasse zuordnet.
2. **Schusskorridor** — steht ein als Mensch erkanntes Objekt in dem Streifen
   um das Fadenkreuz, in dem die Schuesse einschlagen, wird nicht gefeuert,
   auch wenn dahinter ein gueltiges Ziel steht.
3. **Kurzzeitgedaechtnis** — jede Menschen-Box bleibt vier Schritte gueltig.
   Weil der ausgefuehrte Drehwinkel bekannt ist, laesst sich die Verschiebung
   im Bild exakt herausrechnen: eine Drehung nach rechts schiebt alles nach
   links. So ueberbrueckt der Schutz einen Aussetzer des Detektors.

Dazu zwei Plausibilitaetsfilter: die eigene Waffe am unteren Bildrand wird
verworfen, und Menschen-Boxen, die breiter als hoch sind, ebenfalls — sonst
blockieren flache Falscherkennungen entlang der Horizontlinie den Korridor
dauerhaft.

## Messung

Die Doom-Engine liefert ueber den Labels-Buffer die exakten Bildschirmboxen
aller Akteure samt Typ. Beide Messwerkzeuge vergleichen dagegen.

```bash
.venv/bin/python src/eval_detector.py    # Erkennungsqualitaet je Typ
.venv/bin/python src/eval_policy.py      # Wirkung der Sicherungen
```

### Erkennung: beschreibende Prompts schlagen Kategorien

Gemessen auf `arena_mixed`, 640x480, RTX 3090, IoU >= 0.35. Die letzte Spalte
ist die gefaehrliche: ein Mensch, den das Modell fuer ein Monster haelt.

| Prompt-Klassen | Monster | Mensch | Mensch→Monster | Latenz |
|---|---|---|---|---|
| brown insect monster, pink fleshy worm creature, orange flying skull, blue shelled monster, grey armored human soldier | 71 % | 69 % | **6 %** | 1565 ms |
| brown insect monster, pink fleshy creature, floating skull, soldier in grey armor | 83 % | 56 % | 12 % | 1654 ms |
| monster, demon, imp, human, soldier | 86 % | 19 % | 56 % | 2065 ms |
| monster, demon, alien creature, human soldier, man with helmet | 71 % | 38 % | 38 % | 2086 ms |

Der Unterschied ist der Kern des Ganzen: **"brown insect monster" trennt
zuverlaessiger als "imp"**. Semantische Kategorienamen fuehren dazu, dass das
Modell mehr als die Haelfte der Marines fuer Monster haelt; Formulierungen, die
das Sprite beschreiben, druecken das auf 6 %. Die erste Zeile ist der Default.

### Was die Sicherungen bringen

`src/eval_policy.py`, 150 Schritte je Zeile, gleiche Startaufstellung (Seed 21).
Eigenbeschuss heisst: Schuss abgegeben, waehrend laut Engine ein Mensch im
Schusskorridor stand.

| Sicherung | Schuesse | gesperrt | Eigenbeschuss | Monster erlegt |
|---|---|---|---|---|
| ohne (schiesst auf alles) | 95 | 0 | **19** | 7/7 |
| Label + Schusskorridor | 10 | 60 | 1 | 7/7 |
| + Kurzzeitgedaechtnis | 12 | 22 | **0** | 7/7 |

Der Preis ist Vorsicht, nicht Wirksamkeit: alle drei Varianten raeumen die
Arena vollstaendig leer, die abgesicherten brauchen nur laenger dafuer. Das
Gedaechtnis reduziert ausserdem die Zahl der Sperren von 60 auf 22 — es
blockiert gezielter, weil es nicht auf jedem Frame neu raten muss.

### Zwei Fallstricke des Modells

* **Mindestens drei Klassen in den Prompt.** Bei ein oder zwei Klassen geraet
  das Modell auf Doom-Frames in eine Wiederholungsschleife und gibt dieselbe
  Box bis zum Token-Limit aus — Faktor 6 bei der Latenz, ohne besseres
  Ergebnis. `max_new_tokens=96` deckelt den Schaden.
* **Kleinere Aufloesung bringt nichts.** 320x240 kostet Recall und ist kaum
  schneller, weil das Modell intern ohnehin skaliert.

Ausserdem verlangt `model.generate()` zwingend `use_cache=True` — im Beispiel
der Modellkarte fehlt das Argument, und der Aufruf bricht ohne es ab.

## Video fuer Shorts und Reels

`--record-vertical` schreibt zusaetzlich eine Fassung in 1080x1920, mit
groesseren Beschriftungen und allem Wesentlichen oberhalb der Zone, die
Instagram und YouTube mit ihrer eigenen Bedienoberflaeche ueberdecken.

```bash
.venv/bin/python src/main.py --god --steps 700 --waves 5 \
    --record out/lauf.mp4 --record-vertical out/raw_vertical.mp4

.venv/bin/python src/make_social.py --source out/raw_vertical.mp4 \
    --out out/doom_short.mp4 --duration 55 \
    --monsters 6 --humans 5 --holds 107
```

`make_social.py` setzt Vor- und Abspann davor und dahinter, streckt den
Hauptteil auf die Zielaenge und kodiert nach H.264/yuv420p, 30 fps, mit
`+faststart` und einer stummen Tonspur — ohne Audio verweigern manche Uploads
die Annahme. 55 s passt in beide Formate (Shorts <= 60 s, Reels <= 90 s).
Mit `--nvenc` laeuft die Kodierung auf der GPU.

## Ergebnisse werden archiviert

Von den Messungen einer ganzen Arbeitssitzung lag am Ende nur eine noch auf der
Platte, weil jeder Lauf die Logdatei des vorigen überschrieb. Deshalb schreibt
jedes Messwerkzeug sein Ergebnis jetzt nach `results/` — mit Parametern,
Umgebung und einem **Fingerabdruck des Codes**, der es erzeugt hat. Ohne den
lässt sich später nicht entscheiden, ob zwei Zahlen vergleichbar sind.

```bash
.venv/bin/python src/results_show.py            # alle Läufe
.venv/bin/python src/results_show.py --diff 3 7 # zwei vergleichen
```

Nachgetragene Zahlen sind als solche markiert und bekommen bewusst *keinen*
Codestand — der heutige hat sie nicht erzeugt.

## Was gemessen wurde und nicht funktioniert hat

Drei Sackgassen, die im Archiv stehen, weil sie mehr erklären als die Erfolge.

**Der Bewegungskanal.** Deko steht still, Kreaturen animieren sich — der
Pixelvergleich innerhalb der verfolgten Box sollte das trennen. Gemessen kam das
Gegenteil heraus: ein stillstehendes Fass „bewegte" sich stärker als ein
Monster. Gemessen wurde Boxinstabilität mal Texturdetail. Verworfen wurde der
Kanal dann aber durch ein Argument: **ein stehender Fernkämpfer bewegt sich
nicht.** Bewegung trennt belebt von unbelebt, nicht gefährlich von harmlos.

**Der Projektil-Kanal.** Ein Feuerball ist ein helles, eigenständiges Sprite —
also müsste eine Hypothesenklasse `a glowing fireball` den Angriff melden, bevor
er trifft. Vier Formulierungen gemessen, alle mit **höherer Fehlalarm- als
Trefferquote** (bestes Ergebnis: 13 % Recall bei 36 % Fehlalarm). Der Kanal
trägt negative Information und ist nicht eingebaut.

**Die Feuerstoßlänge.** Der Abzug wird über die ganze Aktion gehalten — das
klang nach Verschwendung. Gemessen: 0,85 Schuss je Entscheidung bei 4, 2 und
1 Tic, identisch. Die Kettengun hat ihre eigene Feuerrate. Der Parameter
`--burst` bleibt für andere Waffen, hier ist er wirkungslos.

## Aufbau

| Datei | Inhalt |
|---|---|
| `src/env.py` | ViZDoom-Wrapper, Ringspawn, Schrittsteuerung, Ground Truth |
| `src/detector.py` | LocateAnything-3B, Prompt, Box-Parsing, Dedup, Filter |
| `src/bot.py` | Zielauswahl, Schusssperre, Schutz-Gedaechtnis |
| `src/overlay.py` | Visualisierung (Desktop, quer) |
| `src/vertical.py` | Visualisierung 1080x1920 fuer Shorts/Reels |
| `src/main.py` | Hauptschleife |
| `src/eval_detector.py` | Erkennungsqualitaet gegen die Engine-Wahrheit |
| `src/eval_policy.py` | Wirkung der Sicherungen |
| `src/make_social.py` | Vor-/Abspann und Kodierung für die Plattformen |
| `src/tracking.py` | Spurverfolgung mit Eigenbewegungs-Korrektur |
| `src/classifier.py` | Hypothesen-Turnier für Unbekanntes |
| `src/eval_unknown.py` | Einzelbild gegen Beobachtungsfenster |
| `src/results.py`, `src/results_show.py` | Ergebnisarchiv |
| `src/yolo_detector.py` | YOLO (World oder destilliert) mit derselben Schnittstelle |
| `src/distill.py` | LocateAnything beschriftet, YOLO lernt |
| `src/compare_decoding.py` | Detektoren nebeneinander in Echtzeit, als Video |

Drei Details der Engine, die nicht offensichtlich sind und in `env.py`
gekapselt werden:

* Das Dreh-Delta `TURN_LEFT_RIGHT_DELTA` wirkt **pro Tic**, nicht pro Aktion
  (10 ueber 4 Tics dreht 40 Grad); positive Werte drehen nach rechts.
* `summon` setzt Akteure immer knapp vor den Spieler. Fuer einen Ring auf
  Distanz wird der Spieler per `warp` auf die Zielposition gesetzt, dort
  gespawnt und danach in die Mitte zurueckgesetzt.
* Die Waffenwahl geht nur ueber `Button.SELECT_WEAPON4`; die Konsolenbefehle
  `slot 4` und `weapon chaingun` ignoriert ViZDoom.

## Ergebnisse eines Laufs

700 Schritte, 8 Monster und 5 Marines, `--god` damit die Aufnahme durchlaeuft:

```
Monster:  6/8 erlegt
Menschen: 3/5 am Leben        (zwei von Monstern getoetet, nicht vom Agenten)
Schuesse: 51, Feuer gesperrt: 107
Schuesse mit Mensch im Korridor (Engine-Wahrheit): 5
```

Erzeugte Dateien in `out/`:

| Datei | Format |
|---|---|
| `llm_plays_doom.mp4` | 980x480 quer, Spielbild plus Infopanel |
| `raw_vertical.mp4` | 1080x1920, ein Bild je Agentenschritt |
| `doom_short_1080x1920.mp4` | 55 s, H.264/yuv420p, 30 fps, mit Tonspur — fuer Shorts und Reels |

## Zwei Vorbehalte zu den Zahlen

**Das Modell dekodiert mit Sampling.** Zwei Läufe derselben Konfiguration gaben
19 → 1 → 0 und 13 → 1 → 1 Eigenbeschuss. Richtung und Größenordnung halten, die
exakte Null war ein Sample. Für belastbare Zahlen gehört die Temperatur auf null.

**„Monster erlegt" ist nicht dem Agenten zurechenbar.** In der Arena stehen
befreundete Marines mit BFG, und Monster prügeln sich untereinander. Die Spalte
stand in *jeder* Zeile auf 7/7 — auch dort, wo der Agent sechs Schuss abgab.
Zurechenbar sind `HITCOUNT` und `DAMAGECOUNT`; danach richten sich die neueren
Tabellen.

## Lizenz

LocateAnything-3B steht unter der **NVIDIA License: nur akademische und
nicht-kommerzielle Forschung**. Kommerzieller Einsatz braucht eine Freigabe von
NVIDIA. ViZDoom (MIT) und die mitgelieferten Freedoom-WADs (BSD) sind frei.
