# SN Hunter

Desktopová prohlížečka pro vizuální hledání supernov na FITS snímcích s WCS.
Aktuální snímek porovnává se zarovnaným archivním výřezem DSS2 nebo Pan-STARRS.

## Funkce první verze

- načtení 2D FITS snímku s WCS,
- otevření FITS dialogem nebo přetažením souboru z Průzkumníka do okna,
- interaktivní `asinh`, lineární, odmocninový, logaritmický a mocninný stretch,
- nezávislé nastavení černého a bílého bodu po 0,1 %,
- zapamatování stretche, jeho síly, mezí i pozitivního/negativního zobrazení
  zvlášť pro aktuální a archivní panel,
- samostatný profil černého bodu, bílého bodu a síly pro každý typ stretche;
  hodnoty se obnovují při přepínání i po restartu,
- automatické kontrastní barvy katalogových značek, kurzoru, měřítka a směrové
  růžice pro normální i inverzní zobrazení,
- vyhledání galaxií z HECATE doplněné o jasné galaxie PGC přes HTTPS/TAP
  VizieR,
- filtrování dostupných magnitud,
- archivní výřezy přes CDS HiPS2FITS s lokální cache,
- hromadné předstažení výřezů pro galaxie vybrané pomocí Ctrl/Shift,
- archivní výřez ve vlastní jemné WCS mřížce: Pan-STARRS až 0,25″/px a DSS
  přibližně 1″/px (s bezpečnostním limitem velikosti),
- synchronizace obrazů ve světových souřadnicích bez degradace archivu na
  hrubou pixelovou síť širokoúhlého snímku,
- synchronizovaný posun a zoom,
- živý údaj zoomu v každém panelu, kde 100 % odpovídá jednomu pixelu snímku
  na jeden pixel obrazovky,
- předvolby zoomu 25–800 %, volné zadání procenta z klávesnice a volba
  **Přizpůsobit** pro celý snímek,
- žluté kroužky galaxií přímo v obou snímcích a zelené zvýraznění výběru,
- samostatně vypínatelná vrstva objektů NGC/IC: zelené mlhoviny, modré
  hvězdokupy a fialové ostatní DSO; typy pocházejí z NGC 2000.0 a polohy
  jsou zpřesněné katalogem VII/239A,
- průměr DSO kolečka odpovídá přibližně katalogovému úhlovému rozměru a při
  zoomování se mění spolu s obrazem,
- tooltip s názvem a V magnitudou při najetí na kroužek,
- výběr galaxie kliknutím na její kroužek,
- společný křížový kurzor v obou snímcích se souřadnicemi RA/Dec,
- dynamické úhlové měřítko a krátkou WCS směrovou růžici sever–východ,
- side-by-side a blink porovnání.
- volitelné skrytí archivního panelu; aktuální snímek se roztáhne přes uvolněné
  místo a načtený archiv zůstane zachovaný pro opětovné zobrazení.

## Spuštění

V PowerShellu v adresáři projektu:

```powershell
python -m pip install -e .
sn-hunter
```

V tomto počítači jsou hlavní knihovny již dostupné, takže lze aplikaci spustit
také dvojklikem na `spustit_sn_hunter.bat`, případně bez instalace balíčku:

```powershell
python -m sn_hunter.app
```

Pro kvalitnější a rychlejší reprojekci lze volitelně nainstalovat:

```powershell
python -m pip install -e ".[reproject]"
```

## Použití

1. Otevřete jeden FITS přes **Otevřít FITS…**, nebo tlačítkem **Přidat snímky
   noci…** vložte celou sérii polí.
2. Nastavte limit magnitudy a zvolte **Najít galaxie**.
3. Vyberte galaxii; oba panely se na ni automaticky přiblíží.
4. Zvolte přehlídku a velikost výřezu, potom **Načíst podklad**.
5. Upravte stretch obou panelů nebo zapněte **Blink**.

Archivní výřez nemusí být navázaný na galaxii. Posuňte nebo přibližte aktuální
snímek a zvolte **Načíst střed zobrazení**; střed výřezu se vezme z geometrického
středu právě viditelné oblasti levého panelu.

Pravé kliknutí do kteréhokoli obrazového panelu zkopíruje RA/Dec daného místa
do schránky ve formátu `08 15 34 +15 35 41`, bez znaků jednotek.

## Snímky noci

Seznam **Snímky noci** je oddělený od seznamu galaxií aktuálního pole. První
snímek lze ihned prohlížet, zatímco aplikace postupně na pozadí načítá WCS,
galaxie HECATE a objekty NGC/IC dalších polí. Je-li zapnuté **Připravovat
archivní podklady na pozadí**, stáhne také podklad z právě zvolené přehlídky pro
každou nalezenou galaxii. U položek se zobrazuje stav katalogů a průběh archivu.
Po kliknutí na jiné připravené pole se jeho katalogové značky zobrazí okamžitě;
archivní výřezy se načtou z místní cache.

## Podezřelé body

Zapněte **Označovat podezřelé body** a klikněte levým tlačítkem na zajímavé
místo aktuálního snímku. Aplikace uloží přesné RA/Dec, čas označení a cestu k
FITS souboru. Kandidát se zobrazí výrazným křížkem v aktuálním i archivním
panelu a zůstane zachovaný při přepínání polí i po restartu. Seznam je běžně
sbalený pod tlačítkem **Seznam podezřelých bodů (N)**, aby nezmenšoval hlavní
seznam galaxií. Kliknutí na jeho položku otevře příslušný snímek,
vycentruje uloženou pozici a rovnou pro ni načte archivní výřez ze zvoleného
průzkumu (DSS nebo Pan-STARRS). Chybnou značku lze odstranit tlačítkem pod
seznamem.

Více galaxií lze v seznamu označit pomocí Ctrl nebo Shift a tlačítkem
**Stáhnout označené (N)** předem stáhnout jejich podklady. Tlačítko
**Stáhnout všechny v seznamu (N)** nevyžaduje ruční označování. Dvě stahování
probíhají souběžně a stavový řádek ukazuje právě stahovanou galaxii i průběh;
následné **Načíst podklad** už použije místní cache. Podklad aktivní galaxie se
zobrazí hned po jejím stažení (není nutné čekat na celou dávku) a při přechodu
na jinou předstaženou galaxii se její podklad načte z cache bez dalšího kliknutí.
Při **Blinku** používá archivní fáze vlastní stretch pravého panelu, zatímco
fáze aktuálního FITS přebírá typ stretche, černý a bílý bod, sílu i negativ
z levého panelu. Percentilové úrovně se počítají z původního FITS.

Internet je potřeba jen pro první katalogový dotaz a první stažení konkrétního
archivního výřezu. Stažené FITS soubory se ukládají do
`~/.cache/sn-hunter/archive`.

## Poznámky

- Hlavní katalogový dotaz filtruje galaxie podle homogenizované celkové V
  magnitudy HECATE už na serveru, takže je použitelný i pro širokoúhlé snímky.
  HECATE je katalog blízkých galaxií (zhruba do 200 Mpc).
- Chybějící jasné galaxie PGC doplňuje GLADE 2 a jejich rozměry HyperLEDA.
  Je-li dostupná homogenizovaná barva HyperLEDA `B−V`, aplikace dopočítá
  vizuální odhad `V = B − (B−V)` a označí jej `V≈`. Jinak použije
  konzervativní mez `B ≤ V limit + 1` a hodnotu označí `B≈`; nevydává ji
  tedy za změřenou V magnitudu. Shody se slučují podle PGC čísla, takže
  nevznikají duplicitní kroužky.
- Výchozí limit galaxií se nastavuje podle WCS rozměru každého pole:
  `17,5 mag` do 2°, `16 mag` mezi 2–5° a `14,5 mag` nad 5°. U aktuálního
  snímku jej lze kdykoli ručně změnit. Noční fronta počítá limit pro každý
  FITS samostatně.
- Reprojekce řeší orientaci a měřítko, nikoli rozdílný seeing, filtr nebo PSF.
- Pan-STARRS nepokrývá celou oblohu; DSS2 je bezpečnější výchozí volba.

