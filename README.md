# SN Hunter 1.0

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
- automatické zvětšení výřezu pro rozměrné galaxie na 1,5násobek jejich
  katalogového hlavního rozměru, nejvýše na 15′ (ručně zvolená větší hodnota
  má přednost),
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
- dva nezávislé přepínače kroužků galaxií v horní liště: v aktuálním snímku
  jsou standardně zapnuté, v archivním/blink panelu vypnuté, aby nezakrývaly
  případnou supernovu; oba panely lze samostatně vyčistit nebo označit,
- samostatně vypínatelná vrstva objektů NGC/IC: zelené mlhoviny, modré
  hvězdokupy a fialové ostatní DSO; typy pocházejí z NGC 2000.0 a polohy
  jsou zpřesněné katalogem VII/239A,
- průměr DSO kolečka odpovídá přibližně katalogovému úhlovému rozměru a při
  zoomování se mění spolu s obrazem,
- tooltip s názvem a V magnitudou při najetí na kroužek,
- výběr galaxie kliknutím na její kroužek,
- společný křížový kurzor v obou snímcích se souřadnicemi RA/Dec,
- dynamické úhlové měřítko a krátkou WCS směrovou růžici sever–východ,
- side-by-side a blink porovnání,
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

## Uživatelská příručka

SN Hunter je určený především k rychlé vizuální kontrole většího počtu galaxií.
V levém panelu ukazuje nový pozorovací snímek, v pravém panelu archivní oblohu
zarovnanou podle WCS. Není tedy nutné ručně otáčet snímek ani hledat shodné
měřítko. Cílem je všimnout si bodového zdroje, který v archivním snímku není.

Vstupní FITS musí být dvourozměrný a obsahovat platnou WCS astrometrii. Program
nemění ani nepřepisuje původní FITS soubory.

### Rychlý začátek s jedním polem

1. Otevřete FITS pomocí **Otevřít FITS…**, klávesou `Ctrl+O` nebo přetažením
   souboru z Průzkumníka do okna.
2. Upravte stretch levého panelu tak, aby byly viditelné galaxie i slabé bodové
   zdroje. Pro první pokus je vhodný `asinh`.
3. Zvolte **Najít galaxie**. Výchozí limit magnitudy se nastaví automaticky
   podle velikosti zorného pole, ale lze jej ručně změnit.
4. Klikněte na galaxii v seznamu nebo na její kroužek v levém snímku.
5. Vyberte archivní přehlídku, obvykle **DSS2 Red**, a zvolte **Načíst
   podklad**. Pokud byl výřez předstažený, zobrazí se automaticky z cache.
6. Porovnávejte panely vedle sebe nebo použijte automatický blink v pravém
   panelu. Tlačítky **Předchozí/Další** postupujte seznamem.

### Obrazové panely

Levý panel obsahuje původní aktuální FITS. Pravý panel obsahuje archivní výřez
a při blinku střídá archiv se stejnou oblastí aktuálního snímku. Oba panely
sdílejí nebeský střed a měřítko prostřednictvím WCS.

- Kolečkem myši lze měnit zoom a tažením obraz posouvat.
- V poli **Zoom** lze vybrat předvolbu, napsat vlastní procento nebo použít
  **Přizpůsobit**. Hodnota 100 % znamená jeden pixel snímku na jeden pixel
  obrazovky.
- **Skrýt archivní panel** uvolní místo pro aktuální snímek. Po opětovném
  zobrazení zůstane archiv i poloha pohledu zachovaná.
- Společný kurzor ukazuje stejné nebeské místo v obou panelech. Stavový řádek
  zobrazuje jeho RA/Dec.
- Pravé kliknutí do obrazu zkopíruje souřadnice do schránky ve formátu
  `08 15 34 +15 35 41`.
- Měřítko a směrová růžice `N`/`E` se počítají z WCS aktuálního panelu.

Pokud výřez u galaxie ležící na kraji původního FITS přesahuje mimo nasnímanou
oblast, bude chybějící část aktuálního obrazu černá. Archiv je přesto zobrazený
celý a galaxie zůstane ve středu.

### Stretch, jas a negativ

Každý panel má vlastní typ stretche, černý bod, bílý bod, sílu nebo γ a
přepínač **Negativ**. Nastavení levého a pravého panelu jsou nezávislá. Program
si navíc pamatuje samostatné hodnoty pro `asinh`, `linear`, `sqrt`, `log` a
`power`, takže lze mezi metodami přepínat bez ztráty nastavení.

Při blinku se archivní fáze vykresluje nastavením pravého panelu a aktuální
fáze nastavením levého panelu. Pokud jas obou fází výrazně skáče, upravte jejich
černý a bílý bod samostatně. Kroužky galaxií jsou v pravém panelu standardně
vypnuté, aby nezakryly případnou supernovu.

### Blink a klávesové ovládání

Po načtení archivního podkladu se blink spustí automaticky, pokud nebyl dříve
vypnutý. Probíhá pouze v pravém panelu, takže vlevo zůstává současně k dispozici
celý aktuální snímek.

| Akce | Klávesa |
|---|---|
| Otevřít FITS | `Ctrl+O` |
| Předchozí archivní výřez | `Page Up` |
| Další archivní výřez | `Page Down` |
| Spustit nebo zastavit blink | `End` |

Horní lišta ukazuje pořadí právě prohlíženého výřezu. Blízké galaxie se v
navigaci sloučí do jednoho kroku jen tehdy, když se celé bezpečně vejdou do
společného výřezu. V katalogovém seznamu zůstávají všechny a lze je otevřít
jednotlivě.

### Katalogové značky

- **Galaxie: aktuální** zapíná značky galaxií v aktuálním snímku.
- **Galaxie: archiv** zapíná stejné značky v pravém panelu. Standardně
  jsou vypnuté kvůli nerušenému blinku.
- **Objekty NGC/IC** zobrazují zeleně mlhoviny, modře hvězdokupy a fialově
  ostatní objekty hlubokého nebe.
- Po najetí myší na značku se zobrazí název, dostupná magnituda a u DSO také
  typ objektu.

Katalogové kroužky jsou orientační pomůcka. Při rozhodování o podezřelém zdroji
je vhodné je dočasně vypnout, aby jejich obrys nic nezakrýval.

### Oblast bez katalogové galaxie

Archivní výřez nemusí být navázaný na položku seznamu. Posuňte levý snímek tak,
aby požadovaná oblast ležela uprostřed panelu, a zvolte **Načíst střed
zobrazení**. Tím lze prověřit libovolnou mlhavou skvrnu nebo podezřelý zdroj.

### Doporučený postup při hledání supernovy

1. Nastavte levý obraz tak, aby nebyla vypálená jádra galaxií a současně byly
   viditelné slabé hvězdy.
2. Spusťte blink a sledujte především okolí disku a jádra vybrané galaxie.
3. Zdroj přítomný jen v aktuální fázi prověřte také s jiným stretchem a bez
   katalogových kroužků.
4. Zapněte **Označovat podezřelé body**, kliknutím zdroj uložte a režim zase
   vypněte. K bodu se lze později vrátit ze seznamu.
5. Před případným hlášením ověřte, že nejde o kosmický zásah, hotpixel,
   asteroid, odlesk, stopu družice nebo chybu registrace.

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
seznamem. Podezřelé body jsou pracovní poznámky pro aktuální pozorovací noc:
při spuštění se automaticky odstraní body starší než 7 dní a také body, jejichž
zdrojový FITS soubor už není dostupný.

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
`~/.cache/sn-hunter/archive` (ve Windows typicky
`C:\Users\<uživatel>\.cache\sn-hunter\archive`). Cache archivních FITS je
omezena na 1 GiB; při spuštění a při překročení limitu aplikace automaticky
odstraňuje nejdéle nepoužité výřezy. Katalogová cache je oddělená a podstatně
menší.

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
- Úhlové rozměry galaxií jsou uvnitř programu vedené jako celé průměry.
  Poloosy `MajAxis`/`MinAxis` z HECATE se při načtení převádějí na průměry;
  díky tomu dostanou velké galaxie, například NGC 891, dostatečný archivní
  výřez s okrajem.
- Výchozí limit galaxií se nastavuje podle WCS rozměru každého pole:
  `17,5 mag` do 2°, `16 mag` mezi 2–5° a `14,5 mag` nad 5°. U aktuálního
  snímku jej lze kdykoli ručně změnit. Noční fronta počítá limit pro každý
  FITS samostatně.
- Reprojekce řeší orientaci a měřítko, nikoli rozdílný seeing, filtr nebo PSF.
- Pan-STARRS nepokrývá celou oblohu; DSS2 je bezpečnější výchozí volba.

