# Piano di ottimizzazione NariQ su Vulkan (RX 9070)

Documento condiviso fra Claude e Codex. Chi lavora lo aggiorna **dopo ogni prova**, così l'altro può riprendere senza tempi morti. Il diario tecnico completo resta in [NARIQ_VULKAN_PROGRESS.md](NARIQ_VULKAN_PROGRESS.md). Questo file contiene obiettivo, stato, regole, attività e registro delle prove.

## Obiettivo

Usare Qwen3.8-27B NariQ 3.0 bpw in OpenClaw (Jarvis) al posto di Unsloth UD-Q2_K_XL, a patto che sia veloce quanto il setup attuale.

Motivo: secondo il rapporto tecnico di Nari Labs (arXiv 2611.26157, tabella 3), la divergenza KL dal BF16 di NariQ 3.0 è da 2,4 a 5 volte più bassa di quella di UD-Q2_K_XL (agente 0,0063 contro 0,0275; generale 0,0173 contro 0,0407). Sulla perplexity WikiText i due modelli sono alla pari (6,25 contro 6,23, misura del 24/11), che è atteso: la perplexity non misura la fedeltà al modello originale.

### Traguardi

Riferimento: setup OpenClaw attuale, fork `d4b007d` con Unsloth UD-Q2_K_XL, DFlash2 3 token, cache q4_0, contesto 114688, ubatch 512.

| Misura | Unsloth attuale | NariQ 24/11 | Traguardo minimo | Traguardo pieno |
| --- | ---: | ---: | ---: | ---: |
| Generazione senza bozza, llama-bench tg | ~39 t/s | 17,4 t/s | 30 | 39 |
| Generazione con DFlash2, contesto breve | ~81 t/s | ~10 t/s | 55 | 75 |
| Generazione con DFlash2 a 65K | ~80 t/s | n/d | 50 | 70 |
| Prefill a 32K | ~910 t/s | ~137 t/s | 450 | 800 |
| VRAM di picco, OpenClaw completo | 14,1-14,5 GiB | n/d | ≤ 14,5 GiB | ≤ 14,5 GiB |
| Qualità | | | logits Vulkan = CPU (top-1 31/31, KL < 1e-6) | idem |

Il cambio in OpenClaw si propone all'utente quando sono raggiunti i traguardi minimi. Decide l'utente.

## Stato attuale

Aggiornare questa sezione a ogni passaggio di consegne.

- **Ultimo aggiornamento**: 24/11/2026 sera, Claude.
- **Build di riferimento**: `build/ref-bin` = `build/packaging-x64/bin` (commit `343fe2183`). Le modifiche del 24/11 sera non cambiano la velocità, quindi resta il riferimento.
- **Scoperta chiave (A1)**: in generazione il prodotto Trellis (`trellis_matvec`) è il 90% del tempo: circa 52 ms su 57 per token. Tutto il resto (attenzione, strati ricorrenti, rotazioni, input/output) costa 5,2 ms. Del prodotto, circa 28 ms sono calcolo per decodificare i pesi (~14-15 istruzioni per peso) e circa 23 ms letture. Le rotazioni costano ~1-2 ms reali: il profilo per fase le gonfiava a 30 ms.
- **In corso**: niente. Codice committato e compilabile.
- **Prossimo passo**: A5a (prodotto con attivazioni int8 e `dot4`), vedi la tabella. È il lavoro che sblocca anche la verifica DFlash2 (B2).

## Regole di lavoro

1. **Prima di iniziare**: leggere "Stato attuale" e l'ultima voce del registro, poi `git status` e `git log --oneline -5`. Se ci sono modifiche non committate non descritte qui, chiedere all'utente prima di toccarle.
2. **Scheda video libera**: prima di ogni misura, ComfyUI e OpenClaw devono essere spenti (`Get-Process llama-server, python -ErrorAction SilentlyContinue`). Chrome aperto sottrae 1-2 GB di VRAM e può falsare le prove con il modello completo. Annotare la VRAM occupata da Windows se supera 2 GB.
3. **Una modifica alla volta**, sempre confrontata con `build/ref-bin` nella stessa sessione (A-B-B-A per i guadagni sotto il 5%).
4. **Correttezza prima della velocità**: nessuna modifica si accetta senza i controlli del protocollo (sezione C).
5. **Dopo ogni prova**, anche fallita: aggiungere una voce in cima al registro e aggiornare "Stato attuale". Una prova fallita vale quanto una riuscita: evita che l'altro la ripeta.
6. **Commit**: ogni modifica accettata va in un commit sul branch `nariq-vulkan-rdna4`, messaggio in inglese con i numeri, chiuso dalla riga `Assisted-by: <agente>`. Dopo il commit, copiare i binari in `build/ref-bin`. **Non pubblicare su GitHub senza l'OK dell'utente.**
7. **Se il lavoro si interrompe a metà**: scrivere in "Stato attuale" quali file sono toccati, cosa manca e se il codice compila.
8. Aggiornare `NARIQ_VULKAN_PROGRESS.md` solo per risultati consolidati; i dettagli delle prove vanno nel registro qui sotto.

## Protocollo di misura

### B1. Build

```powershell
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.363.0'
cmd /c scripts\build-vulkan-windows.cmd     # risultato in build\bin, stampa BUILD_OK
```

Per ricompilare solo i binari dopo il primo configure (più veloce):

```cmd
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\VulkanSDK\1.4.363.0\Bin;%PATH%
cmake --build build --config Release --target llama-bench llama-server test-backend-ops test-trellis test-trellis-model
```

### B2. Velocità rapida (ogni prova)

Stessi argomenti per riferimento e candidato; `GGML_VK_TRELLIS_COOPMAT=1` sempre impostata.

```powershell
$env:GGML_VK_TRELLIS_COOPMAT = '1'
$m = '.\models\Qwen3.8-27B-NariQ-3.0bpw.gguf'
& .\build\bin\llama-bench.exe -m $m -p 512 -n 64 -r 3 -b 512 -ub 512 -fa on -ctk q4_0 -ctv q4_0 -ngl 999 -dev Vulkan0
```

Riportare `pp512` e `tg64` di riferimento e candidato.

### B3. Velocità a contesto lungo (fine di ogni attività)

```powershell
& .\build\bin\llama-bench.exe -m $m -p 512 -n 64 -d 32768 -r 1 -b 512 -ub 512 -fa on -ctk q4_0 -ctv q4_0 -ngl 999 -dev Vulkan0
```

### B4. Profilo per fase (quando serve capire dove va il tempo)

`GGML_VK_PERF_LOGGER=1` e `GGML_VK_TRELLIS_PROFILE=1` (con `GGML_VK_PERF_LOGGER_CONCURRENT` non impostata). Le righe `Trellis GPU us:` sommano le fasi di tutte le proiezioni del grafo. I tempi sono gonfiati dalle barriere: confrontare solo profili presi allo stesso modo. Per leggere il codice macchina di uno shader: `GGML_VK_PIPELINE_ISA=<nome pipeline>` scrive `isa-<nome>-<n>.txt`.

### C. Correttezza (obbligatoria prima di accettare)

```powershell
ctest --test-dir build --output-on-failure -R '^test-trellis'
& .\build\bin\test-backend-ops.exe test -b Vulkan0 -o TRELLIS,TRELLIS_GET_ROWS     # 416/416 al 24/11
& .\build\bin\test-backend-ops.exe test -b Vulkan0 -o MUL_MAT,FLASH_ATTN_EXT         # solo se toccati percorsi comuni
```

Logits del modello completo, CPU contro Vulkan (comandi in `NARIQ_VULKAN_PROGRESS.md`, sezione "Riproduzione delle prove"): top-1 31/31 e stessi token greedy. Annotare la KL massima.

## Attività

Stato: `da fare`, `in corso (chi)`, `fatto`, `scartato`. Le stime di guadagno sono ipotesi da verificare.

### Fase A: generazione (un token alla volta)

| ID | Attività | Stato | Note e guadagno atteso |
| --- | --- | --- | --- |
| A1 | Baseline con B2 e B3 per NariQ e Unsloth con lo stesso binario; profilo di un token; tipi di rotazione usati dal modello | fatto | Vedi registro del 24/11 sera |
| A2 | Rotazioni in un solo passaggio | scartato | Il modello usa solo rotazioni a blocchi (128-4096) già in un passaggio in memoria condivisa; saltare del tutto la rotazione d'ingresso dà solo +4% (17,6 → 18,3 t/s) |
| A3 | Riuso della rotazione d'ingresso fra proiezioni con lo stesso input | bassa priorità | Vale al massimo ~1-2 ms per token |
| A4 | Fondere input, rotazione e uscita con il prodotto | bassa priorità | Tutto ciò che non è prodotto vale 5,2 ms per token |
| A5a | **Prodotto con attivazioni int8 e `dot4`**: un passaggio prima del prodotto quantizza le attivazioni ruotate in int8 per gruppi da 16 (scala float e somma intera per gruppo), nell'ordine in cui le leggono i lane (colonne 2q, 2q+1, 2q+8, 2q+9 di ogni tile). Nel prodotto, per ogni peso: `b = (h >> 2) & 0x3f3f3f3f` e `acc += dot4(b, a_rep)` con `a_rep` = byte dell'attivazione replicato 4 volte (`v_perm`), meno `126 * somma_attivazioni` per tile; scala tile × scala attivazioni in float una volta per tile | da fare | Da ~15 a ~9 istruzioni per peso e una lettura di attivazioni invece di due. Atteso: prodotto da ~52 a ~30-35 ms, generazione ~25-28 t/s. Cambia la numerica (come q8_1 in llama.cpp): controllare logits CPU/Vulkan e KL |
| A5b | Letture dei pesi: caricare le parole di una tile una volta per sottogruppo (24 parole per TRL3) invece di 4 letture a 32 bit per lane, condividerle con shuffle | da fare | Oggi ~23 ms di letture per 9,6 GB (~420 GB/s effettivi) |
| A5c | Moltiplicazione dello stato (`state * 0x83dcd12d`, 32 bit, un quarto di velocità su RDNA): cercare una forma che il compilatore non ricomponga | da fare | Spezzarla in due prodotti a 16 bit non serve: il compilatore li ricompone in una sola moltiplicazione a 32 bit |

### Fase B: verifica DFlash2 (2-4 colonne)

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| B1 | Misurare DFlash2 con 3 token anticipati su NariQ; profilo del prodotto a 4 colonne | da fare | Il 24/11 DFlash2 rallentava (10,5 contro 14,7 t/s) |
| B2 | Prodotto a più colonne: decodificare ogni peso una volta per tutte le colonne | da fare | Con A5a ogni colonna in più costa un solo `dot4` per peso. Oggi `trellis_matvec` (batch ≤ 4) lancia un gruppo per token (`gl_WorkGroupID.y`) e decodifica i pesi una volta per token |
| B3 | Scegliere i token anticipati (2-4) in base ai tempi misurati | da fare | |

### Fase C: prefill

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| C1 | Profilo del prefill a 512 token per fase | da fare | Il 24/11: prodotto 89% del tempo |
| C2 | Preparare `act_hi`/`act_lo` una volta per proiezione invece che per gruppo | da fare | Priorità 1 del diario Codex |
| C3 | Prodotto cooperativo con tile più grandi entro i 32 KB di memoria condivisa del driver Windows; decodifica dei pesi una volta per tile | da fare | Il driver AMD Windows concede 32 KB per gruppo di lavoro |
| C4 | Attenzione del prefill: verificare che la cache q4_0 usi la conversione f16 (`74856e4`) anche con NariQ | da fare | Comune ai due modelli |

### Fase D: integrazione in OpenClaw

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| D1 | Profilo `qwen38-nariq` in `D:\Progetti\openclaw\scripts\Start-OpenClaw-Local.ps1` (modello, mmproj NariQ su CPU, DFlash2) e variante nello script di benchmark | da fare | Solo dopo i traguardi minimi |
| D2 | Confronto completo con `tools\Run-Qwen38-VulkanAB.ps1` e VRAM di picco con il contesto di produzione | da fare | |
| D3 | Sessione reale di Jarvis e analisi del log | da fare | |

## Registro delle prove

Voci più recenti in alto. Formato: data, chi, commit o "non committato", cosa, numeri (riferimento → candidato), esito.

- **24/11/2026 sera, Claude** (commit della sessione, vedi `git log`): somma dei 4 byte di `trellis_grid` con spostamenti e somme invece della moltiplicazione per `0x01010101`. Risultato identico, test 416/416, `test-trellis` superato; nel codice macchina di `trellis_matvec` le moltiplicazioni a 32 bit per ciclo passano da 16 a 8. Velocità invariata (tg64 18,07 → 18,05). Esito: tenuta, ma conferma che non basta togliere le moltiplicazioni.
- **24/11/2026 sera, Claude**: prove a esclusione su tg64 (risultati volutamente sbagliati, interruttori temporanei poi rimossi). Normale 17,60 t/s; senza rotazione d'ingresso 18,29; senza prodotto 193,8; senza prodotto né rotazione 230,3; prodotto con decodifica banale (`float(state & 255)`) 35,0. Quindi per token: prodotto ~52 ms (decodifica ~28, letture ~23), resto 5,2 ms, rotazione d'ingresso ~1 ms. `trellis_matvec`: 35 registri, 1 KB di memoria condivisa, ciclo principale ~110 istruzioni per 8 pesi per lane, 4-6 letture a 32 bit dei codici e 2 da 64 bit delle attivazioni. Esito: priorità ad A5a.
- **24/11/2026 sera, Claude**: rotazioni usate dal modello (nuova diagnostica `GGML_VK_TRELLIS_DEBUG=1`, stampa ogni forma una volta): solo `kind=1` a blocchi, lunghezze 128 (larghezze 5120 e 15488), 256 (15616), 1024 (17408 e 1024), 2048 (6144 e 10240), 4096 (12288), sempre `m=1` e senza parti fuse, quindi sempre il kernel a blocchi in memoria condivisa. Esito: A2 scartata.
- **24/11/2026 sera, Claude**: baseline A1 con `build/ref-bin` (`343fe2183`), stessi argomenti di B2. NariQ pp512 153,2, tg64 17,45; Unsloth pp512 1150,2, tg64 39,30. A 32K (`-d 32768 -r 1`): NariQ 146,6 / 17,44, Unsloth 765,8 / 37,27.

- **24/11/2026, Claude**: perplexity WikiText-2, 40 finestre da 512 token, stessa build per i due modelli. NariQ 6,2455 ± 0,147, Unsloth 6,2317 ± 0,148; NariQ meglio in 19 finestre su 40. Alla pari, coerente con le KL di Nari Labs (la perplexity non misura la fedeltà al BF16). Esito: la qualità resta un motivo valido per il cambio.
