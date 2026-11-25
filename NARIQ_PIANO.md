# Piano di ottimizzazione NariQ su Vulkan (RX 9070)

Documento condiviso fra Claude e Codex. Chi lavora lo aggiorna **dopo ogni prova**, così l'altro può riprendere senza tempi morti. Il diario tecnico completo resta in [NARIQ_VULKAN_PROGRESS.md](NARIQ_VULKAN_PROGRESS.md). Questo file contiene obiettivo, stato, regole, attività e registro delle prove.

## Obiettivo

Usare Qwen3.8-27B NariQ 3.0 bpw in OpenClaw (Jarvis) al posto di Unsloth UD-Q2_K_XL, a patto che sia veloce quanto il setup attuale.

Motivo: secondo il rapporto tecnico di Nari Labs (arXiv 2611.26157, tabella 3), la divergenza KL dal BF16 di NariQ 3.0 è da 2,4 a 5 volte più bassa di quella di UD-Q2_K_XL (agente 0,0063 contro 0,0275; generale 0,0173 contro 0,0407). Sulla perplexity WikiText i due modelli sono alla pari (6,25 contro 6,23, misura del 24/11), che è atteso: la perplexity non misura la fedeltà al modello originale.

### Traguardi

Riferimento: setup OpenClaw attuale, fork `d4b007d` con Unsloth UD-Q2_K_XL, DFlash2 3 token, cache q4_0, contesto 114688, ubatch 512.

| Misura | Unsloth attuale | NariQ 24/11 | Traguardo minimo | Traguardo pieno |
| --- | ---: | ---: | ---: | ---: |
| Generazione senza bozza, llama-bench tg | ~39 t/s | 17,4 t/s (24,9 al 25/11) | 30 | 39 |
| Generazione con DFlash2, contesto breve | ~81 t/s (87 il 25/11) | ~10 t/s (25,4 al 25/11 in D2; ~41 su prompt ripetitivo) | 55 | 75 |
| Generazione con DFlash2 a 65K | ~80 t/s (78 il 25/11) | 25,6 t/s al 25/11 (D2) | 50 | 70 |
| Prefill a 32K | ~910 t/s (llama-bench d32768: 766) | ~137 t/s (472 al 25/11, llama-bench d32768) | 450 | 800 |
| VRAM di picco, OpenClaw completo | 14,1-14,5 GiB | 14,96 GiB con DFlash2 al 25/11 (D2) | ≤ 14,5 GiB | ≤ 14,5 GiB |
| Qualità | | | logits Vulkan = CPU (top-1 31/31, KL < 1e-6) | idem |

Il cambio in OpenClaw si propone all'utente quando sono raggiunti i traguardi minimi. Decide l'utente.

## Stato attuale

Aggiornare questa sezione a ogni passaggio di consegne.

- **Ultimo aggiornamento**: 25/11/2026, Claude (E1 chiusa, B6 in corso).
- **Build di riferimento**: `build/ref-bin` = binari del commit `71b71362e` (C3, versione 64 × 32); `build/bin` ha anche `d4d3eb6e3` (micro-batch 64 della bozza, cambia solo la memoria). La build iniziale resta in `build/packaging-x64/bin` (`343fe2183`).
- **Confronto completo in stile OpenClaw (D2, 25/11)**: `D:\Progetti\openclaw\tools\Run-Qwen38-VulkanAB.ps1 -Backends nariq-dflash3r,fork-dflash3r` con `OPENCLAW_AB_UB=512` (varianti `nariq` e `nariq-dflash3r` aggiunte allo script; la build NariQ si cambia con `OPENCLAW_AB_NARIQDIR`). Report `D:\Progetti\openclaw\runtime\benchmarks\20261124-212542-qwen38-backend-ab`. NariQ con DFlash2 3 token: generazione 25,4 / 23,9 / 25,6 t/s (breve / 32K / 65K), prefill 522 / 432, accettazione 35-44%, ciclo 84-90 ms, picco VRAM 14,96 GiB. Unsloth sul fork di produzione: 87,0 / 90,0 / 78,3, prefill 911 / 661, accettazione 44-55%, ciclo 27-32 ms, 14,06 GiB. **Con NariQ DFlash2 oggi non rende**: un ciclo costa circa il doppio di un token (~41 ms) e produce ~2,1 token, quindi la generazione resta ~25 t/s come senza bozza. Senza bozza si risparmia anche ~1,3 GiB di VRAM.
- **Dove sta il margine in generazione**: (1) il prodotto a 4 token (verifica) costa ~1,9 volte quello a 1 token: se scendesse a ~1,2 volte, con l'accettazione misurata DFlash2 darebbe ~40 t/s; (2) il prodotto a 1 token (~36 ms per token) limitato dal calcolo della decodifica (vedi "Valutazione onesta"). Entrambi dipendono dal costo della decodifica per peso, per questo la valutazione HIP (E1) è utile.
- **Ultimi risultati**: generazione senza bozza 17,5 → 24,9 t/s; con DFlash2 a 3 token ~41 t/s sul prompt ripetitivo di prova (prima ~10). **Prefill (C3) 150 → ~602 t/s a contesto vuoto, 147 → 472 a 32K** (llama-bench): traguardo minimo del prefill raggiunto. Il percorso cooperativo è ora predefinito (`GGML_VK_TRELLIS_COOPMAT=0` per disattivarlo): non serve più impostare la variabile.
- **Riferimento NVIDIA**: Nari Labs dichiara 88,9 t/s a 4K e 70,7 a 64K su una RTX PRO 6000 (Blackwell, ~1,8 TB/s, circa 2,8 volte la banda della RX 9070), senza speculazione, cache KV BF16. In proporzione alla banda equivarrebbero a ~32 t/s qui. Il GGUF pubblicato non contiene il modulo MTP ("non ancora quantizzato"), quindi MTP non è disponibile; le alternative alla bozza DFlash2 sono `ngram-simple` (nessun modello, utile su testo ripetitivo e codice) o nessuna speculazione.
- **Profilo di un ciclo DFlash2 (B4)**: il ciclo (~83 ms) è dominato dai prodotti Trellis a 4 token del modello (il prodotto a 4 token costa ~1,9 volte quello a 1 token: la decodifica di ogni peso, ~7 istruzioni, si fa una volta, ma ogni token aggiunge un `dot4` per peso e due letture per tile). La bozza costa ~10 ms per ciclo, ~6 nelle bande della testa di uscita Trellis condivisa (normali: 5120 → 15488 a 4 token 197 µs). Le proiezioni da 48 righe erano anomale (corrette in `7a5cae586`).
- **Valutazione onesta**: la decodifica Trellis costa ~7 istruzioni per peso su RDNA (una moltiplicazione intera a 32 bit, a un quarto di velocità, più estrazioni di bit), contro ~1-2 per i formati IQ/K di Unsloth. Il limite stimato in generazione è ~28-30 t/s senza bozza e ~45-50 con DFlash2 su testo ripetitivo; la parità con Unsloth (39 / ~80) non sembra raggiungibile con questo formato su questa scheda. Ho provato a evitare la moltiplicazione con due prodotti a 24 bit: il compilatore li ricompone in una moltiplicazione a 32 bit, e una versione che non si ricompone costa le stesse istruzioni.
- **Prefill e OpenClaw**: a 472 t/s un prompt da 32K richiede ~68 s (Unsloth ~40 s, prima ~3,5 minuti).
- **Scoperta chiave (A1)**: in generazione il prodotto Trellis (`trellis_matvec`) è il 90% del tempo: circa 52 ms su 57 per token. Tutto il resto (attenzione, strati ricorrenti, rotazioni, input/output) costa 5,2 ms. Del prodotto, circa 28 ms sono calcolo per decodificare i pesi (~14-15 istruzioni per peso) e circa 23 ms letture. Le rotazioni costano ~1-2 ms reali: il profilo per fase le gonfiava a 30 ms.
- **In corso (25/11, Claude)**: B6, verifica a 4 token più economica. HIP chiuso per decisione dell'utente dopo E1.
- **[HIP] E1 conclusa**: su RX 9070 con ROCm 10.2.0, PR #3 `f52affb46` e DLL del venv, B2 224,14 / 8,01 t/s e B3 a profondità 32.768 213,77 / 7,98 t/s (prefill / generazione). CTest 2/2, TRL1–6 e 380/380 prodotti passati; modello 31/31 top-1 a microbatch 1 e 32. I 65/65 layer sono su ROCm0, l'embedding Trellis resta su CPU. Nessuna delle soglie E1 è raggiunta. Raccomandazione: fermare l'ottimizzazione HIP e sottoporre all'utente il prossimo passo.
- **[HIP] Profilo prefill concluso**: unpack + GEMM pesano l'83,1% nel prefill breve e il 75,1% a profondità 32.768. A 32K il prefill profilato è 212,25 t/s. L'intervento con più margine è un prototipo WMMA RDNA4 che sostituisca entrambe le fasi, con validazione e soglie E1. Per 550 t/s servirebbe circa 5,5 volte sul blocco a parità degli altri tempi. Piano in `D:\Progetti\NariQ HIP RDNA4\NARIQ_HIP_OPTIMIZATION.md`. Il prototipo di ottimizzazione resta da provare.
- **Prossimo passo**: decisione dell'utente dopo E1. Su Vulkan, se si continua: B6 (verifica a 4 token più economica: per esempio prodotto a 4 token con le matrici int8 della scheda, oppure meno letture delle attivazioni per token) e A5b (letture dei codici condivise nel sottogruppo). Margini aperti sul prefill: blocchi di token più grandi (`VK_TRELLIS_BATCH` = 128: ogni peso si decodifica 4 volte per 512 token), decodifica in anticipo del passo successivo (doppio buffer in memoria condivisa).
- **Script di prova del server**: lo scratchpad di Claude non è condiviso. Per una prova equivalente avviare `build\bin\llama-server.exe` con gli argomenti dell'esempio in `NARIQ_VULKAN_PROGRESS.md` più `--spec-type draft-dflash --spec-draft-model D:\Progetti\openclaw\models\qwen38-27b-dflash2\Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-draft-device Vulkan0 --spec-draft-ngl 99 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 --spec-draft-n-max 3 --spec-accept rejection -bs --spec-draft-vocab 65536`, `-c 32768 -ub 512`, e inviare a `/completion` un prompt di ~1500 token con `n_predict` 256; leggere `timings.predicted_per_second`, `draft_n`, `draft_n_accepted`.

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
| A5a | **fatto** (`a66f18956`, +33%). **Prodotto con attivazioni int8 e `dot4`**: un passaggio prima del prodotto quantizza le attivazioni ruotate in int8 per gruppi da 16 (scala float e somma intera per gruppo), nell'ordine in cui le leggono i lane (colonne 2q, 2q+1, 2q+8, 2q+9 di ogni tile). Nel prodotto, per ogni peso: `b = (h >> 2) & 0x3f3f3f3f` e `acc += dot4(b, a_rep)` con `a_rep` = byte dell'attivazione replicato 4 volte (`v_perm`), meno `126 * somma_attivazioni` per tile; scala tile × scala attivazioni in float una volta per tile | da fare | Da ~15 a ~9 istruzioni per peso e una lettura di attivazioni invece di due. Atteso: prodotto da ~52 a ~30-35 ms, generazione ~25-28 t/s. Cambia la numerica (come q8_1 in llama.cpp): controllare logits CPU/Vulkan e KL |
| A5b | Letture dei pesi: caricare le parole di una tile una volta per sottogruppo (24 parole per TRL3) invece di 4 letture a 32 bit per lane, condividerle con shuffle | da fare | Oggi ~23 ms di letture per 9,6 GB (~420 GB/s effettivi) |
| A5c | Moltiplicazione dello stato (`state * 0x83dcd12d`, 32 bit, un quarto di velocità su RDNA): cercare una forma che il compilatore non ricomponga | da fare | Spezzarla in due prodotti a 16 bit non serve: il compilatore li ricompone in una sola moltiplicazione a 32 bit |

### Fase B: verifica DFlash2 (2-4 colonne)

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| B1 | Misurare DFlash2 con 3 token anticipati su NariQ | fatto | Dopo B2: senza bozza 24,0, bozza 2/3/4 token 38,7 / 42,2 / 17,1 t/s (prompt ripetitivo). Ciclo ~87 ms |
| B4 | Profilo di un ciclo DFlash2 (modello a 4 token, bozza, teste di uscita) | fatto | Vedi "Stato attuale" e il registro del 25/11. Corrette le proiezioni da 48 righe (`7a5cae586`) |
| B5 | Estendere il prodotto int8 a 5-8 token (bozza a 4-7) se il profilo lo giustifica | da fare | Oggi oltre 4 token si cade su `trellis_matmul` float |
| B2 | Prodotto a più colonne: decodificare ogni peso una volta per tutte le colonne | fatto (`3b8cc30e7`) | 4 token: TRL3 407 → 216 µs; 1 token invariato grazie a una variante dedicata | Con A5a ogni colonna in più costa un solo `dot4` per peso. Oggi `trellis_matvec` (batch ≤ 4) lancia un gruppo per token (`gl_WorkGroupID.y`) e decodifica i pesi una volta per token |
| B3 | Scegliere i token anticipati (2-4) in base ai tempi misurati | da fare | |
| B6 | Verifica a 4 token più economica: portare il prodotto a 4 token da ~1,9 a ~1,2-1,3 volte quello a 1 token | in corso (Claude, 25/11) | Con l'accettazione misurata DFlash2 darebbe ~35-40 t/s |

### Fase C: prefill

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| C1 | Profilo del prefill a 512 token per fase | da fare | Il 24/11: prodotto 89% del tempo |
| C2 | Preparare `act_hi`/`act_lo` una volta per proiezione invece che per gruppo | fatto in C3 | `trellis_prep16.comp`; il residuo è opzionale (`GGML_VK_TRELLIS_MM_RESIDUAL=1`, 28% più lento, usato dal test elemento per elemento) |
| C3 | Prodotto cooperativo con tile più grandi entro i 32 KB di memoria condivisa del driver Windows; decodifica dei pesi una volta per tile | fatto (`af07eaa13`) | 32 uscite × 128 token per gruppo, 256 thread che decodificano 8 tile ogni 64 input, attivazioni FP16 preparate una volta per pezzo con una potenza di 2 per token. pp512 150 → ~490, 32K 147 → 395 |
| C5 | Prodotto cooperativo: forma del gruppo di lavoro | fatto (`71b71362e`) | 64 righe × 32 input per passo, migliore di 2×4, 4×4, 4×1, 8×1, 8×2 (vedi registro); k diviso per le proiezioni da 48 righe |
| C4 | Attenzione del prefill: verificare che la cache q4_0 usi la conversione f16 (`74856e4`) anche con NariQ | da fare | Comune ai due modelli |

### Fase D: integrazione in OpenClaw

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| D1 | Profilo `qwen38-nariq` in `D:\Progetti\openclaw\scripts\Start-OpenClaw-Local.ps1` (modello, mmproj NariQ su CPU, DFlash2) e variante nello script di benchmark | da fare | Solo dopo i traguardi minimi |
| D2 | Confronto completo con `tools\Run-Qwen38-VulkanAB.ps1` e VRAM di picco con il contesto di produzione | fatto (25/11) | Vedi "Stato attuale" e il registro |
| D3 | Sessione reale di Jarvis e analisi del log | da fare | |

### Fase E: alternative

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| E1 | Valutare un backend HIP (ROCm SDK 7.1 già installato in `C:\Program Files\AMD\ROCm\7.1`): compilare il fork Nari Labs con la PR HIP per RX 7900 XTX (nari-labs/nari-llama.cpp#3) per gfx1201 su Windows e misurare NariQ | chiusa (25/11): HIP fermato | Esito Codex con ROCm 10.2.0: pp512 224, tg64 8,0, a 32K 214 / 8,0 t/s, sotto Vulkan in tutto; il vantaggio teorico (WMMA, dot4) è già usato da Vulkan e la decodifica costa uguale. Decisione dell'utente: chiudere HIP, proseguire con B6 su Vulkan. Note iniziali: | Vantaggi possibili per i soli kernel Trellis: moltiplicazione a 24 bit esplicita (`__umul24`), 64 KB di memoria condivisa per gruppo, intrinseci per permutazioni e WMMA, kernel CUDA di Nari Labs adattabili. Svantaggi: tutto il resto del modello girerebbe su llama.cpp HIP, che su questa macchina era più lento di Vulkan in generazione (prove del 09/11 in OpenClaw), e le ottimizzazioni di questo fork (attenzione per il driver AMD, DFlash2, MTP, ecc.) non ci sono |

#### Piano operativo E1 (HIP)

Obiettivo: sapere in poche ore se una build HIP di NariQ parte già meglio della build Vulkan di questo ramo, per decidere se continuare su HIP, su Vulkan o su entrambi (Vulkan eventualmente solo per Unsloth, con il port Vulkan proposto a Nari Labs per altri).

1. **Cartella separata**: lavorare in una nuova cartella, per esempio `D:\Progetti\NariQ HIP RDNA4`, mai in questa (build e test di Vulkan restano intatti). Usare il GGUF di questa cartella con il percorso assoluto, senza copiarlo (10 GB). Una sola misura alla volta sulla scheda: prima di ogni prova controllare che non giri un altro `llama-server`/`llama-bench` (anche di Claude).
2. **Sorgente**: `git clone https://github.com/nari-labs/nari-llama.cpp` (HEAD verificato `8e0e016e0`), poi `git fetch origin pull/3/head:hip-rdna3` e `git checkout hip-rdna3`. Annotare il commit di partenza nel registro.
3. **Toolchain: ROCm 10.2.0 (TheRock), la stessa versione del driver installato** (verificato da Claude il 25/11):
   - **Situazione sul PC.** Driver AMD Software 26.11.2, cioè quello di ROCm 10.2.0 (HIP 7.17). L'HIP SDK di sistema in `C:\Program Files\AMD\ROCm\7.1` è vecchio: è l'unico installato e `HIP_PATH` punta lì. ROCm 10.2.0 c'è già, ma solo come pacchetti pip dentro ComfyUI (`D:\Progetti\Confy UI\ComfyUI_windows_portable\python_embeded`: `rocm[libraries,device-gfx1201]==10.2.0`, da `https://stable.repo.amd.com/rocm/whl-next/`). Ha `hipcc`, `clang++`, `hipInfo` e gli header HIP, ma non il pacchetto `devel`: mancano gli header e le configurazioni CMake di hipBLAS/rocBLAS che servono a `-DGGML_HIP=ON`.
   - **Cosa non toccare.** Non installare né modificare nulla nel Python di ComfyUI o nel suo ambiente di training. Non disinstallare l'HIP SDK 7.1. Non cambiare `HIP_PATH`/`PATH` di sistema: impostarli solo nella shell di build e di prova.
   - **Installazione.** Prima di scaricare, chiedere conferma all'utente indicando pacchetti, origine e dimensione. Poi creare un ambiente separato nella cartella HIP con il Python di sistema 3.12:
     - `py -3.12 -m venv "D:\Progetti\NariQ HIP RDNA4\.venv-rocm"`
     - `pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel,device-gfx1201]==10.2.0"` (dimensione da verificare, probabilmente 1-2 GB: annotarla nel registro)
     - `rocm-sdk init` (estrae il pacchetto devel), poi `rocm-sdk path --root` e `rocm-sdk path --bin` per avere i percorsi.
     - Se l'indice o i comandi `rocm-sdk` non corrispondono, controllare la documentazione TheRock/ROCm 10.2 e annotare cosa è servito.
   - **Verifica.** `hipInfo.exe` di quell'ambiente deve vedere la RX 9070 come `gfx1201`.
   - **Build.** Seguire `docs/build.md`, sezione HIP per Windows, della stessa revisione: configurazione con Ninja, `-DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON`. Usare `clang`/`clang++` di `<root>\lib\llvm\bin`, con `HIP_PATH` e `CMAKE_PREFIX_PATH` che puntano alla radice `rocm-sdk path --root`. Se la PR fissa `gfx1100`, aggiungere `gfx1201` ai target e annotare cosa è servito.
   - **Esecuzione.** I binari HIP cercano le DLL ROCm (`amdhip64_7.dll`, hipBLAS, rocBLAS e i kernel gfx1201): mettere la cartella `rocm-sdk path --bin` (e quella delle librerie, se separata) in testa al `PATH` della sola shell di prova.
   - **Riserva.** Se con ROCm 10.2 la build non riesce dopo tentativi ragionevoli, provare con l'HIP SDK 7.1 già installato e annotarlo. L'HIP SDK 7.2 (luglio) è più vecchio della 10.2 e del driver: non serve.
4. **Correttezza**: `test-backend-ops test -b ROCm0 -o TRELLIS,TRELLIS_GET_ROWS` (o il nome del dispositivo HIP mostrato dall'avvio) e confronto dei logits con la CPU come in "Riproduzione delle prove" di `NARIQ_VULKAN_PROGRESS.md` (`test-trellis-model`, `nariq\compare_logits.py`). Il caso TRL1 sintetico risulta aperto nella PR: annotare se fallisce.
5. **Misure, stesso protocollo di B2/B3**: `llama-bench -m <GGUF> -p 512 -n 64 -r 3 -b 512 -ub 512 -fa on -ctk q4_0 -ctv q4_0 -ngl 999` e lo stesso con `-d 32768 -r 1`. Riferimento Vulkan di questo ramo al 25/11: pp512 ~602, tg64 ~24,4, a 32K pp ~472 / tg ~23,5; Unsloth sul fork di produzione: pp512 ~1150, tg64 ~39, a 32K 766 / 37,3. Annotare anche il picco di VRAM.
6. **Decisione** (scriverla nel registro e chiedere all'utente): se HIP senza modifiche arriva già a tg64 ≥ 28 o pp a 32K ≥ 550, vale la pena ottimizzare HIP; se è sotto la build Vulkan su entrambi, fermarsi.
7. **Se si continua su HIP**, nell'ordine: verificare se il fork Nari Labs supporta DFlash2 e la bozza `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (in questo ramo condivide embedding e testa Trellis); poi nei kernel Trellis la decodifica con `__umul24` o `__builtin_amdgcn_mul_u24` al posto della moltiplicazione a 32 bit, i prodotti a 1-4 token con attivazioni int8 e `dot4` (`__builtin_amdgcn_sudot4`), il prefill con WMMA (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`) e tile da 64 KB di memoria condivisa. Le idee e i numeri della versione Vulkan (A5a, B2, C3, C5 in questo file) valgono anche lì.
8. **Registro**: stesse regole di questo file, voci con prefisso `[HIP]`, aggiornando anche "Stato attuale".

## Registro delle prove

Voci più recenti in alto. Formato: data, chi, commit o "non committato", cosa, numeri (riferimento → candidato), esito.

- **[HIP] 24/11/2026, Codex**, `f52affb46`: profilato il prefill di 512 token nuovi a profondità 32.768, 1 ripetizione. 212,25 t/s, 2.412,3 ms totali. Preparazione 140,2 ms, unpack 361,0, GEMM 1.450,4, prodotto codificato dei pochi token residui 5,6, epilogo 139,7; unpack + GEMM 75,1%. Tutte le 512 proiezioni della finestra misurata sono valide. Prima del run assenti altri processi di inferenza e ComfyUI; 15,77 GB liberi. Picco 12,43 GiB dedicati + 0,16 condivisi. Esito: vale una prova circoscritta di WMMA nativo gfx1201; l'obiettivo 550 t/s resta da verificare. Sorgenti PR #3 e DLL HIP di riferimento controllati dopo la cattura. Piano `NARIQ_HIP_OPTIMIZATION.md` e log `e1-profile/logs/profile-depth32768*` nella cartella HIP.

- **[HIP] 24/11/2026, Codex**, `f52affb46`: cattura con eventi HIP, senza grafi, 512 proiezioni Trellis per ripetizione. A 512 token 237,37 t/s con marker spenti → 234,00 con marker; mediana 2.188 ms, preparazione 136,9, unpack 361,4, GEMM 1.457,6, prodotto dei pochi token residui 5,7, uscita 142,1 ms. Forma TRL3 5120 → 17408: prodotto 0,353 ms a 1 token, 1,183 ms a 4; a 512 token unpack 1,359 ms e GEMM 5,177 ms. Il primo aggregatore ha rifiutato intervalli HIP negativi; corretto per registrare ed escludere le righe anomale (2 su 1.536 righe temporizzate a 4 token, zero a 1 e 512). Esito: prefill breve dominato da unpack + GEMM; verificare a 32K. Prima del run assenti altri processi GPU interessati, 15,77 GB liberi. Log `e1-profile/logs/profile-short*`.

- **[HIP] 24/11/2026, Codex**, `f52affb46`: build dei marker di fase riuscita. Il controllo con lo stesso binario diagnostico, marker e HIP Graphs spenti, dà prefill 8,19 / 11,20 / 237,37 t/s a 1 / 4 / 512 token (`-r 3`, Flash Attention, cache Q4, batch/microbatch 512). Nessun altro processo di inferenza o ComfyUI, 15,77 GB liberi prima del run. Picco 11,88 GiB dedicati. Esito: riferimento per quantificare l'effetto della cattura, con comando e log in `e1-profile` nella cartella HIP.

- **[HIP] 24/11/2026, Codex**, `f52affb46`: B3 finale con le DLL ROCm del venv verificate nel processo. A profondità 32.768 (`-d 32768`), `pp512` 472 → 213,77 t/s, `tg64` 23,5 → 7,98 t/s. Una ripetizione, cache Q4, Flash Attention attiva, batch/microbatch 512. Prima della prova assenti i processi concorrenti e 15,77 GB liberi. Picco WDDM 12,43 GiB dedicati e 0,16 GiB condivisi. Log `e1-logs/bench-rocm102-b3-app-local.*`. Esito E1: `tg64` breve 8,01 < 28 e `pp512` a 32K 213,77 < 550. HIP è sotto il riferimento Vulkan in entrambe le fasi e a entrambe le profondità. Fermare l'ottimizzazione HIP secondo il criterio E1 e chiedere all'utente come proseguire. La riserva HIP SDK 7.1 non è stata necessaria.

- **[HIP] 24/11/2026, Codex**, `f52affb46`: B2 finale con le DLL ROCm del venv app-local, percorsi verificati nel processo. `pp512` 602 → 224,14 t/s (dev. std. 0,64), `tg64` 24,4 → 8,01 t/s (dev. std. 0,14). Tre ripetizioni, cache Q4, Flash Attention, batch/microbatch 512. Prima della prova assenti i processi concorrenti e 15,77 GB liberi. Picco 11,87 GiB dedicati e 0,13 GiB condivisi. Esito: soglia di generazione 28 t/s mancata. Log `e1-logs/bench-rocm102-b2-app-local.*`.

- **[HIP] 24/11/2026, Codex**, `f52affb46`: verifica con DLL core app-local del venv. TRL1–6 e 380/380 prodotti superati. I logits conservano 31/31 top-1, stessi greedy e KL massima `6,12e-8` (microbatch 1), `2,02e-6` (32). I 36 lookup `TRELLIS_GET_ROWS` restano non supportati su HIP. Esito: correttezza della build finale confermata. Log e report `e1-logs/*app-local*`.

- **[HIP] 24/11/2026, Codex**, `f52affb46`: B3 iniziale con 15,77 GB liberi e nessun altro processo di inferenza. L'ispezione dei moduli mostra `hipblas.dll` e `rocblas.dll` dal venv, ma `amdhip64_7.dll` e `rocm_kpack.dll` da `System32`. Il runtime HIP ha versione file `10.0.3686.0` in entrambe le copie e SHA-256 diversi. Interrotto solo il processo HIP B3 dopo 65 s, prima di un risultato valido. Copiati `amdhip64_7.dll`, `rocm_kpack.dll`, `amd_comgr.dll` dal venv accanto ai binari HIP e verificati gli hash; `--list-devices` funziona. Esito: ripetere test e benchmark con DLL locali. Log `e1-logs/bench-rocm102-b3.*`.

- **[HIP] 24/11/2026, Codex**, `f52affb46`: B2, `pp512` 602 → 224,48 t/s (dev. std. 0,84), `tg64` 24,4 → 8,05 t/s (dev. std. 0,18). Tre ripetizioni, cache Q4, Flash Attention attiva, batch e microbatch 512. Prima della prova assenti altri `llama-server`, `llama-bench`, OpenClaw e ComfyUI; `hipInfo` 15,77 GB liberi. Picco del processo WDDM 11,87 GiB dedicati e 0,13 GiB condivisi. Esito: B2 completato, soglia di generazione 28 t/s mancata. Log `e1-logs/bench-rocm102-b2.*`.

- **[HIP] 24/11/2026, Codex**, `f52affb46`: logits del GGUF originale contro CPU con continuazione forzata, 31 posizioni. A microbatch 1 top-1 31/31, stessi token greedy, KL max `6,12e-8`. A microbatch 32 top-1 31/31, stessi token greedy, KL max `2,02e-6`. Entrambi passano la soglia `1e-3` del comparatore della PR. A microbatch 32 la KL è sopra l'obiettivo `1e-6` usato come traguardo Vulkan. Il modello carica 65/65 layer su ROCm0 (9.339,94 MiB) e mantiene 304,09 MiB in `CPU_Mapped` per l'embedding Trellis. Esito: qualificato per misurare questa build HIP, con il limite del lookup su CPU. Log `e1-logs/logits-*.log` e report `e1-logs/*-parity.json`.

- **[HIP] 24/11/2026, Codex**, `f52affb46`: `ROCm0` = RX 9070 `gfx1201`, wave32. CTest 2/2 e `test-trellis -v ROCm0` superati, incluso TRL1 con 35 token. `test-backend-ops` passa 380/380 prodotti `TRELLIS` eseguiti. I casi `TRELLIS_GET_ROWS` sono dichiarati non supportati ed esclusi dal conteggio. Esito: prodotti corretti, lookup Trellis privo di kernel HIP. Log `e1-logs/ctest-trellis.log`, `trellis-rocm0.log` e `backend-ops-rocm0.log`.

- **[HIP] 24/11/2026, Codex**, `f52affb46`: build Release riuscita per `gfx1201` con `HIP_DEVICE_LIB_PATH=<root SDK>\lib\llvm\amdgcn\bitcode`, nella sola shell. Compilati `llama-server`, `llama-bench`, `test-backend-ops`, `test-trellis`, `test-trellis-model` e `test-trellis-metadata`. Il sorgente della PR resta a `f52affb46`. Il GGUF coincide con la baseline Vulkan (SHA-256 `1eaad934aa05579b8eba47ed9b26450392f9875fc2354efe8cfa8b29be8f4419`). Esito: build pronta. Log `e1-logs/build-rocm102-bitcode.log`.

- **[HIP] 24/11/2026, Codex**, `f52affb46`: Ninja Release, AMD clang 24, `GGML_HIP=ON`, `GPU_TARGETS=gfx1201`, HIP e hipBLAS del venv trovati da CMake. La prima build si ferma in `acc.cu` perché clang non trova il bitcode ROCm. Verificato `lib/llvm/amdgcn/bitcode/oclc_isa_version_1201.bc` nel SDK isolato. Esito: ripetere con `HIP_DEVICE_LIB_PATH` impostata nella sola shell. Log `e1-logs/configure-rocm102.log` e `e1-logs/build-rocm102.log`.

- **[HIP] 24/11/2026, Codex**: `rocm-sdk init` riuscito. `hipInfo` del venv riconosce AMD Radeon RX 9070, `gfx1201`, 15,92 GB totali e 15,77 GB liberi al momento del controllo. Percorsi SDK e output in `e1-logs/sdk-paths.json` e `e1-logs/hipinfo.log`. Esito: runtime e scheda compatibili.

- **[HIP] 24/11/2026, Codex**: installazione pip riuscita nel venv isolato dal solo indice stabile AMD. Installati `rocm`, `rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-devel`, `rocm-sdk-device-gfx1201`, tutti a `10.2.0`. Log e report in `e1-logs/pip-install.log` e `e1-logs/pip-report.json` della cartella HIP. Esito: installazione completata.

- **[HIP] 24/11/2026, Codex**: preflight ROCm 10.2.0 Windows x64. Verificato l'indice stabile `https://stable.repo.amd.com/rocm/whl-next/` con richieste HEAD, per un totale di 1.949.538.247 byte: `rocm` 29.416, `rocm-sdk-core` 829.862.637, `rocm-sdk-libraries` 124.764.403, `rocm-sdk-devel` 672.529.822, `rocm-sdk-device-gfx1201` 322.351.969. Eventuale bootstrap separato, fino a 30.532 byte nell'indice. Python 3.12.10 x64 disponibile, HIP SDK 7.1 presente. Esito: utente ha confermato il download.
- **[HIP] 24/11/2026, Codex**, `f52affb46`: clonato Nari Labs (HEAD `8e0e016e0`) nella cartella separata e verificato `refs/pull/3/head` sul branch locale `hip-rdna3`. Il commit della PR mantiene in FP32 il prodotto Trellis in batch. Il GGUF da 10.123.626.592 byte sarà letto dal percorso assoluto nella cartella Vulkan. Esito: sorgenti pronti per la configurazione.

- **25/11/2026 notte, Claude**: D2, confronto completo in stile OpenClaw (contesto 114688, cache q4_0, ubatch 512, visione su CPU, DFlash2 3 token con rejection, prompt sintetici dello script: breve, 32K, 65K). NariQ `build/bin` (con `d4d3eb6e3`): gen 25,41 / 23,92 / 25,61 t/s, prefill 522,4 / 432,0, accettati 271/720, 130/373, 146/332, picco VRAM 14,962 GiB + 0,198 condivisa. Unsloth sul fork di produzione `d4b007d`: 86,96 / 90,03 / 78,31, prefill 910,8 / 661,1, accettati 291/661, 159/288, 154/303, 14,056 + 0,236. Ciclo di verifica NariQ 84-90 ms contro 27-32. Esito: con NariQ DFlash2 non aumenta la velocità; prefill al 57-65% di Unsloth; generazione circa 3,4 volte più lenta.
- **25/11/2026 notte, Claude**, `d4d3eb6e3`: memoria della bozza DFlash2 con NariQ. Il contesto della bozza ereditava ubatch 512 e il suo grafo calcola la testa per ogni token: con la testa Trellis (tutti i 248.320 logit in bande, poi riordinati) la riserva era 1437 MiB contro 160 con Unsloth. Selezionare le sole righe di output prima della testa non si può (`dflash.cpp` richiede i logit di tutte le posizioni: assert); ubatch della bozza limitato a 64: 213 MiB, picco VRAM 15,03 + 0,91 GiB condivisa → 14,56 + 0,17 (verifica preliminare dello script). Esito: accettato.

- **25/11/2026 notte, Claude**, `71b71362e`: forma del prodotto cooperativo. pp512 per righe × input per passo: 2 × 4 519 t/s, 4 × 4 483, **4 × 2 602-626**, 4 × 1 601, 8 × 1 566-569, 8 × 2 520 (prove alternate, varianti temporanee poi rimosse; 380/380 test TRELLIS per ognuna). Con 4 × 2 e k diviso per le proiezioni da 48 righe: pp512 485 → 602 (con residuo 353 → 452), a 32K 395 → 472, tg64 24,4. KL sul prompt 6,6e-8, top-1 identico. Prima, con 2 × 4, la sola divisione di k aveva dato pp512 ~481 → ~504. 416/416, ctest 4/4. Esito: accettato.

- **25/11/2026 notte, Claude**, `af07eaa13`: C3, nuovo prodotto cooperativo (`trellis_prep16.comp` + `trellis_matmul_cm.comp`). pp512 150,6 → 485-502 t/s; con il residuo 353; a 32K (`-d 32768`) 146,6 → 395,4; tg64 invariato (24,75). Perf 5120 → 17408 a 16 token: TRL3 1041 → 695 µs. Logits del modello completo con micro-batch da 32 token, sulle 25 posizioni del prompt: KL media 6,7e-8 (un termine FP16), 1,3e-8 (con residuo), 6,4e-9 (kernel precedente), top-1 identico. Un solo termine FP16 senza potenza di 2 per token non era stato provato; con un solo termine `test-trellis-vulkan-coopmat` fallisce di poco (TRL1: -10,3191 contro -10,3261, tolleranza 2e-4 × (1 + |atteso|)), quindi quel test usa la variante con residuo. 416/416 con e senza percorso cooperativo, ctest 4/4. Esito: accettato, percorso cooperativo predefinito.

- **25/11/2026 notte, Claude**, `7a5cae586`: prodotti int8 con meno di 64 gruppi di lavoro divisi lungo k su fino a 16 gruppi, con somma finale (`trellis_reduce.comp`). TRL5 5120 → 48: 1 token 29,8 → 7,1 µs, 4 token 77,0 → 9,5 µs. tg64 24,09 → 24,92; server con DFlash2 3 token 39,47 → 40,75 t/s (accettati 180/225 e 182/228), ciclo ~86,5 → ~82,6 ms. Il profilatore ora nomina i nodi Trellis con tipo, k, n e token. 416/416, ctest 4/4. Esito: accettato.
- **25/11/2026 notte, Claude**: B4, profilo di un ciclo DFlash2 (`GGML_VK_PERF_LOGGER=1`, operazioni in serie). Grafo di verifica a 4 token ~94 ms: prodotti Trellis delle proiezioni grandi (5120 → 17408 TRL3 15,0 ms, TRL2 10,9, 17408 → 5120 TRL3 10,1, ...) e 7,2 ms per le proiezioni da 48 righe; grafo della bozza ~10 ms, di cui ~6 nelle bande della testa Trellis condivisa. In isolamento: banda 5120 → 15488 TRL2 102 µs (1 token) e 197 µs (4 token), normale; 5120 → 48 TRL5 30 / 77 µs, anomalo. Esito: corretto il caso da 48 righe; il resto del ciclo è il costo intrinseco della decodifica.

- **25/11/2026 notte, Claude**: B1 sul server (`-c 32768 -ub 512`, cache q4_0, prompt sintetico ripetitivo da 1527 token, 256 token generati, `GGML_VK_TRELLIS_COOPMAT=1`): senza bozza 23,96 t/s; DFlash2 2 token 38,68 (accettati 168/176); 3 token 42,18 (186/210); 4 token 17,06 (197/232, batch 5 fuori dal percorso int8). Esito: usare 3 token; il ciclo costa ~87 ms, da profilare (B4).
- **25/11/2026 notte, Claude**, `3b8cc30e7`: B2, un gruppo di lavoro per 16 uscite e tutti i token del batch (fino a 4), con variante a 1 token (`trellis_matvec_q8`) e fino a 4 (`trellis_matvec_q8x4`). Nuovi casi perf 5120 → 17408 a 1/2/3/4/16 token. µs per prodotto: TRL3 1 token 111,5 → 112,5, 4 token 407,3 → 215,9; TRL4 105,7 → 107,2 e 386,6 → 217,3. Una prima versione con un solo kernel per tutti i casi rallentava il token singolo dell'11% (124 µs). tg64 24,06, 416/416, ctest 4/4. Esito: accettato.

- **24/11/2026 notte, Claude**, `a66f18956`: A5a, prodotto int8. Nuovi `trellis_quant.comp` (attivazioni int8 per gruppi di 16, valori replicati per lane + somma e scala) e `trellis_matvec_q8` in `trellis_product.comp`; usato per batch ≤ 4 se il dispositivo ha `integer_dot_product`, `GGML_VK_TRELLIS_Q8=0` torna al prodotto float. Terza area di scratch da 6 byte per input e token (`ggml_vk_trellis_quant_size`). tg64 18,08 → 24,12 t/s; pp512 invariato (150,5). Logits del modello completo, `test-trellis-model --ubatch 1 --predict 6` contro CPU: KL media 4,0e-5, massima 2,2e-4, top-1 31/31 (prodotto float: KL media 1,1e-8). Soglie `test-backend-ops` per t ≤ 4: prodotto e gruppo d'ingresso 5e-5 (misurato ~2e-5), MLP codificato 2e-4 (misurato fino a 8,8e-5); `test-trellis-vulkan*` girano con `GGML_VK_TRELLIS_Q8=0` (controlli elemento per elemento del formato). 416/416, ctest 4/4. Esito: accettato.

- **24/11/2026 sera, Claude** (commit della sessione, vedi `git log`): somma dei 4 byte di `trellis_grid` con spostamenti e somme invece della moltiplicazione per `0x01010101`. Risultato identico, test 416/416, `test-trellis` superato; nel codice macchina di `trellis_matvec` le moltiplicazioni a 32 bit per ciclo passano da 16 a 8. Velocità invariata (tg64 18,07 → 18,05). Esito: tenuta, ma conferma che non basta togliere le moltiplicazioni.
- **24/11/2026 sera, Claude**: prove a esclusione su tg64 (risultati volutamente sbagliati, interruttori temporanei poi rimossi). Normale 17,60 t/s; senza rotazione d'ingresso 18,29; senza prodotto 193,8; senza prodotto né rotazione 230,3; prodotto con decodifica banale (`float(state & 255)`) 35,0. Quindi per token: prodotto ~52 ms (decodifica ~28, letture ~23), resto 5,2 ms, rotazione d'ingresso ~1 ms. `trellis_matvec`: 35 registri, 1 KB di memoria condivisa, ciclo principale ~110 istruzioni per 8 pesi per lane, 4-6 letture a 32 bit dei codici e 2 da 64 bit delle attivazioni. Esito: priorità ad A5a.
- **24/11/2026 sera, Claude**: rotazioni usate dal modello (nuova diagnostica `GGML_VK_TRELLIS_DEBUG=1`, stampa ogni forma una volta): solo `kind=1` a blocchi, lunghezze 128 (larghezze 5120 e 15488), 256 (15616), 1024 (17408 e 1024), 2048 (6144 e 10240), 4096 (12288), sempre `m=1` e senza parti fuse, quindi sempre il kernel a blocchi in memoria condivisa. Esito: A2 scartata.
- **24/11/2026 sera, Claude**: baseline A1 con `build/ref-bin` (`343fe2183`), stessi argomenti di B2. NariQ pp512 153,2, tg64 17,45; Unsloth pp512 1150,2, tg64 39,30. A 32K (`-d 32768 -r 1`): NariQ 146,6 / 17,44, Unsloth 765,8 / 37,27.

- **24/11/2026, Claude**: perplexity WikiText-2, 40 finestre da 512 token, stessa build per i due modelli. NariQ 6,2455 ± 0,147, Unsloth 6,2317 ± 0,148; NariQ meglio in 19 finestre su 40. Alla pari, coerente con le KL di Nari Labs (la perplexity non misura la fedeltà al BF16). Esito: la qualità resta un motivo valido per il cambio.
