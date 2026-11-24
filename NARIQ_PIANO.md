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
| Generazione con DFlash2, contesto breve | ~81 t/s | ~10 t/s (~41 al 25/11, prompt ripetitivo) | 55 | 75 |
| Generazione con DFlash2 a 65K | ~80 t/s | n/d | 50 | 70 |
| Prefill a 32K | ~910 t/s (llama-bench d32768: 766) | ~137 t/s (472 al 25/11, llama-bench d32768) | 450 | 800 |
| VRAM di picco, OpenClaw completo | 14,1-14,5 GiB | n/d | ≤ 14,5 GiB | ≤ 14,5 GiB |
| Qualità | | | logits Vulkan = CPU (top-1 31/31, KL < 1e-6) | idem |

Il cambio in OpenClaw si propone all'utente quando sono raggiunti i traguardi minimi. Decide l'utente.

## Stato attuale

Aggiornare questa sezione a ogni passaggio di consegne.

- **Ultimo aggiornamento**: 25/11/2026 notte, Claude.
- **Build di riferimento**: `build/ref-bin` = binari del commit `71b71362e` (C3, versione 64 × 32). La build iniziale resta in `build/packaging-x64/bin` (`343fe2183`).
- **Ultimi risultati**: generazione senza bozza 17,5 → 24,9 t/s; con DFlash2 a 3 token ~41 t/s sul prompt ripetitivo di prova (prima ~10). **Prefill (C3) 150 → ~602 t/s a contesto vuoto, 147 → 472 a 32K** (llama-bench): traguardo minimo del prefill raggiunto. Il percorso cooperativo è ora predefinito (`GGML_VK_TRELLIS_COOPMAT=0` per disattivarlo): non serve più impostare la variabile.
- **Riferimento NVIDIA**: Nari Labs dichiara 88,9 t/s a 4K e 70,7 a 64K su una RTX PRO 6000 (Blackwell, ~1,8 TB/s, circa 2,8 volte la banda della RX 9070), senza speculazione, cache KV BF16. In proporzione alla banda equivarrebbero a ~32 t/s qui. Il GGUF pubblicato non contiene il modulo MTP ("non ancora quantizzato"), quindi MTP non è disponibile; le alternative alla bozza DFlash2 sono `ngram-simple` (nessun modello, utile su testo ripetitivo e codice) o nessuna speculazione.
- **Profilo di un ciclo DFlash2 (B4)**: il ciclo (~83 ms) è dominato dai prodotti Trellis a 4 token del modello (il prodotto a 4 token costa ~1,9 volte quello a 1 token: la decodifica di ogni peso, ~7 istruzioni, si fa una volta, ma ogni token aggiunge un `dot4` per peso e due letture per tile). La bozza costa ~10 ms per ciclo, ~6 nelle bande della testa di uscita Trellis condivisa (normali: 5120 → 15488 a 4 token 197 µs). Le proiezioni da 48 righe erano anomale (corrette in `7a5cae586`).
- **Valutazione onesta**: la decodifica Trellis costa ~7 istruzioni per peso su RDNA (una moltiplicazione intera a 32 bit, a un quarto di velocità, più estrazioni di bit), contro ~1-2 per i formati IQ/K di Unsloth. Il limite stimato in generazione è ~28-30 t/s senza bozza e ~45-50 con DFlash2 su testo ripetitivo; la parità con Unsloth (39 / ~80) non sembra raggiungibile con questo formato su questa scheda. Ho provato a evitare la moltiplicazione con due prodotti a 24 bit: il compilatore li ricompone in una moltiplicazione a 32 bit, e una versione che non si ricompone costa le stesse istruzioni.
- **Prefill e OpenClaw**: a 472 t/s un prompt da 32K richiede ~68 s (Unsloth ~40 s, prima ~3,5 minuti).
- **Scoperta chiave (A1)**: in generazione il prodotto Trellis (`trellis_matvec`) è il 90% del tempo: circa 52 ms su 57 per token. Tutto il resto (attenzione, strati ricorrenti, rotazioni, input/output) costa 5,2 ms. Del prodotto, circa 28 ms sono calcolo per decodificare i pesi (~14-15 istruzioni per peso) e circa 23 ms letture. Le rotazioni costano ~1-2 ms reali: il profilo per fase le gonfiava a 30 ms.
- **In corso**: niente. Codice committato e compilabile.
- **Prossimo passo**: D1-D2, cioè profilo `qwen38-nariq` nel launcher di OpenClaw (`D:\Progetti\openclaw\scripts\Start-OpenClaw-Local.ps1`, ora nessuna variabile Trellis da impostare: il percorso cooperativo è predefinito) e confronto completo con prompt reali, DFlash2 a 3 token e VRAM di picco, per decidere con l'utente. Margini ancora aperti sul prefill: blocchi di token più grandi (`VK_TRELLIS_BATCH` = 128: ogni peso si decodifica 4 volte per 512 token), decodifica in anticipo del passo successivo (doppio buffer in memoria condivisa).
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
| D2 | Confronto completo con `tools\Run-Qwen38-VulkanAB.ps1` e VRAM di picco con il contesto di produzione | da fare | |
| D3 | Sessione reale di Jarvis e analisi del log | da fare | |

### Fase E: alternative

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| E1 | Valutare un backend HIP (ROCm SDK 7.1 già installato in `C:\Program Files\AMD\ROCm\7.1`): compilare il fork Nari Labs con la PR HIP per RX 7900 XTX (nari-labs/nari-llama.cpp#3) per gfx1201 su Windows e misurare NariQ | da valutare con l'utente | Vantaggi possibili per i soli kernel Trellis: moltiplicazione a 24 bit esplicita (`__umul24`), 64 KB di memoria condivisa per gruppo, intrinseci per permutazioni e WMMA, kernel CUDA di Nari Labs adattabili. Svantaggi: tutto il resto del modello girerebbe su llama.cpp HIP, che su questa macchina era più lento di Vulkan in generazione (prove del 09/11 in OpenClaw), e le ottimizzazioni di questo fork (attenzione per il driver AMD, DFlash2, MTP, ecc.) non ci sono |

## Registro delle prove

Voci più recenti in alto. Formato: data, chi, commit o "non committato", cosa, numeri (riferimento → candidato), esito.

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
