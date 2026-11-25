# Piano di ottimizzazione NariQ su Vulkan (RX 9070)

Documento condiviso fra Claude e Codex. Chi lavora lo aggiorna **dopo ogni prova**, così l'altro può riprendere senza tempi morti. Il diario tecnico completo resta in [NARIQ_VULKAN_PROGRESS.md](NARIQ_VULKAN_PROGRESS.md). Questo file contiene obiettivo, stato, regole, attività e registro delle prove.

## Obiettivo

Usare Qwen3.8-27B NariQ 3.0 bpw in OpenClaw (Jarvis) al posto di Unsloth UD-Q2_K_XL, a patto che sia veloce quanto il setup attuale.

Motivo: secondo il rapporto tecnico di Nari Labs (arXiv 2611.26157, tabella 3), la divergenza KL dal BF16 di NariQ 3.0 è da 2,4 a 5 volte più bassa di quella di UD-Q2_K_XL (agente 0,0063 contro 0,0275; generale 0,0173 contro 0,0407). Sulla perplexity WikiText i due modelli sono alla pari (6,25 contro 6,23, misura del 24/11), che è atteso: la perplexity non misura la fedeltà al modello originale.

### Traguardi

Riferimento: setup OpenClaw attuale, fork `d4b007d` con Unsloth UD-Q2_K_XL, DFlash2 3 token, cache q4_0, contesto 114688, ubatch 512.

| Misura | Unsloth attuale | NariQ Vulkan, 25/11 | Traguardo minimo | Traguardo pieno |
| --- | ---: | ---: | ---: | ---: |
| Generazione senza bozza, llama-bench tg | ~39 t/s | ~25 t/s (B2) | 30 | 39 |
| Generazione con DFlash2, contesto breve | 87 t/s (D2, confronto precedente) | 37,23 t/s (D2) | 55 | 75 |
| Generazione con DFlash2 a 65K | 78 t/s (D2, confronto precedente) | 39,90 t/s (D2) | 50 | 70 |
| Prefill a 32K | ~910 t/s (llama-bench d32768: 766) | 498,95 t/s (B3 C8, 512 token con prefisso in cache) | 450 | 800 |
| VRAM di picco, OpenClaw completo | 14,1-14,5 GiB | 13,65 GiB D2 testuale con DFlash2. Profilo completo da verificare. | ≤ 14,5 GiB | ≤ 14,5 GiB |
| Qualità | | Logits identici al dot4. CPU 31/31 top-1, KL max 2,10e-4. | CPU top-1 31/31, KL < 1e-6 | idem |

Il cambio in OpenClaw si propone all'utente quando sono raggiunti i traguardi minimi. Decide l'utente.

## Stato attuale

Aggiornare questa sezione a ogni passaggio di consegne.

- **Ultimo aggiornamento**. 25/11/2026, Claude, F1-F3 (profilo, verifica a 5-16 token, lunghezza della bozza).
- **Build di riferimento**. Per C8 usare `build/ref-bin` e verificare il commit con `llama-server --version`. Hadamard con `GGML_VK_TRELLIS_BLOCK_SHUFFLE=1` e prefill 256 con `GGML_VK_TRELLIS_MM_WIDE=1`. C7 (`09b633ca2`) è archiviata in `build/c8/ref-before-c8` e C6 (`60b4d242d`) in `build/c7/ref-before-c7`.
- **Confronto iniziale in stile OpenClaw (D2, 25/11)**: `D:\Progetti\openclaw\tools\Run-Qwen38-VulkanAB.ps1 -Backends nariq-dflash3r,fork-dflash3r` con `OPENCLAW_AB_UB=512` (varianti `nariq` e `nariq-dflash3r` aggiunte allo script; la build NariQ si cambia con `OPENCLAW_AB_NARIQDIR`). Report `D:\Progetti\openclaw\runtime\benchmarks\20261124-212542-qwen38-backend-ab`. NariQ con DFlash2 3 token: generazione 25,4 / 23,9 / 25,6 t/s (breve / 32K / 65K), prefill 522 / 432, accettazione 35-44%, ciclo 84-90 ms, picco VRAM 14,96 GiB. Unsloth sul fork di produzione: 87,0 / 90,0 / 78,3, prefill 911 / 661, accettazione 44-55%, ciclo 27-32 ms, 14,06 GiB. **Su quella build NariQ DFlash2 rendeva poco**: un ciclo costa circa il doppio di un token (~41 ms) e produce ~2,1 token, quindi la generazione resta ~25 t/s come senza bozza. Senza bozza si risparmia anche ~1,3 GiB di VRAM.
- **Dove sta il margine in generazione**: (1) il prodotto a 4 token (verifica) costa ~1,9 volte quello a 1 token: se scendesse a ~1,2 volte, con l'accettazione misurata DFlash2 darebbe ~40 t/s; (2) il prodotto a 1 token (~36 ms per token) limitato dal calcolo della decodifica (vedi "Valutazione onesta"). Entrambi dipendono dal costo della decodifica per peso, per questo la valutazione HIP (E1) è utile.
- **Ultimi risultati**: generazione senza bozza 17,5 → 24,9 t/s; con DFlash2 a 3 token ~41 t/s sul prompt ripetitivo di prova (prima ~10). **Prefill (C3) 150 → ~602 t/s a contesto vuoto, 147 → 472 a 32K** (llama-bench): traguardo minimo del prefill raggiunto. Il percorso cooperativo è ora predefinito (`GGML_VK_TRELLIS_COOPMAT=0` per disattivarlo): non serve più impostare la variabile.
- **Riferimento NVIDIA**: Nari Labs dichiara 88,9 t/s a 4K e 70,7 a 64K su una RTX PRO 6000 (Blackwell, ~1,8 TB/s, circa 2,8 volte la banda della RX 9070), senza speculazione, cache KV BF16. In proporzione alla banda equivarrebbero a ~32 t/s qui. Il GGUF pubblicato non contiene il modulo MTP ("non ancora quantizzato"), quindi MTP non è disponibile; le alternative alla bozza DFlash2 sono `ngram-simple` (nessun modello, utile su testo ripetitivo e codice) o nessuna speculazione.
- **Profilo di un ciclo DFlash2 (B4)**: il ciclo (~83 ms) è dominato dai prodotti Trellis a 4 token del modello (il prodotto a 4 token costa ~1,9 volte quello a 1 token: la decodifica di ogni peso, ~7 istruzioni, si fa una volta, ma ogni token aggiunge un `dot4` per peso e due letture per tile). La bozza costa ~10 ms per ciclo, ~6 nelle bande della testa di uscita Trellis condivisa (normali: 5120 → 15488 a 4 token 197 µs). Le proiezioni da 48 righe erano anomale (corrette in `7a5cae586`).
- **Stima del decoder precedente a B6**: la decodifica Trellis costa ~7 istruzioni per peso su RDNA (una moltiplicazione intera a 32 bit, a un quarto di velocità, più estrazioni di bit), contro ~1-2 per i formati IQ/K di Unsloth. Il limite stimato in generazione è ~28-30 t/s senza bozza e ~45-50 con DFlash2 su testo ripetitivo; la parità con Unsloth (39 / ~80) non sembra raggiungibile con questo formato su questa scheda. Ho provato a evitare la moltiplicazione con due prodotti a 24 bit: il compilatore li ricompone in una moltiplicazione a 32 bit, e una versione che non si ricompone costa le stesse istruzioni.
- **Prefill e OpenClaw**: a 472 t/s un prompt da 32K richiede ~68 s (Unsloth ~40 s, prima ~3,5 minuti).
- **Scoperta chiave (A1)**: in generazione il prodotto Trellis (`trellis_matvec`) è il 90% del tempo: circa 52 ms su 57 per token. Tutto il resto (attenzione, strati ricorrenti, rotazioni, input/output) costa 5,2 ms. Del prodotto, circa 28 ms sono calcolo per decodificare i pesi (~14-15 istruzioni per peso) e circa 23 ms letture. Le rotazioni costano ~1-2 ms reali: il profilo per fase le gonfiava a 30 ms.
- **B6, idea 1 validata**: `GGML_VK_TRELLIS_Q8_PACKED=1` attiva il dot4 misto compatto a 2–4 token su dispositivi con accelerazione packed unsigned e mista. Un token usa il kernel originale. TRL3 a 4 token 217,93 → 191,84 µs, rapporto 4/1 da 1,81 a 1,61. TRL4 210,11 → 196,82 µs. `tg64` 24,98 → 24,96 t/s. DFlash2 B1 44,00 → 48,38 e D2 26,44/28,42/28,46 → 29,17/31,77/30,66 t/s (breve/32K/65K), stesso testo e stessa accettazione. ISA AMD, 416/416 Vulkan, CTest 4/4 e logits identici al percorso precedente. Il traguardo 1,3 resta aperto. HIP chiuso.
- **B6, idea 2 validata**. WMMA INT8 16×16×16 su RX 9070 con driver AMD 26.11.2, opt-in per batch da 2–4 token. A 4 token TRL3 195,03 → 147,65 µs (-24,3%) e TRL4 190,34 → 146,97 µs (-22,8%), rapporti 4/1 1,35 e 1,38. B1 51,31 → 65,13 t/s e D2 29,02/31,89/31,31 → 37,23/40,32/39,90 t/s (breve/32K/65K), stessi hash del testo e stessa accettazione. D2 testuale con 13,65 GiB dedicati e 0,10 condivisi di picco. Vulkan 416/416, CTest 4/4, logits bit identici al dot4, CPU top-1 31/31 e KL max 2,10e-4. Restano aperti 1,3 nel microbenchmark e le soglie finali OpenClaw.
- **A5b conclusa**: due letture dirette condivise fra gli otto stati della stessa lane per TRL1–4, solo sul kernel da un token. Microbenchmark TRL3 120,43 → 118,75 µs. `tg64` 24,04 → 24,25 t/s nel breve e 21,37 → 21,36 a 32K. Vulkan 416/416, CTest 4/4 e logits identici a B6. Accettata come modifica piccola e numericamente esatta, senza nuove pipeline. Prototipi shuffle archiviati in `build/a5b`.
- **B6, analisi di partenza (Claude, solo lettura del codice)**: in `trellis_matvec_q8x4` (`trellis_product.comp`, ramo `TRELLIS_MATVEC_Q8`) ogni peso costa, per ogni token, un `dot4` dei suoi 4 byte da 6 bit con l'attivazione replicata 4 volte; ogni lane legge per token e tile un `uvec4` di `reps` più un `uvec2` di `qmeta` (24 byte, uguali per le 8 lane con lo stesso `lane % 4`). A 4 token: 16 `dot4` ogni 4 pesi più 96 byte di letture per lane e tile. Idee in ordine:
  1. **Valore del peso in un byte**: la somma dei 4 byte da 6 bit va da 0 a 252, quindi sta in un byte senza segno. Calcolarla una volta per peso (`dotPacked4x8EXT` senza segno con `0x01010101`, una sola istruzione `v_dot4_u32_u8`, oppure le somme a spostamenti di `trellis_grid`), impacchettare i 4 valori di una lane in un `uint` e fare per token **un solo** `dot4` a segno misto (pesi senza segno, attivazioni int8; `v_dot4_i32_iu8` su RDNA3/4) con le 4 attivazioni int8 impacchettate senza replica. Il -126 resta come oggi, tramite la somma delle attivazioni in `qmeta`. A 4 token: 4 `dot4` (somme) + impacchettamento (~3 operazioni) + 4 `dot4` (token), invece di 16; letture per token e tile 4 + 8 byte invece di 24. Serve un nuovo formato di `trellis_quant.comp` (4 int8 per lane senza replica) e una variante del kernel; la variante a 1 token probabilmente resta migliore con il formato attuale (da misurare). Controllare nell'ISA (`GGML_VK_PIPELINE_ISA`) che il driver usi davvero l'istruzione a segno misto.
  2. Con il valore in un byte con segno (sottraendo 126, da -126 a 126), anche un prodotto cooperativo int8 (16 × 16 × 16, token riempiti fino a 16) diventa possibile per 2-16 token: utile anche per B5. Più lavoro, da valutare dopo la 1.
  3. A5b (letture dei codici condivise nel sottogruppo) resta separata.
  Misura: casi perf `test-backend-ops` 5120 → 17408 TRL3/TRL4 a 1 e 4 token (oggi 112,5 / 215,9 µs per TRL3), poi tg64 e il server DFlash2 3 token come in B1/D2. Obiettivo: 4 token ≤ ~1,3 volte 1 token.
- **[HIP] E1 conclusa**: su RX 9070 con ROCm 10.2.0, PR #3 `f52affb46` e DLL del venv, B2 224,14 / 8,01 t/s e B3 a profondità 32.768 213,77 / 7,98 t/s (prefill / generazione). CTest 2/2, TRL1–6 e 380/380 prodotti passati; modello 31/31 top-1 a microbatch 1 e 32. I 65/65 layer sono su ROCm0, l'embedding Trellis resta su CPU. Nessuna delle soglie E1 è raggiunta. Raccomandazione: fermare l'ottimizzazione HIP e sottoporre all'utente il prossimo passo.
- **[HIP] Profilo prefill concluso**: unpack + GEMM pesano l'83,1% nel prefill breve e il 75,1% a profondità 32.768. A 32K il prefill profilato è 212,25 t/s. L'intervento con più margine è un prototipo WMMA RDNA4 che sostituisca entrambe le fasi, con validazione e soglie E1. Per 550 t/s servirebbe circa 5,5 volte sul blocco a parità degli altri tempi. Piano in `D:\Progetti\NariQ HIP RDNA4\NARIQ_HIP_OPTIMIZATION.md`. Il prototipo di ottimizzazione resta da provare.
- **C6 validata**. INT8 a un termine 759 t/s su pp512 e 242/416 test. INT8 con residuo 537 t/s e 416/416. Il percorso accettato usa FP16 a 256 token per riusare più a lungo la decodifica dei pesi. pp512 615,46 → 696,46 (+13,2%), pp8192 588,70 → 667,32 (+13,4%), pp512 a profondità 32768 444,14 → 529,35 (+19,2%) t/s. Picco dedicato B3 11,86 → 11,84 GiB. WikiText-2 40 finestre PPL 6,2456 → 6,2456. In una finestra da 385 token contro CPU, con coda dopo il blocco da 256, stesso top-1 (100,000%) e stessa KL massima a sei decimali (0,000037). Vulkan 416/416 e CTest 4/4.
- **C7 validata**. Hadamard a blocchi con le prime cinque tappe nei sottogruppi da 32. Nel profilo il totale delle rotazioni va da 101,77 a 70,25 ms. pp512 713,00 → 725,61 (+1,8%), pp8192 674,01 → 705,90 (+4,7%), pp512 a profondità 32768 478,03 → 482,66 (+1,0%) t/s. Vulkan 416/416 e CTest 4/4. WikiText-2 40 finestre PPL 6,2456. A 385 token contro CPU, top-1 100% e KL massima 0,000037, come C6. La fusione di SwiGLU con la preparazione e Hadamard del down ha attivato 64/64 coppie, ma ha dato -1,35% su pp512 e +1,61% su pp8192. Il prototipo è archiviato in build/c7.
- **C8 validata**. Riuso delle due parole per otto stati TRL1-4 nel prodotto FP16. pp512 702,96 -> 730,60 t/s (+3,9 %), pp8192 683,90 -> 696,88 t/s (+1,9 %), pp512 a profondità 32768 478,68 -> 498,95 t/s (+4,2 %). tg64 25,04 -> 24,56 t/s (-1,9 %) nel breve e 21,70 -> 21,59 t/s (-0,5 %) a profondità 32768. Picco dedicato B3 11,84 -> 11,84 GiB. Vulkan 416/416 anche con residuo FP16, CTest 4/4. Confronto CPU con test-trellis-model: 31/31 top-1 e stessi token greedy, KL massima 0,00006238. Su 385 token, top-1 100% e KL massima 0,000037. WikiText-2, 40 finestre, PPL 6,2456. Nel profilo A-B-B-A, sostituire temporaneamente il calcolo del valore dei pesi a partire dallo stato con un valore sintetico, mantenendo la lettura e l'estrazione degli stati, ha ridotto il tempo del prodotto Trellis da 520,42 a 490,13 ms (-5,8%). La prova stima il costo di quella parte del decoder nel kernel attuale. Il prototipo diagnostico è archiviato in `build/c8/no-hash.patch`.
- **D2 su C8**. Prompt breve con prefill 109,03 / 70,66 t/s (1,40 / 2,17 s), generazione 104,45 / 50,50 t/s. 32K, cache fredda con prefill 878,22 / 580,40 t/s (36,60 / 55,38 s), generazione 100,26 / 47,87 t/s. Follow-up a 32K con prefill 44,64 / 27,97 t/s (1,19 / 1,89 s), generazione 81,28 / 46,95 t/s. 65K, cache fredda con prefill 752,03 / 531,11 t/s (86,63 / 122,66 s), generazione 84,24 / 44,00 t/s. Ordine Unsloth / NariQ C8. Picco VRAM dedicata del processo 12,74 / 13,60 GiB. Il follow-up ha riusato 32.141 / 32.141 token. Richieste chat testuali in streaming con documentazione e sorgenti reali del fork, un campione per caso. Stesse richieste verificate tramite hash. Log in `build/d2-c8`.
- **Prossimo passo** (25/11, Claude, dopo F1-F3). Generazione: la verifica a 4 token (147 µs) è decodifica (~107) più parte WMMA (~40); idee rimaste: accumulo int32 su più tile con una scala di attivazione per 64 input che includa la scala dei pesi (meno conversioni e fma, qualità da misurare), testa Trellis della bozza (~3,5 ms per ciclo). Prefill: F5 (scritture LDS a 32 bit con conversione magic bias, passo K a 4 tile). Le opzioni C6-C8 e B6 sono ancora opt-in: renderle predefinite prima di D1.
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
| A5b | Riuso delle parole Trellis fra le due finestre di una lane | fatto | Due letture dirette nel kernel da un token TRL1–4. `tg64` +0,9% nel breve, stabile a 32K. Shuffle scartato dopo A/B sulla RX 9070 |
| A5c | Moltiplicazione dello stato (`state * 0x83dcd12d`, 32 bit, un quarto di velocità su RDNA): cercare una forma che il compilatore non ricomponga | da fare | Spezzarla in due prodotti a 16 bit non serve: il compilatore li ricompone in una sola moltiplicazione a 32 bit |

### Fase B: verifica DFlash2 (2-4 colonne)

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| B1 | Misurare DFlash2 con 3 token anticipati su NariQ | fatto | Dopo B2: senza bozza 24,0, bozza 2/3/4 token 38,7 / 42,2 / 17,1 t/s (prompt ripetitivo). Ciclo ~87 ms |
| B4 | Profilo di un ciclo DFlash2 (modello a 4 token, bozza, teste di uscita) | fatto | Vedi "Stato attuale" e il registro del 25/11. Corrette le proiezioni da 48 righe (`7a5cae586`) |
| B5 | Estendere la verifica a 5–8 token (4–7 token draft) | da fare | Riutilizzare la tile dei pesi decodificata in due prodotti WMMA e conservare la stessa somma FP32 per gruppo. Confrontare separatamente forma 5120 → 17408 e D2, partendo da 4 token draft. |
| B2 | Prodotto a più colonne: decodificare ogni peso una volta per tutte le colonne | fatto (`3b8cc30e7`) | 4 token: TRL3 407 → 216 µs; 1 token invariato grazie a una variante dedicata | Con A5a ogni colonna in più costa un solo `dot4` per peso. Oggi `trellis_matvec` (batch ≤ 4) lancia un gruppo per token (`gl_WorkGroupID.y`) e decodifica i pesi una volta per token |
| B3 | Scegliere i token anticipati (2-4) in base ai tempi misurati | da fare | |
| B6 | Verifica a 4 token più economica | Idee 1 e 2 validate, opt-in | Rapporti 4/1 1,35 e 1,38. D2 37,23/40,32/39,90 t/s. Il traguardo 1,3 resta aperto. |

### Fase C: prefill

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| C1 | Profilo del prefill a 512 token per fase | da fare | Il 24/11: prodotto 89% del tempo |
| C2 | Preparare `act_hi`/`act_lo` una volta per proiezione invece che per gruppo | fatto in C3 | `trellis_prep16.comp`; il residuo è opzionale (`GGML_VK_TRELLIS_MM_RESIDUAL=1`, 28% più lento, usato dal test elemento per elemento) |
| C3 | Prodotto cooperativo con tile più grandi entro i 32 KB di memoria condivisa del driver Windows; decodifica dei pesi una volta per tile | fatto (`af07eaa13`) | 32 uscite × 128 token per gruppo, 256 thread che decodificano 8 tile ogni 64 input, attivazioni FP16 preparate una volta per pezzo con una potenza di 2 per token. pp512 150 → ~490, 32K 147 → 395 |
| C5 | Prodotto cooperativo: forma del gruppo di lavoro | fatto (`71b71362e`) | 64 righe × 32 input per passo, migliore di 2×4, 4×4, 4×1, 8×1, 8×2 (vedi registro); k diviso per le proiezioni da 48 righe |
| C4 | Attenzione del prefill: verificare che la cache q4_0 usi la conversione f16 (`74856e4`) anche con NariQ | da fare | Comune ai due modelli |
| C6 | Valutazione prefill INT8 e riuso pesi FP16 | Completato, FP16 a 256 opt-in | INT8 a un termine fuori tolleranza e a due termini più lento. pp512 615,46 → 696,46 (+13,2%), pp8192 588,70 → 667,32 (+13,4%), pp512 a profondità 32768 444,14 → 529,35 (+19,2%) t/s. |
| C7 | Profilo e ottimizzazione di SwiGLU/Hadamard | Hadamard opt-in validata | pp512 713,00 → 725,61 (+1,8%), pp8192 674,01 → 705,90 (+4,7%), pp512 a profondità 32768 478,03 → 482,66 (+1,0%) t/s. Fusione SwiGLU archiviata dopo A-B-B-A. |
| C8 | Profilare e ottimizzare la decodifica dei pesi nel prefill | Validato | Riuso esatto TRL1-4. pp512 702,96 -> 730,60 t/s (+3,9 %). pp8192 683,90 -> 696,88 t/s (+1,9 %). pp512 a profondità 32768 478,68 -> 498,95 t/s (+4,2 %). |

### Fase D: integrazione in OpenClaw

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| D1 | Profilo `qwen38-nariq` in `D:\Progetti\openclaw\scripts\Start-OpenClaw-Local.ps1` (modello, mmproj NariQ su CPU, DFlash2) e variante nello script di benchmark | da fare | Solo dopo i traguardi minimi |
| D2 | Confronto completo con `tools\Run-Qwen38-VulkanAB.ps1` e VRAM di picco con il contesto di produzione | fatto (25/11) | Vedi "Stato attuale" e il registro |
| D3 | Sessione reale di Jarvis e analisi del log | da fare | |

### Fase F: dopo C8 (ordine concordato con l'utente il 25/11)

Priorità: prima la generazione, poi il prefill. TurboQuant/Turbo3, contesti 128K-256K e riduzione della VRAM sono rimandati. Non servono le opzioni dei kernel standard (tile XL `GGML_VK_AMD_PROP_XL`, split-K, RM): i prodotti Trellis non le usano.

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| F1 | Profilo C8 con tutte le opzioni attive: ciclo DFlash2 a 3 token a contesto breve e a 32K, prefill a 32K. Ripartizione fra Trellis 1 / 4 token, attenzione (percorsi FA del fork), delta-net, bozza e testa della bozza | fatto (Claude, 25/11) | Verifica a 4 token: prodotti Trellis ~91% del grafo; bozza ~11% del ciclo (metà nella testa Trellis). Attenzione e delta-net ~1% anche a 32K. Vedi registro |
| F2 | Senza codice: `--spec-draft-n-max` 4-5 e `--spec-draft-p-min` con n-max 3 | fatto (Claude, 25/11), con F3 | A temperatura 1,0 nessuna configurazione batte n-max 3 senza p-min; vedi registro |
| F3 | B5: verifica WMMA int8 estesa a 5-8 token | fatto (Claude, 25/11), opt-in con `GGML_VK_TRELLIS_VERIFY_CM=1` | `trellis_verify16_cm.comp` per 5-16 token: TRL3 5120 → 17408 a 16 token 425 → 175 µs. Utile con accettazione alta (greedy +17%), neutro a temperatura 1,0 |
| F4 | In base al profilo: testa di uscita della bozza o decodifica a 1 token | da fare | |
| F5 | Prefill: scritture LDS a 32 bit per coppie FP16 con conversione "magic bias" (somma dei byte OR 0x6400 in FP16, una sottrazione impacchettata per due pesi), poi passo K a 4 tile | da fare | |

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

- **25/11/2026, Claude**, non committato → commit di questa voce: F1-F3.
  - **Prefill con la bozza e senza cache dei prompt**: con `--cache-ram 0` e `cache_prompt: false` (il mio script di prova) il prefill a 32K con DFlash2 crolla a 255 t/s (625 senza bozza); con `--cache-ram 4096 --ctx-checkpoints 12` e `cache_prompt` attivo, come OpenClaw, 595 t/s. Non riguarda la produzione; causa non indagata.
  - **F1, profilo** (`GGML_VK_PERF_LOGGER`, operazioni in serie, quindi tempi gonfiati; C8 con tutte le opzioni). Breve: grafo di verifica a 4 token 64,6 ms, di cui prodotti Trellis ~59 (5120 → 17408 23,1; 17408 → 5120 14,4; 5120 → 10240 5,9; 6144 → 5120 5,8; ...), testa del modello 2,1, attenzione 0,9, delta-net 0,7. Grafo della bozza 7,3 ms: testa Trellis condivisa 3,5 (bande 5120 → 15488 / 15616 a 4 token), strati propri della bozza 2,6. A 32K: verifica 63,6 ms, attenzione 0,7. Conclusione: il margine è tutto nel prodotto Trellis a 4 token e nella decodifica.
  - **Kernel di verifica, prove** (TRL3 5120 → 17408 a 4 token, riferimento 147 µs): valore del peso impacchettato con due `dot4` (maschera `0xfcfcfcfc`, costanti `0x01010101` e `0x40404040`, XOR `0x80808080`) e una colonna per token: 150 µs, nonostante ~25 istruzioni in meno ogni 8 pesi nell'ISA; più lettura anticipata del tile successivo 157; due blocchi di righe per sottogruppo 155; moltiplicazione spezzata in due prodotti a 16 bit con costanti opache 165 (il compilatore usa comunque `v_mul_lo_u32`). Esclusioni: senza WMMA 123, senza decodifica 118, senza letture delle attivazioni 135. Il kernel a 1 token resta 107 µs. Conclusione: limite di istruzioni della decodifica (~107 µs) più ~40 µs della parte WMMA (conversioni, fma per tile, memoria condivisa); le varianti provate non lo riducono.
  - **F3, B5**: la colonna per token permette 16 token con un solo prodotto 16 × 16 × 16. Nuovo `trellis_verify16_cm.comp` per 5-16 token (2-4 restano su `trellis_verify_cm.comp` di Codex), stessa opzione `GGML_VK_TRELLIS_VERIFY_CM=1`. TRL3 a 16 token 425 → 175 µs; 2-4 token invariati (147). 416/416 con e senza opzioni, ctest 4/4; logits contro CPU a micro-batch 4 / 8 / 16: top-1 31/31, KL massima 2,1e-4 / 4,4e-4 / 9,3e-5 (soglia 1e-3). Tolleranza di `test-backend-ops` per t ≤ 16: 5e-5 (prodotto), 2e-4 (MLP).
  - **F2, server** (richiesta chat reale: riassunto di `NARIQ_VULKAN_PROGRESS.md`, 2502 token di prompt, 384-512 generati, opzioni C8, cache come OpenClaw). Greedy, un run per configurazione: riferimento n-max 3 45,9 t/s; nuovo n-max 4 49,3, 6 53,7, 7 53,8. Temperatura 1,0, media su 3 seed: riferimento n-max 3 44,6 (716/1305 accettati), n-max 7 con p-min 0,5 43,2 (753/1492), con p-min 0,7 38,5 (662/909). A 32K, temperatura 1,0: riferimento 43,9, n-max 7 40,9. Costo del ciclo: 61,7 ms a 3 token, 66,3 a 7 (+7,5%), ma a temperatura 1,0 i token accettati per ciclo quasi non crescono. Con n-max 5-7 il prefill della richiesta breve scende da ~480 a ~390-430 t/s (non indagato). Esito: B5 tenuto come opzione per testo molto prevedibile (codice, greedy); per OpenClaw a temperatura 1,0 resta n-max 3.

- **25/11/2026, Codex**, D2 C8 confronto finale. Prompt breve con prefill 109,03 / 70,66 t/s (1,40 / 2,17 s), generazione 104,45 / 50,50 t/s. 32K, cache fredda con prefill 878,22 / 580,40 t/s (36,60 / 55,38 s), generazione 100,26 / 47,87 t/s. Follow-up a 32K con prefill 44,64 / 27,97 t/s (1,19 / 1,89 s), generazione 81,28 / 46,95 t/s. 65K, cache fredda con prefill 752,03 / 531,11 t/s (86,63 / 122,66 s), generazione 84,24 / 44,00 t/s. Ordine Unsloth / NariQ C8. Picco VRAM dedicata del processo 12,74 / 13,60 GiB. Il follow-up ha riusato 32.141 / 32.141 token. Richieste chat testuali in streaming con documentazione e sorgenti reali del fork, un campione per caso. Stesse richieste verificate tramite hash. Log in `build/d2-c8`.

- **25/11/2026, Codex**, D2 C8 production-answer server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.585 GiB. Log build/d2-c8/production-answer.

- **25/11/2026, Codex**, D2 C8 production-answer-reply. Prefill 95 token a 76.93 t/s in 1.23 s, cache 0. Generazione 1024 token a 99.40 t/s, accettazione DFlash2 56.6%. Primo token 1.25 s, primo testo 11.55 s. VRAM libera stimata al preflight 2.325 GiB con il solo server di prova. Log build/d2-c8/production-answer-reply.json.

- **25/11/2026, Codex**, D2 C8 nariq-answer server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.589 GiB. Log build/d2-c8/nariq-answer.

- **25/11/2026, Codex**, D2 C8 nariq-answer-reply. Prefill 95 token a 62.11 t/s in 1.53 s, cache 0. Generazione 1024 token a 48.39 t/s, accettazione DFlash2 62.7%. Primo token 1.53 s, primo testo 19.26 s. VRAM libera stimata al preflight 1.393 GiB con il solo server di prova. Log build/d2-c8/nariq-answer-reply.json.

- **25/11/2026, Codex**, D2 C8 nariq-answer retry. Il primo tentativo e uscito prima di avviare il server perche il wrapper PowerShell ha passato il flag --reply-only come caratteri separati. Corretto il passaggio del flag come array.

- **25/11/2026, Codex**, D2 C8 nariq-answer server. Suite terminata con exit 2. VRAM stimata libera prima dell'avvio 14.592 GiB. Log build/d2-c8/nariq-answer.

- **25/11/2026, Codex**, D2 C8 nariq server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.58 GiB. Log build/d2-c8/nariq.

- **25/11/2026, Codex**, D2 C8 nariq-cold65k. Prefill 65145 token a 531.11 t/s in 122.66 s, cache 0. Generazione 384 token a 44.00 t/s, accettazione DFlash2 64.4%. Primo token 122.73 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.117 GiB con il solo server di prova. Log build/d2-c8/nariq-cold65k.json.

- **25/11/2026, Codex**, D2 C8 nariq-warm32k. Prefill 53 token a 27.97 t/s in 1.89 s, cache 32141. Generazione 256 token a 46.95 t/s, accettazione DFlash2 72.0%. Primo token 1.93 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.119 GiB con il solo server di prova. Log build/d2-c8/nariq-warm32k.json.

- **25/11/2026, Codex**, D2 C8 nariq-cold32k. Prefill 32145 token a 580.40 t/s in 55.38 s, cache 0. Generazione 384 token a 47.87 t/s, accettazione DFlash2 70.5%. Primo token 55.92 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.265 GiB con il solo server di prova. Log build/d2-c8/nariq-cold32k.json.

- **25/11/2026, Codex**, D2 C8 nariq-short. Prefill 153 token a 70.66 t/s in 2.17 s, cache 0. Generazione 384 token a 50.50 t/s, accettazione DFlash2 67.2%. Primo token 2.17 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.382 GiB con il solo server di prova. Log build/d2-c8/nariq-short.json.

- **25/11/2026, Codex**, D2 C8 production server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.579 GiB. Log build/d2-c8/production.

- **25/11/2026, Codex**, D2 C8 production-cold65k. Prefill 65145 token a 752.03 t/s in 86.63 s, cache 0. Generazione 384 token a 84.24 t/s, accettazione DFlash2 63.4%. Primo token 86.71 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.972 GiB con il solo server di prova. Log build/d2-c8/production-cold65k.json.

- **25/11/2026, Codex**, D2 C8 production-warm32k. Prefill 53 token a 44.64 t/s in 1.19 s, cache 32141. Generazione 256 token a 81.28 t/s, accettazione DFlash2 51.5%. Primo token 1.24 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.978 GiB con il solo server di prova. Log build/d2-c8/production-warm32k.json.

- **25/11/2026, Codex**, D2 C8 production-cold32k. Prefill 32145 token a 878.22 t/s in 36.60 s, cache 0. Generazione 384 token a 100.26 t/s, accettazione DFlash2 70.5%. Primo token 36.96 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 2.218 GiB con il solo server di prova. Log build/d2-c8/production-cold32k.json.

- **25/11/2026, Codex**, D2 C8 production-short. Prefill 153 token a 109.03 t/s in 1.40 s, cache 0. Generazione 384 token a 104.45 t/s, accettazione DFlash2 61.5%. Primo token 1.42 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 2.315 GiB con il solo server di prova. Log build/d2-c8/production-short.json.

- **25/11/2026, Codex**, C8 finale. Riuso delle due parole per otto stati TRL1-4 nel prodotto FP16. pp512 702,96 -> 730,60 t/s (+3,9 %), pp8192 683,90 -> 696,88 t/s (+1,9 %), pp512 a profondità 32768 478,68 -> 498,95 t/s (+4,2 %). tg64 25,04 -> 24,56 t/s (-1,9 %) nel breve e 21,70 -> 21,59 t/s (-0,5 %) a profondità 32768. Picco dedicato B3 11,84 -> 11,84 GiB. Vulkan 416/416 anche con residuo FP16, CTest 4/4. Confronto CPU con test-trellis-model: 31/31 top-1 e stessi token greedy, KL massima 0,00006238. Su 385 token, top-1 100% e KL massima 0,000037. WikiText-2, 40 finestre, PPL 6,2456.

- **25/11/2026, Codex**, C8 deep-a-reuse-2. pp512/tg0 d32768 487.54 t/s, pp0/tg64 d32768 21.98 t/s Exit 0. Processi concorrenti assenti e 14.444 GiB stimati liberi. Log build/c8/deep-a-reuse-2.

- **25/11/2026, Codex**, C8 deep-b-reuse-2. pp512/tg0 d32768 494.75 t/s, pp0/tg64 d32768 21.38 t/s Exit 0. Processi concorrenti assenti e 14.476 GiB stimati liberi. Log build/c8/deep-b-reuse-2.

- **25/11/2026, Codex**, C8 deep-b-reuse-1. pp512/tg0 d32768 503.15 t/s, pp0/tg64 d32768 21.8 t/s Exit 0. Processi concorrenti assenti e 14.475 GiB stimati liberi. Log build/c8/deep-b-reuse-1.

- **25/11/2026, Codex**, C8 deep-a-reuse-1. pp512/tg0 d32768 469.81 t/s, pp0/tg64 d32768 21.42 t/s Exit 0. Processi concorrenti assenti e 14.444 GiB stimati liberi. Log build/c8/deep-a-reuse-1.

- **25/11/2026, Codex**, C8 ops-b-reuse-residual.   416/416 tests passed Exit 0. Processi concorrenti assenti e 14.474 GiB stimati liberi. Log build/c8/ops-b-reuse-residual.

- **25/11/2026, Codex**, C8 logits-model-reuse. Top-1 31/31, KL max 6.23817688615544E-05, stessi token greedy True. Exit 0. Processi concorrenti assenti e 14.47 GiB stimati liberi. Log build/c8/logits-model-reuse.

- **25/11/2026, Codex**, C8 sintesi-reuse-8k. A-B-B-A pp8192: 683,90 -> 696,88 t/s (+1,9%). Entrambe le misure candidate superano entrambi i riferimenti. WikiText-2 40 finestre PPL 6,2456. A 385 token contro CPU KL max 0,000037 e top-1 100%, come C7. Prossima misura B3 a profondita 32768.

- **25/11/2026, Codex**, C8 pp8192-a-reuse-2. pp8192/tg0 d0 679.81 t/s Exit 0. Processi concorrenti assenti e 14.475 GiB stimati liberi. Log build/c8/pp8192-a-reuse-2.

- **25/11/2026, Codex**, C8 pp8192-b-reuse-2. pp8192/tg0 d0 698.2 t/s Exit 0. Processi concorrenti assenti e 14.482 GiB stimati liberi. Log build/c8/pp8192-b-reuse-2.

- **25/11/2026, Codex**, C8 pp8192-b-reuse-1. pp8192/tg0 d0 695.56 t/s Exit 0. Processi concorrenti assenti e 14.476 GiB stimati liberi. Log build/c8/pp8192-b-reuse-1.

- **25/11/2026, Codex**, C8 pp8192-a-reuse-1. pp8192/tg0 d0 687.98 t/s Exit 0. Processi concorrenti assenti e 14.476 GiB stimati liberi. Log build/c8/pp8192-a-reuse-1.

- **25/11/2026, Codex**, C8 ppl-b-reuse. 0.07.672.210 I perplexity: 1.24 seconds per pass - ETA 0.37.413.864 I Final estimate: PPL = 6.2456 +/- 0.14665 Exit 0. Processi concorrenti assenti e 14.47 GiB stimati liberi. Log build/c8/ppl-b-reuse.

- **25/11/2026, Codex**, C8 kld-b-reuse. Maximum KLD:   0.000037 | Same top p: 100.000 ± 0.000 % Exit 0. Processi concorrenti assenti e 14.471 GiB stimati liberi. Log build/c8/kld-b-reuse.

- **25/11/2026, Codex**, C8 sintesi-reuse-breve. Con riuso esatto TRL1-4, Vulkan 416/416, CTest 4/4. A-B-B-A pp512 702,96 -> 730,60 t/s (+3,9%), tg64 25,04 -> 24,56 t/s. In corso le verifiche sul modello e a contesto esteso.

- **25/11/2026, Codex**, C8 sintesi-diagnostica. Profilo A-B-B-A caldo, hash originale 520,42 ms di prodotto Trellis in media, hash sostituito con valori sintetici 490,13 ms (-5,8%). Il diagnostico mantiene letture ed estrazione degli stati, ma i pesi risultanti non sono validi. Shader diagnostico assente dalla build candidata. Archivio build/c8/no-hash.patch.

- **25/11/2026, Codex**, C8 short-a-reuse-2. pp512/tg0 d0 706.28 t/s, pp0/tg64 d0 25.38 t/s Exit 0. Processi concorrenti assenti e 14.473 GiB stimati liberi. Log build/c8/short-a-reuse-2.

- **25/11/2026, Codex**, C8 short-b-reuse-2. pp512/tg0 d0 735.61 t/s, pp0/tg64 d0 24.56 t/s Exit 0. Processi concorrenti assenti e 14.483 GiB stimati liberi. Log build/c8/short-b-reuse-2.

- **25/11/2026, Codex**, C8 short-b-reuse-1. pp512/tg0 d0 725.59 t/s, pp0/tg64 d0 24.56 t/s Exit 0. Processi concorrenti assenti e 14.476 GiB stimati liberi. Log build/c8/short-b-reuse-1.

- **25/11/2026, Codex**, C8 short-a-reuse-1. pp512/tg0 d0 699.64 t/s, pp0/tg64 d0 24.7 t/s Exit 0. Processi concorrenti assenti e 14.476 GiB stimati liberi. Log build/c8/short-a-reuse-1.

- **25/11/2026, Codex**, C8 ctest-b-reuse. 100% tests passed, 0 tests failed out of 4 Exit 0. Processi concorrenti assenti e 14.471 GiB stimati liberi. Log build/c8/ctest-b-reuse.

- **25/11/2026, Codex**, C8 ops-b-reuse.   416/416 tests passed Exit 0. Processi concorrenti assenti e 14.471 GiB stimati liberi. Log build/c8/ops-b-reuse.

- **25/11/2026, Codex**, C8 build-reuse. Build Windows con decoder esatto e riuso delle due parole per TRL1-4. Exit 0. Log build/c8/build-reuse.log.

- **25/11/2026, Codex**, C8 spirv-reuse. SPIR-V per i tre prodotti FP16 (128, 256, residuo) compilato e validato con il riuso delle parole. Log build/c8/reuse-*.spv.

- **25/11/2026, Codex**, C8 profile-short-a2. Trellis GPU us: input=18748 rotate_in=50440.4 product=506040 rotate_out=27713.8 output=21440.4 decode_rows=30.76 store_rows=17.6 Profilo C6 wide, C7 attiva con pp512, exit 0. Nessun processo concorrente e 14.469 GiB di VRAM stimati liberi. Log build/c8/profile-short-a2.

- **25/11/2026, Codex**, C8 profile-short-diagnostic-b2. Trellis GPU us: input=18711.6 rotate_in=50454.9 product=489802 rotate_out=27429 output=21464.8 decode_rows=30.48 store_rows=17.88 Profilo C6 wide, C7 attiva con pp512, exit 0. Nessun processo concorrente e 14.44 GiB di VRAM stimati liberi. Log build/c8/profile-short-diagnostic-b2.

- **25/11/2026, Codex**, C8 profile-short-diagnostic-b1. Trellis GPU us: input=18756.6 rotate_in=50165.1 product=490462 rotate_out=27677.3 output=21489.3 decode_rows=57.08 store_rows=35.76 Profilo C6 wide, C7 attiva con pp512, exit 0. Nessun processo concorrente e 14.461 GiB di VRAM stimati liberi. Log build/c8/profile-short-diagnostic-b1.

- **25/11/2026, Codex**, C8 build-diagnostic. Build Windows del solo diagnostico senza hash dei pesi (output sintetico). Exit 0. Log build/c8/build-diagnostic.log.

- **25/11/2026, Codex**, C8 profile-short-a1. Trellis GPU us: input=20843.1 rotate_in=50459.7 product=534791 rotate_out=29423.6 output=21244.3 decode_rows=31.2 store_rows=17.36 Profilo C6 wide, C7 attiva con pp512, exit 0. Nessun processo concorrente e 14.473 GiB di VRAM stimati liberi. Log build/c8/profile-short-a1.

- **25/11/2026, Codex**, C7 finale. pp512 713,00 → 725,61 (+1,8%), pp8192 674,01 → 705,90 (+4,7%), pp512 a profondità 32768 478,03 → 482,66 (+1,0%) t/s. tg64 25,13 → 25,33 nel breve e 21,55 → 22,20 a profondità 32768. Vulkan 416/416 e CTest 4/4. WikiText-2 40 finestre PPL 6,2456. A 385 token contro CPU, top-1 100% e KL massima 0,000037, come C6. Il log ISA conferma lo shader `trellis_block_shuffle` sul modello. Esito: accettato come opt-in. Log `build/c7/*-final-*` e `ppl-b-final.*`.

- **25/11/2026, Codex**, C7 ppl-b-final. 0.07.532.481 I perplexity: 1.08 seconds per pass - ETA 0.37.487.347 I Final estimate: PPL = 6.2456 +/- 0.14665 Exit 0. Processi concorrenti assenti e 14.473 GiB stimati liberi. Log build/c7/ppl-b-final.

- **25/11/2026, Codex**, C7 deep-a-final-2. pp512/tg0 d32768 476.58 t/s, pp0/tg64 d32768 21.56 t/s Exit 0. Processi concorrenti assenti e 14.399 GiB stimati liberi. Log build/c7/deep-a-final-2.

- **25/11/2026, Codex**, C7 deep-b-final-2. pp512/tg0 d32768 490.17 t/s, pp0/tg64 d32768 22.14 t/s Exit 0. Processi concorrenti assenti e 14.499 GiB stimati liberi. Log build/c7/deep-b-final-2.

- **25/11/2026, Codex**, C7 deep-b-final-1. pp512/tg0 d32768 475.15 t/s, pp0/tg64 d32768 22.26 t/s Exit 0. Processi concorrenti assenti e 14.67 GiB stimati liberi. Log build/c7/deep-b-final-1.

- **25/11/2026, Codex**, C7 deep-a-final-1. pp512/tg0 d32768 479.49 t/s, pp0/tg64 d32768 21.54 t/s Exit 0. Processi concorrenti assenti e 14.674 GiB stimati liberi. Log build/c7/deep-a-final-1.

- **25/11/2026, Codex**, C7 pp8192-a-final-2. pp8192/tg0 d0 673.91 t/s Exit 0. Processi concorrenti assenti e 14.674 GiB stimati liberi. Log build/c7/pp8192-a-final-2.

- **25/11/2026, Codex**, C7 pp8192-b-final-2. pp8192/tg0 d0 717.59 t/s Exit 0. Processi concorrenti assenti e 14.674 GiB stimati liberi. Log build/c7/pp8192-b-final-2.

- **25/11/2026, Codex**, C7 pp8192-b-final-1. pp8192/tg0 d0 694.21 t/s Exit 0. Processi concorrenti assenti e 14.67 GiB stimati liberi. Log build/c7/pp8192-b-final-1.

- **25/11/2026, Codex**, C7 pp8192-a-final-1. pp8192/tg0 d0 674.11 t/s Exit 0. Processi concorrenti assenti e 14.673 GiB stimati liberi. Log build/c7/pp8192-a-final-1.

- **25/11/2026, Codex**, C7 short-a-final-2. pp512/tg0 d0 722.16 t/s, pp0/tg64 d0 25.48 t/s Exit 0. Processi concorrenti assenti e 14.666 GiB stimati liberi. Log build/c7/short-a-final-2.

- **25/11/2026, Codex**, C7 short-b-final-2. pp512/tg0 d0 745.95 t/s, pp0/tg64 d0 25.33 t/s Exit 0. Processi concorrenti assenti e 14.669 GiB stimati liberi. Log build/c7/short-b-final-2.

- **25/11/2026, Codex**, C7 short-b-final-1. pp512/tg0 d0 705.26 t/s, pp0/tg64 d0 25.33 t/s Exit 0. Processi concorrenti assenti e 14.66 GiB stimati liberi. Log build/c7/short-b-final-1.

- **25/11/2026, Codex**, C7 short-a-final-1. pp512/tg0 d0 703.83 t/s, pp0/tg64 d0 24.78 t/s Exit 0. Processi concorrenti assenti e 14.67 GiB stimati liberi. Log build/c7/short-a-final-1.

- **25/11/2026, Codex**, C7 kld-b-final. Maximum KLD:   0.000037 | Same top p: 100.000 ± 0.000 % Exit 0. Processi concorrenti assenti e 14.67 GiB stimati liberi. Log build/c7/kld-b-final.

- **25/11/2026, Codex**, C7 ctest-b-final. 100% tests passed, 0 tests failed out of 4 Exit 0. Processi concorrenti assenti e 14.666 GiB stimati liberi. Log build/c7/ctest-b-final.

- **25/11/2026, Codex**, C7 ops-b-final.   416/416 tests passed Exit 0. Processi concorrenti assenti e 14.659 GiB stimati liberi. Log build/c7/ops-b-final.

- **25/11/2026, Codex**, C7 Hadamard subgroup A-B-B-A. pp512 703,92 → 725,62 t/s (+3,08%), 416/416. Profilo ingresso + uscita Hadamard 101,77 → 70,25 ms nel grafo caldo. Archiviate le varianti SwiGLU. Consolidare solo Hadamard e ripetere misure e qualità.

- **25/11/2026, Codex**, C7 short-a-shuffle-2. pp512/tg0 d0 719.9 t/s, pp0/tg64 d0 24.6 t/s Exit 0. Processi concorrenti assenti e 14.663 GiB stimati liberi. Log build/c7/short-a-shuffle-2.

- **25/11/2026, Codex**, C7 short-b-shuffle-2. pp512/tg0 d0 712.82 t/s, pp0/tg64 d0 25.49 t/s Exit 0. Processi concorrenti assenti e 14.668 GiB stimati liberi. Log build/c7/short-b-shuffle-2.

- **25/11/2026, Codex**, C7 short-b-shuffle-1. pp512/tg0 d0 738.41 t/s, pp0/tg64 d0 24.51 t/s Exit 0. Processi concorrenti assenti e 14.662 GiB stimati liberi. Log build/c7/short-b-shuffle-1.

- **25/11/2026, Codex**, C7 short-a-shuffle-1. pp512/tg0 d0 687.94 t/s, pp0/tg64 d0 24.66 t/s Exit 0. Processi concorrenti assenti e 14.662 GiB stimati liberi. Log build/c7/short-a-shuffle-1.

- **25/11/2026, Codex**, C7 profile-short-shuffle. Profilo C6 wide con pp512, exit 0. Nessun processo concorrente e 14.665 GiB di VRAM stimati liberi. Log build/c7/profile-short-shuffle.

- **25/11/2026, Codex**, C7 ops-b-shuffle.   416/416 tests passed Exit 0. Processi concorrenti assenti e 14.661 GiB stimati liberi. Log build/c7/ops-b-shuffle.

- **25/11/2026, Codex**, C7 build Hadamard subgroup. Build Windows e spirv-val riusciti. Le prime cinque tappe Hadamard da 32 valori passano per subgroupShuffleXor, le altre riprendono nella memoria condivisa. Fusione SwiGLU disattivata in questa prova.

- **25/11/2026, Codex**, C7 pp8192 della fusione. Fusione con scheduler a 8192: 670,02 → 680,78 t/s (+1,61%) in A-B-B-A, GPU dedicata invariata. KLD contro CPU su 385 token uguale a C6: top-1 100%, KL max 0,000037. Il breve resta -1,35%. Provare Hadamard con scambi nei sottogruppi, a fusione SwiGLU disattivata.

- **25/11/2026, Codex**, C7 kld-b-scheduled. Maximum KLD:   0.000037 | Same top p: 100.000 ± 0.000 % Exit 0. Processi concorrenti assenti e 14.586 GiB stimati liberi. Log build/c7/kld-b-scheduled.

- **25/11/2026, Codex**, C7 pp8192-a-scheduled-2. pp8192/tg0 d0 667.64 t/s Exit 0. Processi concorrenti assenti e 14.582 GiB stimati liberi. Log build/c7/pp8192-a-scheduled-2.

- **25/11/2026, Codex**, C7 pp8192-b-scheduled-2. pp8192/tg0 d0 679.96 t/s Exit 0. Processi concorrenti assenti e 14.588 GiB stimati liberi. Log build/c7/pp8192-b-scheduled-2.

- **25/11/2026, Codex**, C7 pp8192-b-scheduled-1. pp8192/tg0 d0 681.59 t/s Exit 0. Processi concorrenti assenti e 14.579 GiB stimati liberi. Log build/c7/pp8192-b-scheduled-1.

- **25/11/2026, Codex**, C7 pp8192-a-scheduled-1. pp8192/tg0 d0 672.39 t/s Exit 0. Processi concorrenti assenti e 14.581 GiB stimati liberi. Log build/c7/pp8192-a-scheduled-1.

- **25/11/2026, Codex**, C7 fusione completa su pp512. Lo scheduler tiene adiacenti i nodi e mantiene vivi gli ingressi con dipendenze di allocazione. Il profilo conferma 64/64 fusioni, buffer di calcolo 1444,52 MiB per entrambe le build. Ultimo grafo 750,19 → 740,50 ms, ma il benchmark breve A-B-B-A dà 704,31 → 694,82 t/s (-1,35%). Vulkan 416/416. Provare pp8192 per misurare su carico più lungo.

- **25/11/2026, Codex**, C7 short-a-scheduled-2. pp512/tg0 d0 717.42 t/s, pp0/tg64 d0 24.63 t/s Exit 0. Processi concorrenti assenti e 14.586 GiB stimati liberi. Log build/c7/short-a-scheduled-2.

- **25/11/2026, Codex**, C7 short-b-scheduled-2. pp512/tg0 d0 693.79 t/s, pp0/tg64 d0 24.56 t/s Exit 0. Processi concorrenti assenti e 14.585 GiB stimati liberi. Log build/c7/short-b-scheduled-2.

- **25/11/2026, Codex**, C7 short-b-scheduled-1. pp512/tg0 d0 695.84 t/s, pp0/tg64 d0 24.57 t/s Exit 0. Processi concorrenti assenti e 14.58 GiB stimati liberi. Log build/c7/short-b-scheduled-1.

- **25/11/2026, Codex**, C7 short-a-scheduled-1. pp512/tg0 d0 691.2 t/s, pp0/tg64 d0 24.71 t/s Exit 0. Processi concorrenti assenti e 14.584 GiB stimati liberi. Log build/c7/short-a-scheduled-1.

- **25/11/2026, Codex**, C7 profile-short-fused-scheduled. Profilo C6 wide con pp512, exit 0. Nessun processo concorrente e 14.58 GiB di VRAM stimati liberi. Log build/c7/profile-short-fused-scheduled.

- **25/11/2026, Codex**, C7 ops-b-scheduled.   416/416 tests passed Exit 0. Processi concorrenti assenti e 14.589 GiB stimati liberi. Log build/c7/ops-b-scheduled.

- **25/11/2026, Codex**, C7 seconda fusione A-B-B-A. 17/64 coppie anche con alias sicuro. A-B-B-A 706,45 → 708,87 t/s (+0,34%), 416/416. Rimosso il permesso di alias sperimentale. Ora si usa il meccanismo Vulkan esistente per tenere adiacenti i due nodi e mantenere vivi gli ingressi fino all'uscita.

- **25/11/2026, Codex**, C7 short-a-prefix-2. pp512/tg0 d0 689.48 t/s, pp0/tg64 d0 25.37 t/s Exit 0. Processi concorrenti assenti e 14.588 GiB stimati liberi. Log build/c7/short-a-prefix-2.

- **25/11/2026, Codex**, C7 short-b-prefix-2. pp512/tg0 d0 704.45 t/s, pp0/tg64 d0 24.49 t/s Exit 0. Processi concorrenti assenti e 14.588 GiB stimati liberi. Log build/c7/short-b-prefix-2.

- **25/11/2026, Codex**, C7 short-b-prefix-1. pp512/tg0 d0 713.29 t/s, pp0/tg64 d0 24.58 t/s Exit 0. Processi concorrenti assenti e 14.577 GiB stimati liberi. Log build/c7/short-b-prefix-1.

- **25/11/2026, Codex**, C7 short-a-prefix-1. pp512/tg0 d0 723.41 t/s, pp0/tg64 d0 25.44 t/s Exit 0. Processi concorrenti assenti e 14.579 GiB stimati liberi. Log build/c7/short-a-prefix-1.

- **25/11/2026, Codex**, C7 profile-short-fused-prefix. Profilo C6 wide con pp512, exit 0. Nessun processo concorrente e 14.582 GiB di VRAM stimati liberi. Log build/c7/profile-short-fused-prefix.

- **25/11/2026, Codex**, C7 ops-b-prefix.   416/416 tests passed Exit 0. Processi concorrenti assenti e 14.582 GiB stimati liberi. Log build/c7/ops-b-prefix.

- **25/11/2026, Codex**, C7 verifica con alias sicuro. Il confronto dei tensori nel GGUF trova pre e s16 diversi in 64/64 coppie gate/up. Nessuna coppia può riusare identica la preparazione di ingresso. Build Release riuscita per la variante che consente l'alias dimostrato sicuro.

- **25/11/2026, Codex**, C7 prima fusione A-B-B-A. 17 coppie fuse su 64 nel modello e 416/416. pp512 702,77 → 708,77 t/s (+0,85%), con ampia oscillazione. Il profilo scende da 750,19 a 744,96 ms nell'ultimo grafo. Prova successiva consente l'alias solo se il prefisso già scritto del down non può sovrascrivere i valori di gate/up ancora da leggere.

- **25/11/2026, Codex**, C7 short-a-2. pp512/tg0 d0 704.96 t/s, pp0/tg64 d0 24.67 t/s Exit 0. Processi concorrenti assenti e 14.584 GiB stimati liberi. Log build/c7/short-a-2.

- **25/11/2026, Codex**, C7 short-b-2. pp512/tg0 d0 690.88 t/s, pp0/tg64 d0 24.61 t/s Exit 0. Processi concorrenti assenti e 14.585 GiB stimati liberi. Log build/c7/short-b-2.

- **25/11/2026, Codex**, C7 short-b-1. pp512/tg0 d0 726.66 t/s, pp0/tg64 d0 25.5 t/s Exit 0. Processi concorrenti assenti e 14.589 GiB stimati liberi. Log build/c7/short-b-1.

- **25/11/2026, Codex**, C7 short-a-1. pp512/tg0 d0 700.57 t/s, pp0/tg64 d0 25.46 t/s Exit 0. Processi concorrenti assenti e 14.581 GiB stimati liberi. Log build/c7/short-a-1.

- **25/11/2026, Codex**, C7 profile-short-fused. Profilo C6 wide con pp512, exit 0. Nessun processo concorrente e 14.586 GiB di VRAM stimati liberi. Log build/c7/profile-short-fused.

- **25/11/2026, Codex**, C7 ops-b.   416/416 tests passed Exit 0. Processi concorrenti assenti e 14.577 GiB stimati liberi. Log build/c7/ops-b.

- **25/11/2026, Codex**, C7 build fusione. SPIR-V e build Windows Release riusciti. La fusione riutilizza il blocco Hadamard e lascia i prodotti e le rotazioni di uscita nei kernel esistenti. In verifica profilo e risultati Vulkan.

- **25/11/2026, Codex**, C7 profilo a 32K. Dopo il prefisso da 32768, ultimo grafo pp512 profilato: 1126,78 ms, SwiGLU 11,75 ms, Hadamard intra-Trellis 77,01 ms ingresso e 40,22 ms uscita. Per la larghezza MLP di 17408, Hadamard a 17 blocchi da 1024. Prototipo limitato a SwiGLU split con prodotti Trellis e Hadamard a blocchi del down.

- **25/11/2026, Codex**, C7 profile-32k. Profilo C6 wide con pp512, exit 0. Nessun processo concorrente e 13.759 GiB di VRAM stimati liberi. Log build/c7/profile-32k.

- **25/11/2026, Codex**, C7 profilo breve. Su 512 token, seconda esecuzione profilata: 750,19 ms GPU, 128 prodotti gate/up 266,33 ms inclusi input/uscita, 64 prodotti down 167,75 ms, 64 SwiGLU 11,70 ms. Hadamard intra-Trellis 68,86 ms ingresso e 32,91 ms uscita per tutti i prodotti. Provare fusione mirata di SwiGLU, preparazione e Hadamard di ingresso del down.

- **25/11/2026, Codex**, C7 profile-short. Profilo C6 wide con pp512, exit 0. Nessun processo concorrente e 14.338 GiB di VRAM stimati liberi. Log build/c7/profile-short.

- **25/11/2026, Codex**, C7 avvio. Riferimento C6 60b4d242d. Profilare il prefill con wide=1 e misurare le componenti MLP prima di scegliere la fusione.

- **25/11/2026, Codex**, C6 finale. pp512 615,46 → 696,46 (+13,2%), pp8192 588,70 → 667,32 (+13,4%), pp512 a profondità 32768 444,14 → 529,35 (+19,2%) t/s. Vulkan 416/416, CTest 4/4 e WikiText-2 40 finestre PPL 6,2456 → 6,2456. In una finestra da 385 token contro CPU, con coda dopo il blocco da 256, stesso top-1 (100,000%) e stessa KL massima a sei decimali (0,000037). Accettato prefill FP16 a 256 come opt-in. Log `build/c6/*-final-*` e `kld-*`.

- **25/11/2026, Codex**, C6 kld-b. Mean    KLD:  -0.000000 ±   0.000001 | Maximum KLD:   0.000037 | Same top p: 100.000 ± 0.000 %. Exit 0. 14.626 GiB stimati liberi prima della prova. Log build/c6/kld-b.

- **25/11/2026, Codex**, C6 kld-a. Mean    KLD:  -0.000000 ±   0.000001 | Maximum KLD:   0.000037 | Same top p: 100.000 ± 0.000 %. Exit 0. 14.626 GiB stimati liberi prima della prova. Log build/c6/kld-a.

- **25/11/2026, Codex**, C6 kld-cpu. 0.07.884.012 I perplexity: 3.44 seconds per pass - ETA 0.08.132.419 I Final estimate: PPL = 5.9031 +/- 1.12566. Exit 0. 14.63 GiB stimati liberi prima della prova. Log build/c6/kld-cpu.

- **25/11/2026, Codex**, C6 ppl-0-final-b. 0.08.490.256 I perplexity: 1.37 seconds per pass - ETA 0.39.422.870 I Final estimate: PPL = 6.2456 +/- 0.14665. Exit 0. Processi concorrenti assenti, 14.616 GiB stimati liberi. Log build/c6/ppl-0-final-b.

- **25/11/2026, Codex**, C6 ppl-0-final-a. 0.07.585.844 I perplexity: 1.20 seconds per pass - ETA 0.43.026.915 I Final estimate: PPL = 6.2456 +/- 0.14665. Exit 0. Processi concorrenti assenti, 14.608 GiB stimati liberi. Log build/c6/ppl-0-final-a.

- **25/11/2026, Codex**, C6 verifica log WikiText-2. Prima PPL FP16 terminata con PPL 6,2456 e circa 11,14 GiB nel processo Vulkan. Il wrapper richiedeva la riga di offload, omessa al livello di log predefinito di llama-perplexity. Ripetizione A/B con --verbose per confermare 65/65.

- **25/11/2026, Codex**, C6 ppl-0-final-a. 0.09.529.850 I perplexity: 1.60 seconds per pass - ETA 0.45.027.696 I Final estimate: PPL = 6.2456 +/- 0.14665. Exit 1. Processi concorrenti assenti, 14.618 GiB stimati liberi. Log build/c6/ppl-0-final-a.

- **25/11/2026, Codex**, C6 deep-0-final-b. pp512/tg0 d32768 529.35 t/s, pp0/tg64 d32768 22.24 t/s. Exit 0. Processi concorrenti assenti, 14.618 GiB stimati liberi. Log build/c6/deep-0-final-b.

- **25/11/2026, Codex**, C6 deep-0-final-a. pp512/tg0 d32768 444.14 t/s, pp0/tg64 d32768 21.51 t/s. Exit 0. Processi concorrenti assenti, 14.621 GiB stimati liberi. Log build/c6/deep-0-final-a.

- **25/11/2026, Codex**, C6 pp8192-0-final-b. pp8192/tg0 d0 667.32 t/s. Exit 0. Processi concorrenti assenti, 14.619 GiB stimati liberi. Log build/c6/pp8192-0-final-b.

- **25/11/2026, Codex**, C6 pp8192-0-final-a. pp8192/tg0 d0 588.7 t/s. Exit 0. Processi concorrenti assenti, 14.616 GiB stimati liberi. Log build/c6/pp8192-0-final-a.

- **25/11/2026, Codex**, C6 short-0-final-b. pp512/tg0 d0 696.46 t/s, pp0/tg64 d0 24.7 t/s. Exit 0. Processi concorrenti assenti, 14.621 GiB stimati liberi. Log build/c6/short-0-final-b.

- **25/11/2026, Codex**, C6 short-0-final-a. pp512/tg0 d0 615.46 t/s, pp0/tg64 d0 25.55 t/s. Exit 0. Processi concorrenti assenti, 14.617 GiB stimati liberi. Log build/c6/short-0-final-a.

- **25/11/2026, Codex**, C6 ctest-0-final. 100% tests passed, 0 tests failed out of 4. Exit 0. Processi concorrenti assenti, 14.617 GiB stimati liberi. Log build/c6/ctest-0-final.

- **25/11/2026, Codex**, C6 ops-0-final. 416/416 tests passed. Exit 0. Processi concorrenti assenti, 14.616 GiB stimati liberi. Log build/c6/ops-0-final.

- **25/11/2026, Codex**, C6 integrazione batch 256. Blocco FP16 256 come opt-in GGML_VK_TRELLIS_MM_WIDE=1, con scratch verificato e fallback a 128 per i resti, per la variante con residuo e dove la scheda non ha i requisiti. SPIR-V di prodotto e preparazione validati. Build Windows in corso.

- **25/11/2026, Codex**, C6 A-B-B-A batch 256. FP16 606,11 → 692,31 t/s (+14,2%) su pp512, 416/416, VRAM di picco +21 MiB sul bench. Prototipo archiviato. Integrare come opt-in solo per blocchi completi, mantenere kernel e batch da 128 per i resti.

- **25/11/2026, Codex**, C6 short-0-wide-a2. pp512/tg0 d0 604.97 t/s, pp0/tg64 d0 24.73 t/s. Exit 0. Processi concorrenti assenti, 14.621 GiB stimati liberi. Log build/c6/short-0-wide-a2.

- **25/11/2026, Codex**, C6 short-0-wide-b2. pp512/tg0 d0 692.95 t/s, pp0/tg64 d0 24.59 t/s. Exit 0. Processi concorrenti assenti, 14.609 GiB stimati liberi. Log build/c6/short-0-wide-b2.

- **25/11/2026, Codex**, C6 short-0-wide-b1. pp512/tg0 d0 691.66 t/s, pp0/tg64 d0 24.61 t/s. Exit 0. Processi concorrenti assenti, 14.614 GiB stimati liberi. Log build/c6/short-0-wide-b1.

- **25/11/2026, Codex**, C6 short-0-wide-a1. pp512/tg0 d0 607.24 t/s, pp0/tg64 d0 25.51 t/s. Exit 0. Processi concorrenti assenti, 14.619 GiB stimati liberi. Log build/c6/short-0-wide-a1.

- **25/11/2026, Codex**, C6 ops-0-wide. 416/416 tests passed. Exit 0. Processi concorrenti assenti, 14.611 GiB stimati liberi. Log build/c6/ops-0-wide.

- **25/11/2026, Codex**, C6 batch 256. Compilati e validati SPIR-V del prototipo FP16 a 256 token. Risultati completi scritti direttamente e coda parziale gestita in LDS. Raddoppia la durata del riuso dei pesi rispetto alla build a 128. In corso build e misura.

- **25/11/2026, Codex**, C6 A-B-B-A layout contiguo. FP16 per tile 620,17 → 607,74 t/s (-2,0%) su pp512 e 416/416. Scartato. Archiviati codice e DLL INT8, store diretto e layout. Prossima prova FP16 su blocchi di 256 token con riuso della decodifica dei pesi.

- **25/11/2026, Codex**, C6 short-0-tiled-a2. pp512/tg0 d0 628.23 t/s, pp0/tg64 d0 24.63 t/s. Exit 0. Processi concorrenti assenti, 14.62 GiB stimati liberi. Log build/c6/short-0-tiled-a2.

- **25/11/2026, Codex**, C6 short-0-tiled-b2. pp512/tg0 d0 607.69 t/s, pp0/tg64 d0 24.72 t/s. Exit 0. Processi concorrenti assenti, 14.619 GiB stimati liberi. Log build/c6/short-0-tiled-b2.

- **25/11/2026, Codex**, C6 short-0-tiled-b1. pp512/tg0 d0 607.78 t/s, pp0/tg64 d0 24.66 t/s. Exit 0. Processi concorrenti assenti, 14.618 GiB stimati liberi. Log build/c6/short-0-tiled-b1.

- **25/11/2026, Codex**, C6 short-0-tiled-a1. pp512/tg0 d0 612.1 t/s, pp0/tg64 d0 25.54 t/s. Exit 0. Processi concorrenti assenti, 14.613 GiB stimati liberi. Log build/c6/short-0-tiled-a1.

- **25/11/2026, Codex**, C6 ops-0-tiled. 416/416 tests passed. Exit 0. Processi concorrenti assenti, 14.612 GiB stimati liberi. Log build/c6/ops-0-tiled.

- **25/11/2026, Codex**, C6 layout a tile. Build Release riuscita per FP16 a tile contigue, con un solo termine nello scratch. In verifica la parità e il confronto A-B-B-A.

- **25/11/2026, Codex**, C6 A-B-B-A store diretto. FP16 608,32 → 611,91 t/s (+0,59%) su pp512. Il vantaggio iniziale non si ripete fuori dal rumore. INT8 residuo 548,41 t/s con store diretto e 416/416. In compilazione un layout contiguo per le attivazioni FP16 della matrice cooperativa.

- **25/11/2026, Codex**, C6 short-0-direct-a2. pp512/tg0 d0 612.6 t/s, pp0/tg64 d0 24.79 t/s. Exit 0. Processi concorrenti assenti, 14.623 GiB stimati liberi. Log build/c6/short-0-direct-a2.

- **25/11/2026, Codex**, C6 short-0-direct-b2. pp512/tg0 d0 617.54 t/s, pp0/tg64 d0 24.73 t/s. Exit 0. Processi concorrenti assenti, 14.618 GiB stimati liberi. Log build/c6/short-0-direct-b2.

- **25/11/2026, Codex**, C6 short-0-direct-b1. pp512/tg0 d0 606.27 t/s, pp0/tg64 d0 24.69 t/s. Exit 0. Processi concorrenti assenti, 14.62 GiB stimati liberi. Log build/c6/short-0-direct-b1.

- **25/11/2026, Codex**, C6 short-0-direct-a1. pp512/tg0 d0 604.03 t/s, pp0/tg64 d0 24.76 t/s. Exit 0. Processi concorrenti assenti, 14.608 GiB stimati liberi. Log build/c6/short-0-direct-a1.

- **25/11/2026, Codex**, C6 short-2-direct. pp512/tg0 d0 548.41 t/s, pp0/tg64 d0 24.83 t/s. Exit 0. Processi concorrenti assenti, 14.616 GiB stimati liberi. Log build/c6/short-2-direct.

- **25/11/2026, Codex**, C6 ops-2-direct. 416/416 tests passed. Exit 0. Processi concorrenti assenti, 14.62 GiB stimati liberi. Log build/c6/ops-2-direct.

- **25/11/2026, Codex**, C6 short-0-direct. pp512/tg0 d0 627.19 t/s, pp0/tg64 d0 25.56 t/s. Exit 0. Processi concorrenti assenti, 14.615 GiB stimati liberi. Log build/c6/short-0-direct.

- **25/11/2026, Codex**, C6 ops-0-direct. 416/416 tests passed. Exit 0. Processi concorrenti assenti, 14.615 GiB stimati liberi. Log build/c6/ops-0-direct.

- **25/11/2026, Codex**, C6 diagnostica numerica. ISA AMD conferma v_wmma_i32_16x16x16_iu8. Nei 26 logits di prompt, KL massima CPU FP16 3,06e-7, INT8 a un termine 8,22e-4, INT8 con residuo 3,02e-8. Tutti 31/31 top-1 nel test completo. La soglia 1e-6 sul test completo non passa nemmeno con FP16 perché include la generazione INT8 a un token.

- **25/11/2026, Codex**, C6 store diretto. Compilati con glslc e spirv-val i due prodotti con store diretto dei risultati. La scala per token si applica nel sottogruppo. Risparmia 8 KiB di LDS per gruppo e usa il percorso originale per le code non multiple di 16.

- **25/11/2026, Codex**, C6 isa-1. TRL3 395.37 us, TRL4 455.82 us. Exit 0. Processi concorrenti assenti, 13.838 GiB stimati liberi. Log build/c6/isa-1.

- **25/11/2026, Codex**, C6 logits-2. top-1 31/31, greedy True, KL max 0.000161149522276782, media 7.60262718245935E-06. Exit 1. Processi concorrenti assenti, 13.866 GiB stimati liberi. Log build/c6/logits-2.

- **25/11/2026, Codex**, C6 logits-1. top-1 31/31, greedy True, KL max 0.000821862253521766, media 0.000205871004766816. Exit 1. Processi concorrenti assenti, 13.866 GiB stimati liberi. Log build/c6/logits-1.

- **25/11/2026, Codex**, C6 logits-0. top-1 31/31, greedy True, KL max 6.16944258966882E-05, media 4.03096388117599E-06. Exit 1. Processi concorrenti assenti, 13.866 GiB stimati liberi. Log build/c6/logits-0.

- **25/11/2026, Codex**, C6 prima decisione. pp512 FP16 601,01, INT8 a un termine 758,86 (+26,3%) ma 242/416, INT8 con residuo 537,16 (-10,6%) e 416/416. Nessuna variante qualificata. Verificare logits e provare store cooperativo diretto per togliere LDS al prefill.

- **25/11/2026, Codex**, C6 short-2. pp512/tg0 d0 537.16 t/s, pp0/tg64 d0 24.43 t/s. Exit 0. Processi concorrenti assenti, 13.865 GiB stimati liberi. Log build/c6/short-2.

- **25/11/2026, Codex**, C6 short-1. pp512/tg0 d0 758.86 t/s, pp0/tg64 d0 24.43 t/s. Exit 0. Processi concorrenti assenti, 13.866 GiB stimati liberi. Log build/c6/short-1.

- **25/11/2026, Codex**, C6 short-0. pp512/tg0 d0 601.01 t/s, pp0/tg64 d0 24.67 t/s. Exit 0. Processi concorrenti assenti, 13.825 GiB stimati liberi. Log build/c6/short-0.

- **25/11/2026, Codex**, C6 ops-2. 416/416 tests passed. Exit 0. Processi concorrenti assenti, 13.82 GiB stimati liberi. Log build/c6/ops-2.

- **25/11/2026, Codex**, C6 ops-1. 242/416 tests passed. Exit 1. Processi concorrenti assenti, 13.767 GiB stimati liberi. Log build/c6/ops-1.

- **25/11/2026, Codex**, C6 build Release. BUILD_OK con MSVC e Vulkan SDK. Inizio test e confronto con FP16 del commit 10339c849.

- **25/11/2026, Codex**, C6 compilazione shader. Vulkan S8/S8/S32 16x16x16 verificato in B6. Prototipo che riusa la forma C3, con scala per token e uno o due termini INT8. Entrambe le varianti passano glslc e spirv-val. Build e risultati da verificare.

- **25/11/2026, Codex**, B6 idea 2 finale, build Windows Release con barriera iniziale. In A-B-B-A a quattro token TRL3 195,03 → 147,65 µs (-24,3%) e TRL4 190,34 → 146,97 µs (-22,8%), rapporti 4/1 1,35 e 1,38. Vulkan 416/416, CTest 4/4 e identità bit per bit dei logits contro A5b. Contro CPU top-1 31/31, stessi greedy e KL max 2,10e-4. DFlash2 B1 51,31 → 65,13, D2 29,02/31,89/31,31 → 37,23/40,32/39,90 t/s (breve/32K/65K) con gli stessi hash e la stessa accettazione. Picco D2 13,65 GiB dedicati e 0,10 condivisi. Accettata come opzione sperimentale. Log `build/b6-cm/final-*` e baseline `v4-server-0.*`.

- **25/11/2026, Codex**, B6 idea 2, final-server-1 (exit 0). short 37.23 t/s, accettati 281/689, ciclo 59.76 ms | ctx32k 40.32 t/s, accettati 156/297, ciclo 63.88 ms | ctx65k 39.90 t/s, accettati 157/297, ciclo 64.55 ms. Picco 13.65 GiB dedicati e 0.10 condivisi. Processi concorrenti assenti prima della prova, 14.945 GiB stimati liberi. Log in build/b6-cm/final-server-1.*.

- **25/11/2026, Codex**, B6 idea 2, final-b1-1 (exit 0). b1 65.13 t/s, accettati 192/192, ciclo 61.17 ms. Picco 11.85 GiB dedicati e 0.10 condivisi. Processi concorrenti assenti prima della prova, 14.945 GiB stimati liberi. Log in build/b6-cm/final-b1-1.*.

- **25/11/2026, Codex**, B6 idea 2, final-logits-u4 (exit 0). CPU top-1 31/31, greedy uguale: True, KL max 2.101E-004. Contro A5b packed KL max 0.000E+000, logits bit identici: True. Processi concorrenti assenti prima della prova, 14.956 GiB stimati liberi. Log in build/b6-cm/final-logits-u4.*.

- **25/11/2026, Codex**, B6 idea 2, final-ctest (exit 0). CTest Trellis 4/4. Processi concorrenti assenti prima della prova, 14.945 GiB stimati liberi. Log in build/b6-cm/final-ctest.*.

- **25/11/2026, Codex**, B6 idea 2, final-perf-a2 (exit 0), 5120 → 17408: TRL3 t1 107.72 µs, TRL3 t4 190.90 µs, TRL4 t1 106.78 µs, TRL4 t4 186.56 µs. Processi concorrenti assenti, 14.945 GiB stimati liberi. Log build/b6-cm/final-perf-a2.log.

- **25/11/2026, Codex**, B6 idea 2, final-perf-b2 (exit 0), 5120 → 17408: TRL3 t1 107.76 µs, TRL3 t4 147.65 µs, TRL4 t1 106.59 µs, TRL4 t4 147.27 µs. Processi concorrenti assenti, 14.946 GiB stimati liberi. Log build/b6-cm/final-perf-b2.log.

- **25/11/2026, Codex**, B6 idea 2, final-perf-b1 (exit 0), 5120 → 17408: TRL3 t1 111.14 µs, TRL3 t4 147.65 µs, TRL4 t1 106.51 µs, TRL4 t4 146.67 µs. Processi concorrenti assenti, 14.956 GiB stimati liberi. Log build/b6-cm/final-perf-b1.log.

- **25/11/2026, Codex**, B6 idea 2, final-perf-a1 (exit 0), 5120 → 17408: TRL3 t1 112.35 µs, TRL3 t4 199.17 µs, TRL4 t1 110.91 µs, TRL4 t4 194.11 µs. Processi concorrenti assenti, 14.956 GiB stimati liberi. Log build/b6-cm/final-perf-a1.log.

- **25/11/2026, Codex**, B6 idea 2, sorgente finale con barriera iniziale: BUILD_OK e test-backend-ops   416/416 tests passed (exit 0), processi concorrenti assenti e 14.945 GiB stimati liberi. Log build/b6-cm/build-final.log e final-check.log.

- **25/11/2026, Codex**, B6 idea 2, conclusione A/B v4: D2 +29,5% / +30,8% / +26,9%, cicli 76,66/80,76/82,27 → 59,21/61,74/64,82 ms e stessi token accettati 281/689, 156/297, 157/297. Picco dedicato 13,647 GiB e condiviso 0,099 in entrambi. B1 51,31 → 64,85, stesso hash e 192/192. B2 ~25 tg64, B3 23,70 → 23,72 tg64 e 466,28 → 471,02 pp512. In revisione aggiunta la barriera iniziale per rendere ordinati gli azzeramenti e le riscritture sparse eseguite da lane diverse. Nessuna modifica all’accumulo. Esito: ricompilare e qualificare questa versione.

- **25/11/2026, Codex**, B6 idea 2, v4-bench3-1 (exit 0). pp512 471.02 t/s e tg64 23.72 t/s, profondità 32768, 1 ripetizione. Processi concorrenti assenti prima della prova, 14.808 GiB stimati liberi. Log in build/b6-cm/v4-bench3-1.*.

- **25/11/2026, Codex**, B6 idea 2, v4-bench3-0 (exit 0). pp512 466.28 t/s e tg64 23.70 t/s, profondità 32768, 1 ripetizione. Processi concorrenti assenti prima della prova, 14.863 GiB stimati liberi. Log in build/b6-cm/v4-bench3-0.*.

- **25/11/2026, Codex**, B6 idea 2, v4-server-1 (exit 0). short 37.58 t/s, accettati 281/689, ciclo 59.21 ms | ctx32k 41.72 t/s, accettati 156/297, ciclo 61.74 ms | ctx65k 39.74 t/s, accettati 157/297, ciclo 64.82 ms. Picco 13.65 GiB dedicati e 0.10 condivisi. Processi concorrenti assenti prima della prova, 14.851 GiB stimati liberi. Log in build/b6-cm/v4-server-1.*.

- **25/11/2026, Codex**, B6 idea 2, v4-server-0 (exit 0). short 29.02 t/s, accettati 281/689, ciclo 76.66 ms | ctx32k 31.89 t/s, accettati 156/297, ciclo 80.76 ms | ctx65k 31.31 t/s, accettati 157/297, ciclo 82.27 ms. Picco 13.65 GiB dedicati e 0.10 condivisi. Processi concorrenti assenti prima della prova, 14.882 GiB stimati liberi. Log in build/b6-cm/v4-server-0.*.

- **25/11/2026, Codex**, B6 idea 2, v4-b1-1 (exit 0). b1 64.85 t/s, accettati 192/192, ciclo 61.44 ms. Picco 11.85 GiB dedicati e 0.10 condivisi. Processi concorrenti assenti prima della prova, 14.864 GiB stimati liberi. Log in build/b6-cm/v4-b1-1.*.

- **25/11/2026, Codex**, B6 idea 2, v4-b1-0 (exit 0). b1 51.31 t/s, accettati 192/192, ciclo 77.65 ms. Picco 11.85 GiB dedicati e 0.10 condivisi. Processi concorrenti assenti prima della prova, 14.359 GiB stimati liberi. Log in build/b6-cm/v4-b1-0.*.

- **25/11/2026, Codex**, B6 idea 2, v4-bench-1 (exit 0). pp512 628.92 t/s e tg64 25.20 t/s, 3 ripetizioni. Processi concorrenti assenti prima della prova, 14.562 GiB stimati liberi. Log in build/b6-cm/v4-bench-1.*.

- **25/11/2026, Codex**, B6 idea 2, v4-bench-0 (exit 0). pp512 623.63 t/s e tg64 25.54 t/s, 3 ripetizioni. Processi concorrenti assenti prima della prova, 14.53 GiB stimati liberi. Log in build/b6-cm/v4-bench-0.*.

- **25/11/2026, Codex**, B6 idea 2, v4-logits-u4 (exit 0). CPU top-1 31/31, greedy uguale: True, KL max 2.101E-004. Contro A5b packed KL max 0.000E+000, logits bit identici: True. Processi concorrenti assenti prima della prova, 14.543 GiB stimati liberi. Log in build/b6-cm/v4-logits-u4.*.

- **25/11/2026, Codex**, B6 idea 2, v4-ctest (exit 0). CTest Trellis 4/4. Processi concorrenti assenti prima della prova, 14.541 GiB stimati liberi. Log in build/b6-cm/v4-ctest.*.

- **25/11/2026, Codex**, B6 idea 2, v4 A-B-B-A con accumulatori cooperativi FP32 nei registri: a 4 token TRL3 190,45 → 147,60 µs, TRL4 186,95 → 146,95 µs. A 1 token TRL3 107,76 → 107,70 µs e TRL4 106,39 → 106,67. Nessun processo concorrente e 14,54–14,55 GiB stimati liberi. La variante evita store e reload dei 256 parziali ad ogni tile. Esito: verificare logits esatti e DFlash2. Log `build/b6-cm/v4-perf-*`.

- **25/11/2026, Codex**, B6 idea 2, v4-perf-a2 (exit 0), 5120 → 17408: TRL3 t1 107.95 µs, TRL3 t4 190.68 µs, TRL4 t1 106.66 µs, TRL4 t4 185.93 µs. Processi concorrenti assenti, 14.548 GiB stimati liberi. Log build/b6-cm/v4-perf-a2.log.

- **25/11/2026, Codex**, B6 idea 2, v4-perf-b2 (exit 0), 5120 → 17408: TRL3 t1 107.53 µs, TRL3 t4 147.68 µs, TRL4 t1 106.71 µs, TRL4 t4 146.42 µs. Processi concorrenti assenti, 14.542 GiB stimati liberi. Log build/b6-cm/v4-perf-b2.log.

- **25/11/2026, Codex**, B6 idea 2, v4-perf-b1 (exit 0), 5120 → 17408: TRL3 t1 107.87 µs, TRL3 t4 147.51 µs, TRL4 t1 106.63 µs, TRL4 t4 147.47 µs. Processi concorrenti assenti, 14.543 GiB stimati liberi. Log build/b6-cm/v4-perf-b1.log.

- **25/11/2026, Codex**, B6 idea 2, v4-perf-a1 (exit 0), 5120 → 17408: TRL3 t1 107.56 µs, TRL3 t4 190.21 µs, TRL4 t1 106.11 µs, TRL4 t4 187.96 µs. Processi concorrenti assenti, 14.548 GiB stimati liberi. Log build/b6-cm/v4-perf-a1.log.

- **25/11/2026, Codex**, B6 idea 2, v4 BUILD_OK e test-backend-ops   416/416 tests passed (exit 0), processi concorrenti assenti e 14.548 GiB stimati liberi. Log build/b6-cm/build-v4.log e v4-check.log.

- **25/11/2026, Codex**, B6 idea 2, A-B-B-A v3: 4 token TRL3 190,65 → 207,04 µs (+8,6%), TRL4 186,44 → 222,90 (+19,6%); un token neutro. Processi assenti e 14,54–14,55 GiB stimati liberi. Le somme parziali del dot4 sono mantenute, ma il passaggio di ogni parziale per LDS costa troppo. Esito: non accettare v3. Provare a conservare gli accumulatori cooperativi nei registri; archiviare shader e DLL v3. Log `build/b6-cm/v3-perf-*`.

- **25/11/2026, Codex**, B6 idea 2, v3-perf-a2 (exit 0), 5120 → 17408: TRL3 t1 107.55 µs, TRL3 t4 190.48 µs, TRL4 t1 106.26 µs, TRL4 t4 185.67 µs. Processi concorrenti assenti, 14.543 GiB stimati liberi. Log build/b6-cm/v3-perf-a2.log.

- **25/11/2026, Codex**, B6 idea 2, v3-perf-b2 (exit 0), 5120 → 17408: TRL3 t1 107.61 µs, TRL3 t4 205.60 µs, TRL4 t1 106.16 µs, TRL4 t4 222.93 µs. Processi concorrenti assenti, 14.548 GiB stimati liberi. Log build/b6-cm/v3-perf-b2.log.

- **25/11/2026, Codex**, B6 idea 2, v3-perf-b1 (exit 0), 5120 → 17408: TRL3 t1 107.20 µs, TRL3 t4 208.48 µs, TRL4 t1 106.62 µs, TRL4 t4 222.87 µs. Processi concorrenti assenti, 14.548 GiB stimati liberi. Log build/b6-cm/v3-perf-b1.log.

- **25/11/2026, Codex**, B6 idea 2, v3-perf-a1 (exit 0), 5120 → 17408: TRL3 t1 107.49 µs, TRL3 t4 190.81 µs, TRL4 t1 106.27 µs, TRL4 t4 187.20 µs. Processi concorrenti assenti, 14.541 GiB stimati liberi. Log build/b6-cm/v3-perf-a1.log.

- **25/11/2026, Codex**, B6 idea 2, tentativo microbenchmark v3: nessuna prova GPU avviata per un errore di sintassi nel wrapper A/B. Corretta la prima riga, ripetere senza modifiche allo shader.

- **25/11/2026, Codex**, B6 idea 2, v3 BUILD_OK e test-backend-ops 416/416 tests passed (exit 0), processi concorrenti assenti e 14.543 GiB stimati liberi. Log build/b6-cm/build-v3.log e v3-check.log.

- **25/11/2026, Codex**, B6 idea 2, D2 v2 conclusa: 28,43/31,99/30,59 → 32,85/32,93/30,39 t/s (breve/32K/65K), ciclo stimato 78,26/80,52/84,20 → 66,96/71,93/74,05 ms. Accettati 281/689, 156/297, 157/297 → 279/697, 148/323, 142/340. Hash diversi per tutti e tre; picco dedicato 13,64 → 13,65 GiB. Esito: WMMA ha margine sul ciclo; v2 non promossa per la diversità numerica e il B1 peggiorato. Log `build/b6-cm/server-[01].*`.

- **25/11/2026, Codex**, B6 idea 2, server-1 (exit 0). short 32.85 t/s, accettati 279/697, ciclo 66.96 ms | ctx32k 32.93 t/s, accettati 148/323, ciclo 71.93 ms | ctx65k 30.39 t/s, accettati 142/340, ciclo 74.05 ms. Picco 13.65 GiB dedicati e 0.10 condivisi. Processi concorrenti assenti prima della prova, 14.334 GiB stimati liberi. Log in build/b6-cm/server-1.*.

- **25/11/2026, Codex**, B6 idea 2, v3 preparazione: B1 v2 cambia hash del testo e accettazione (192/192 → 167/264). La variante v3 usa le 16 colonne della matrice per 4 token × 4 gruppi di quattro pesi, così ogni somma parziale e il suo accumulo FP32 seguono il dot4. Compilazione standalone e spirv-val passati. V2 archiviato in `build/b6-cm/trellis_verify_cm-v2.comp`. Esito: completare D2 v2 prima di ricompilare.

- **25/11/2026, Codex**, B6 idea 2, server-0 (exit 0). short 28.43 t/s, accettati 281/689, ciclo 78.26 ms | ctx32k 31.99 t/s, accettati 156/297, ciclo 80.52 ms | ctx65k 30.59 t/s, accettati 157/297, ciclo 84.20 ms. Picco 13.64 GiB dedicati e 0.35 condivisi. Processi concorrenti assenti prima della prova, 14.342 GiB stimati liberi. Log in build/b6-cm/server-0.*.

- **25/11/2026, Codex**, B6 idea 2, b1-1 (exit 0). b1 43.12 t/s, accettati 167/264, ciclo 67.20 ms. Picco 11.85 GiB dedicati e 0.10 condivisi. Processi concorrenti assenti prima della prova, 14.343 GiB stimati liberi. Log in build/b6-cm/b1-1.*.

- **25/11/2026, Codex**, B6 idea 2, b1-0 (exit 0). b1 52.19 t/s, accettati 192/192, ciclo 76.34 ms. Picco 11.85 GiB dedicati e 0.10 condivisi. Processi concorrenti assenti prima della prova, 14.343 GiB stimati liberi. Log in build/b6-cm/b1-0.*.

- **25/11/2026, Codex**, B6 idea 2, bench-1 (exit 0). pp512 608.91 t/s e tg64 24.97 t/s, 3 ripetizioni. Processi concorrenti assenti prima della prova, 14.343 GiB stimati liberi. Log in build/b6-cm/bench-1.*.

- **25/11/2026, Codex**, B6 idea 2, bench-0 (exit 0). pp512 631.48 t/s e tg64 25.68 t/s, 3 ripetizioni. Processi concorrenti assenti prima della prova, 14.343 GiB stimati liberi. Log in build/b6-cm/bench-0.*.

- **25/11/2026, Codex**, B6 idea 2, chiarimento logits: il confronto con CPU passa il protocollo A5a (31/31 top-1, greedy identico, KL max 1,50e-4 contro 2,10e-4 di A5b). Il confronto aggiuntivo con A5b a 1e-6 fallisce (3,36e-4, top-1 31/31) e spiega il codice uscita 1 del wrapper. Il nuovo shader completa la somma intera del tile prima del FMA e raggruppa le somme float diversamente da dot4. Il traguardo qualità per OpenClaw resta aperto; misurare D2 con confronti di hash e accettazione.

- **25/11/2026, Codex**, B6 idea 2, logits-u4 (exit 1). CPU top-1 1 su 31 posizioni, greedy uguale: True, KL max 1.500E-004. Contro A5b packed KL max 3.364E-004, top-1 1. Processi concorrenti assenti prima della prova, 14.343 GiB stimati liberi. Log in build/b6-cm/logits-u4.*.

- **25/11/2026, Codex**, B6 idea 2, ctest (exit 0). CTest Trellis 4/4. Processi concorrenti assenti prima della prova, 14.343 GiB stimati liberi. Log in build/b6-cm/ctest.*.

- **25/11/2026, Codex**, B6 idea 2, sintesi A-B-B-A corretta: a quattro token TRL3 190,65 → 156,99 µs (-17,7%), TRL4 188,09 → 159,54 (-15,2%); a un token TRL3 108,11 → 108,01, TRL4 107,06 → 106,83. I rapporti passano da 1,76 a 1,45 e da 1,76 a 1,49. Processi assenti e 14,342 GiB stimati liberi in ogni run, baseline build/ref-bin con packed. Esito: promossa alla prova end-to-end, soglia 1,3 non raggiunta.

- **25/11/2026, Codex**, B6 idea 2, perf-a2 (exit 0), 5120 → 17408: TRL3 t1 108.13 µs, TRL3 t4 191.00 µs, TRL4 t1 107.47 µs, TRL4 t4 190.28 µs. Processi concorrenti assenti, 14.342 GiB stimati liberi. Log build/b6-cm/perf-a2.log.

- **25/11/2026, Codex**, B6 idea 2, perf-b2 (exit 0), 5120 → 17408: TRL3 t1 107.98 µs, TRL3 t4 156.86 µs, TRL4 t1 106.77 µs, TRL4 t4 159.51 µs. Processi concorrenti assenti, 14.342 GiB stimati liberi. Log build/b6-cm/perf-b2.log.

- **25/11/2026, Codex**, B6 idea 2, perf-b1 (exit 0), 5120 → 17408: TRL3 t1 108.04 µs, TRL3 t4 157.11 µs, TRL4 t1 106.89 µs, TRL4 t4 159.56 µs. Processi concorrenti assenti, 14.342 GiB stimati liberi. Log build/b6-cm/perf-b1.log.

- **25/11/2026, Codex**, B6 idea 2, perf-a1 (exit 0), 5120 → 17408: TRL3 t1 108.08 µs, TRL3 t4 190.30 µs, TRL4 t1 106.64 µs, TRL4 t4 185.90 µs. Processi concorrenti assenti, 14.342 GiB stimati liberi. Log build/b6-cm/perf-a1.log.

- **25/11/2026, Codex**, B6 idea 2, build v2 riuscita (`BUILD_OK`) e test-backend-ops Vulkan 416/416 con WMMA opt-in, inclusi i casi TRL1–6 e GET_ROWS. Processi concorrenti assenti, 14,342 GiB liberi stimati. Log `build/b6-cm/build-v2.log` e `check-v2.log`. Esito: misurare A-B-B-A.

- **25/11/2026, Codex**, B6 idea 2, test-backend-ops v1: 356/416. Processi concorrenti assenti, VRAM libera stimata 14,266 GiB. Il prototipo leggeva la scala a un offset doppio rispetto al buffer compatto B6. Log `build/b6-cm/check-v1.log`. Esito: correggere l’offset e ripetere. Il primo risultato prestazionale resta indicativo.

- **25/11/2026, Codex**, B6 idea 2, ISA v1: RX 9070, 14,336 GiB stimati liberi, processi concorrenti assenti. TRL3 5120 → 17408 a 4 token 156,44 µs in una prova singola con estrazione ISA. Il driver ha emesso v_wmma_i32_16x16x16_iu8 con entrambi gli ingressi con segno. Log `build/b6-cm/isa-v1-real.log` e `isa-trellis_verify_cm-0.txt`. Esito: validare e misurare A/B senza diagnostica.

- **25/11/2026, Codex**, B6 idea 2, ripetizione ISA: il primo cambio del wrapper non aveva sostituito il comando test. Ancora 0 casi eseguiti. Corretto il comando e controllato il contenuto del wrapper.

- **25/11/2026, Codex**, B6 idea 2, tentativo ISA v1: preflight Windows riuscito, processi assenti e 14,336 GiB stimati liberi. Il caso 5120 → 17408 è solo perf, quindi in modalità test sono stati eseguiti 0 casi. Esito: usare perf per ISA e suite completa per correttezza.

- **25/11/2026, Codex**, B6 idea 2, secondo preflight ISA: elenco processi accessibile, ma il wrapper cercava vram-free.py nella nuova cartella dei log. Nessuna prova GPU avviata. Copiato lo script WDDM già usato in B6.

- **25/11/2026, Codex**, B6 idea 2, primo preflight ISA: nel sandbox Get-CimInstance non ha accesso alla lista dei worker Python/Node. Nessun test GPU avviato. Esito: rifare il preflight con accesso ai processi Windows prima di misurare.

- **25/11/2026, Codex**, B6 idea 2, build Windows Release riuscita (BUILD_OK). Shader integrato nella stessa dispatch del dot4 solo in opt-in e con la forma Vulkan richiesta. Riutilizza le attivazioni INT8 e la scala per tile da 16. Log `build/b6-cm/build-v1.log`. Esito: verificare un caso TRL3 e poi misurare.

- **25/11/2026, Codex**, B6 idea 2, preflight: sulla RX 9070 il driver AMD 26.11.2 (LLPC) espone `VK_KHR_cooperative_matrix`, forma subgroup 16×16×16 S8/S8/S32/S32 senza saturazione, oltre alle forme miste. Verificato con `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`; `vulkaninfo` da solo mostra estensione e feature. Primo shader compilato con glslc 1.4.363.0 e accettato da spirv-val. Riferimenti: PR [Nari Labs #5](https://github.com/nari-labs/nari-llama.cpp/pull/5) per prefill CUDA INT8 e [upstream #30106](https://github.com/ggml-org/llama.cpp/pull/30106) per tile Vulkan piccoli. Log e sonda: `build/b6-cm/capabilities.*`. Esito: misurare il prototipo mantenendo le scale ogni 16 input.

- **25/11/2026, Codex**, A5b finale: microbenchmark A-B-B-A, 5120 → 17408. A un token TRL3 120,43 → 118,75 µs (-1,4%), TRL4 112,91 → 112,70 µs. A quattro token TRL3 196,07 → 197,87 e TRL4 196,55 → 197,48 µs, prodotti a quattro token esclusi da A5b. Prima dei run assenti processi concorrenti, 14,08–14,10 GiB stimati liberi. Esito: accettate due letture dirette a un token per i casi che rientrano in 64 bit, sulla base della parità esatta, del piccolo guadagno breve e del risultato neutro a 32K. Log `build/a5b/final-perf-*`.

- **25/11/2026, Codex**, A5b non committato: B3 finale A-B-B-A con 32.768 token già in cache, una ripetizione per run. `tg64` 21,77/21,28/21,45/20,98, medie 21,37 → 21,36 t/s. `pp512` 422,40 → 428,68, stessa GPU e offload, nessun cambiamento ai kernel di prefill. Picco 11,86 GiB dedicati + 0,03 condivisi in A e B. Prima dei run nessun processo concorrente e circa 14,04 GiB stimati liberi. Esito: nessun guadagno di generazione misurato a 32K. Log `build/a5b/final-b3-*`.

- **25/11/2026, Codex**, A5b non committato: B2 finale A-B-B-A, 3 ripetizioni per run. `tg64` 24,00/24,13/24,38/24,08 t/s, medie 24,04 → 24,25 (+0,9%). `pp512` medio 595,50 → 612,69 t/s, con variabilità fra i run. Nessuna modifica al prodotto di prefill e nessuna attribuzione di quel delta ad A5b. Picco dedicato 11,11 GiB, 14,03–14,05 GiB stimati liberi e nessun processo concorrente prima dei run. Esito: beneficio breve piccolo, provare B3 prima di decidere. Log `build/a5b/final-b2-*`.

- **25/11/2026, Codex**, A5b non committato: build finale con modifica solo negli shader, senza pipeline supplementari. 416/416 Vulkan, CTest 4/4. Logits con la stessa continuazione del riferimento CPU B6: top-1 31/31 e greedy uguale, KL max 1,960e-4 a microbatch 1 e 2,101e-4 a microbatch 4. Catture Vulkan identiche bit per bit a B6. Esito: correttezza del dispatch finale verificata. Log `build/a5b/final-ops.log`, `final-ctest.log` e `final-logits-*`.

- **25/11/2026, Codex**, A5b non committato: B2 sul GGUF, A-B-C-C-B-A, riferimento `522f29491`, stesso B6 opt-in su tutti i run. Prefill/tg64 medi A 594,75/23,87 t/s, B due shuffle 601,85/23,63, C letture dirette 606,68/24,33. `tg64` A 23,72 e 24,01, C 24,34 e 24,32 (+1,9% medio). Picco 11,11 GiB dedicati per tutti. Nessun processo concorrente, ~2,13 GiB occupati prima dei run. Esito: selezionato il solo kernel da un token con letture dirette per TRL1–4. La variante multispeculativa mantiene il suo decoder. Patch dei prototipi `build/a5b/two-words-prototypes.patch`, log `b2-*`.

- **25/11/2026, Codex**, A5b non committato: estrazione con due parole, 131.072 stati confrontati offline con l'oracolo a tre parole. Perf A-B-C-C-B-A, A = B6, B = due shuffle, C = due letture dirette. TRL3 1/4 token A 122,95/194,44, B 122,92/200,82, C 117,62/194,71 µs. TRL4 A 110,41/188,68, B 117,70/212,35, C 112,90/197,62 µs. Vulkan 416/416 per B e C. Dedicata occupata fino a 2,15 GiB prima dei run, nessun processo concorrente. Esito: verificare il guadagno TRL3 a un token sul GGUF completo, i microbenchmark non giustificano il cambio per quattro token. Log `build/a5b/perf2-*` e `two-words-correctness-*`.

- **25/11/2026, Codex**, A5b non committato: prima variante tre shuffle, A-B-B-A su 5120 → 17408 con B6 attivo. TRL3 1/4 token 127,41/202,75 → 125,27/202,35 µs. TRL4 115,80/207,31 → 116,10/214,31 µs. 416/416 Vulkan superati e `ds_bpermute_b32` confermato nell'ISA AMD. 14,18–14,23 GiB stimati liberi e nessun processo concorrente. Esito: non adottata, costo dello shuffle superiore al guadagno in parte dei casi. Patch e log `build/a5b/three-shuffle.patch`, `perf-*`, `shared-correctness.log`. Provare entrambi i modi a due parole: per ogni lane TRL1–4, `offset + 16 + 7 * bits <= 64`.

- **25/11/2026, Codex**, A5b non committato: verificati offline i due indirizzi e l'offset della seconda finestra per TRL1–4 su tutte le 32 lane, incluso l'avvolgimento a fine tile. 128 mappature uguali all'indirizzamento di B6. Esito: verificare il costo effettivo degli shuffle sulla RX 9070. Build `build/a5b/build.log`.

- **25/11/2026, Codex**, B6: DFlash2 D2 sui tre prompt testuali dello script OpenClaw, seed 42 condiviso, contesto 114.688, cache Q4, FA, batch/microbatch 512 e proiettore caricato su CPU. Breve/32K/65K: gen 26,44/28,42/28,46 → 29,17/31,77/30,66 t/s (+10,3%/+11,8%/+7,7%), ciclo stimato 84,15/90,63/90,50 → 76,28/81,07/84,01 ms. Testo identico per SHA-256 e accettazione identica (281/689, 156/297, 157/297). A 65K, 33.503 token nuovi e 31.523 in cache. Picco 13,65 GiB dedicati + 0,10 condivisi per entrambi. Prima di ciascun avvio nessun processo concorrente, 14,63/14,64 GiB dedicati stimati liberi. Esito: accettata idea 1 come opt-in via `GGML_VK_TRELLIS_Q8_PACKED=1`. Log `build/b6/server-d2-*`, runner `build/b6/server_b6.py`. Configurazione OpenClaw di produzione intatta.

- **25/11/2026, Codex**, B6 non committato: DFlash2 B1 A-B-B-A, prompt ripetitivo ricostruito da 1.527 token, 256 in generazione, contesto 32.768, cache Q4, microbatch 512, bozza 3 token con rejection. Vecchio 43,79/44,21 → compatto 47,48/49,28 t/s, medie 44,00 → 48,38 (+10,0%). Ciclo stimato 90,56 → 82,38 ms, tutti i run accettano 192/192 e restituiscono lo stesso testo. Picco dedicato ~11,85 GiB in entrambi. Nessun processo concorrente, 14,62 GiB liberi stimati prima di ogni avvio. Esito: guadagno end-to-end ripetibile sul prompt favorevole. Log `build/b6/server-b1-*`.

- **25/11/2026, Codex**, B6 non committato: conferma di `test-backend-ops perf` sul dispatch finale, TRL3 1/4 token 120,73/217,93 → 119,42/191,84 µs, TRL4 112,41/210,11 → 110,11/196,82 µs. Stesso kernel a 1 token; rapporto a 4/1 token TRL3 da 1,81 a 1,61, sopra il traguardo 1,3. `tg64` A-B-B-A, 3 ripetizioni per run, 24,979/24,966/24,952/24,984 t/s, medie 24,98 → 24,96. Cache Q4, Flash Attention, batch/microbatch 512, 65/65 layer Vulkan. Prima delle misure assenti processi concorrenti, 14,62 GiB dedicati stimati liberi. Esito: guadagno nel prodotto a 4 token e generazione singola stabile. Log `build/b6/perf-final-*` e `tg-*`.

- **25/11/2026, Codex**, B6 non committato: `test-trellis-model` sul GGUF 3.0 bpw, stessa continuazione CPU, 31 posizioni. Microbatch 4 vecchio/compatto: top-1 31/31, greedy identico, KL media 2,985e-5 e massima 2,101e-4; SHA-256 dei logits GPU identico tra i due percorsi. Microbatch 1 con B6 attivo: top-1 31/31, KL media 2,824e-5, massima 1,960e-4. Esito: stessa numerica del percorso int8 A5a. Log `build/b6/logits-*`.

- **25/11/2026, Codex**, B6 non committato: ricompilata la variante selettiva, attiva con `GGML_VK_TRELLIS_Q8_PACKED=1` solo per 2–4 token e con accelerazione Vulkan del dot4 unsigned e misto. L'ISA del driver AMD in `build/b6/isa-trellis_matvec_q8x4_packed-*.txt` contiene `v_dot4_u32_u8` per le somme e `v_dot4_i32_iu8` per il prodotto. 416/416 casi Vulkan e CTest 4/4 superati. Esito: percorso hardware e correttezza sintetica verificati; proseguire con logits del GGUF e benchmark.

- **25/11/2026, Codex**, B6 non committato: idea 1 compilata. `OpUDot` per sommare le parti di ciascun peso e `OpSUDot` per moltiplicare quattro pesi da 8 bit senza segno con quattro attivazioni int8. Primo `test-backend-ops perf`, forma 5120 → 17408, A-B-B-A con `GGML_VK_TRELLIS_Q8_PACKED=0/1` sullo stesso binario. TRL3 1/4 token 126,67/221,81 → 134,88/201,66 µs; TRL4 118,18/227,40 → 125,49/206,72 µs. `build/ref-bin` nella stessa sessione: TRL3 130,17/236,32 e TRL4 123,11/223,20 µs. 14,62 GiB dedicati stimati liberi prima dei run, nessun processo concorrente. Esito: variante utile a 4 token (-9,1% / -9,1%), più lenta a 1 token, rapporto previsto TRL3 con dispatch selettivo ~1,59. Da verificare ISA del driver e guadagno del server. Log `build/b6/perf-*`.

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
