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

- **Ultimo aggiornamento**. 26/11/2026, Codex, F4 pack chat, misura server del pack esatto.
- **Build di riferimento**. Usare `build/ref-bin` per i confronti con F4 e verificare il commit con `llama-server --version`. La verifica F4 resta opt-in con `GGML_VK_TRELLIS_VERIFY_I32=1`. La C8 con stamp 48739f77c, equivalente al commit riscritto 805178e04, è archiviata in `build/f4-pack-server/ref-before-f4`.
- **Confronto iniziale in stile OpenClaw (D2, 25/11)**: `D:\Progetti\openclaw\tools\Run-Qwen38-VulkanAB.ps1 -Backends nariq-dflash3r,fork-dflash3r` con `OPENCLAW_AB_UB=512` (varianti `nariq` e `nariq-dflash3r` aggiunte allo script; la build NariQ si cambia con `OPENCLAW_AB_NARIQDIR`). Report `D:\Progetti\openclaw\runtime\benchmarks\20261124-212542-qwen38-backend-ab`. NariQ con DFlash2 3 token: generazione 25,4 / 23,9 / 25,6 t/s (breve / 32K / 65K), prefill 522 / 432, accettazione 35-44%, ciclo 84-90 ms, picco VRAM 14,96 GiB. Unsloth sul fork di produzione: 87,0 / 90,0 / 78,3, prefill 911 / 661, accettazione 44-55%, ciclo 27-32 ms, 14,06 GiB. **Su quella build NariQ DFlash2 rendeva poco**: un ciclo costa circa il doppio di un token (~41 ms) e produce ~2,1 token, quindi la generazione resta ~25 t/s come senza bozza. Senza bozza si risparmia anche ~1,3 GiB di VRAM.
- **Dove sta il margine in generazione**: (1) il prodotto a 4 token (verifica) costa ~1,9 volte quello a 1 token: se scendesse a ~1,2 volte, con l'accettazione misurata DFlash2 darebbe ~40 t/s; (2) il prodotto a 1 token (~36 ms per token) limitato dal calcolo della decodifica (vedi "Valutazione onesta"). Entrambi dipendono dal costo della decodifica per peso, per questo la valutazione HIP (E1) è utile.
- **Ultimi risultati**: generazione senza bozza 17,5 → 24,9 t/s; con DFlash2 a 3 token ~41 t/s sul prompt ripetitivo di prova (prima ~10). **Prefill (C3) 150 → ~602 t/s a contesto vuoto, 147 → 472 a 32K** (llama-bench): traguardo minimo del prefill raggiunto. Il percorso cooperativo è ora predefinito (`GGML_VK_TRELLIS_COOPMAT=0` per disattivarlo): non serve più impostare la variabile.
- **Riferimento NVIDIA**: Nari Labs dichiara 88,9 t/s a 4K e 70,7 a 64K su una RTX PRO 6000 (Blackwell, ~1,8 TB/s, circa 2,8 volte la banda della RX 9070), senza speculazione, cache KV BF16. In proporzione alla banda equivarrebbero a ~32 t/s qui. Il GGUF pubblicato non contiene il modulo MTP ("non ancora quantizzato"), quindi MTP non è disponibile; le alternative alla bozza DFlash2 sono `ngram-simple` (nessun modello, utile su testo ripetitivo e codice) o nessuna speculazione.
- **Profilo di un ciclo DFlash2 (B4)**: il ciclo (~83 ms) è dominato dai prodotti Trellis a 4 token del modello (il prodotto a 4 token costa ~1,9 volte quello a 1 token: la decodifica di ogni peso, ~7 istruzioni, si fa una volta, ma ogni token aggiunge un `dot4` per peso e due letture per tile). La bozza costa ~10 ms per ciclo, ~6 nelle bande della testa di uscita Trellis condivisa (normali: 5120 → 15488 a 4 token 197 µs). Le proiezioni da 48 righe erano anomale (corrette in `902513980`).
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
- **Novità in rete (25/11, Claude)**: nel fork Nari Labs sono aperte PR per RDNA4 molto più veloci della PR #3 usata in E1 (tutte su RX 9070 XT, 64 CU contro i nostri 56, Linux, ROCm 10.1, llama-bench):
  - [#7](https://github.com/nari-labs/nari-llama.cpp/pull/7) (HIP, lifetime42): tg32 20,2 → 32,6 t/s, pp512 706 → 898. Decodifica: somma dei byte con `sad_u8` sommata direttamente a 2^23 in virgola mobile (niente conversione intero → float), moltiplicazione spezzata in due `__umul24` (in GLSL non esiste un equivalente che il compilatore non ricomponga), GEMV FP32 con 8 tile di uscita (128 righe) per gruppo e attivazioni condivise in LDS, letture coalescenti da 128 byte, split-K bilanciato su 64 CU, fusioni di lanci (q/k/v, gate/up, bande della testa), Hadamard in una sola passata.
  - [#8](https://github.com/nari-labs/nari-llama.cpp/pull/8) (HIP): prefill a un solo termine FP16 per batch ≥ 64 con tile 128 × 256 (8 wave, 25 KiB di LDS, 2 gruppi per CU): pp512 898 → 1356, pp2048 1318; tg32 32,8. Batch 1-8 su GEMV FP32, 9-63 su WMMA FP16 con residuo. KL contro il percorso scalare: media 1,2e-6, top-1 100%.
  - [#5](https://github.com/nari-labs/nari-llama.cpp/pull/5) (CUDA): prefill INT8 su tensor core con scala potenza di 2 per riga, pp8192 675 → 1424 su RTX 5060 Ti; PPL 2,4655 → 2,4678.
  - Upstream llama.cpp ha integrato a settembre il prodotto int8 coopmat1 per RDNA3/RDNA4 ([ggml-org/llama.cpp#27952](https://github.com/ggml-org/llama.cpp/pull/27952)); riguarda i formati normali (Unsloth), già presente nella base del fork.
  - Confronto con noi (9070 non XT): tg64 ~24,5 contro 32,8 (XT; in proporzione ai CU ~29 qui), pp512 ~754 contro 1356 (~1190 qui). Da verificare con una misura sulla nostra scheda.
- **D2 su build 128**. 32K a cache fredda: NariQ 580,40 -> 726,08 t/s di prefill e 47,87 -> 45,70 t/s di generazione. Unsloth 878,22 -> 882,07 t/s di prefill e 100,26 -> 105,69 t/s di generazione. 65K a cache fredda: NariQ 531,11 -> 644,36 t/s di prefill e 44,00 -> 41,19 t/s di generazione. Unsloth 752,03 -> 750,75 t/s di prefill e 84,24 -> 87,35 t/s di generazione. Picco VRAM dedicata del processo 12,74 / 13,61 GiB (Unsloth / NariQ). Stessi prompt D2 e stessa build 128 per entrambi. Log build/d2-b128.
- **F4 INT32**. Scala su 32 input e accumulo INT32 su due tile, opt-in. TRL3 a quattro token, up 150,94 -> 131,86 us (12,6% in meno), down 174,53 -> 149,22 us (14,5% in meno). TRL4 up 148,92 -> 133,17 us (10,6% in meno), down 172,68 -> 155,10 us (10,2% in meno). DFlash2 A-B-B-A sullo stesso seed: breve 50,99 -> 51,26 t/s, a 32K 46,50 -> 51,60 t/s. Ciclo stimato a 32K 62,41 -> 56,24 ms, accettazione 64,1% -> 64,1%. Vulkan 416/416, CTest 4/4, CPU top-1 31/31 e stessi token greedy. KL massima 0,00013943. Log build/f4-int32. Sul breve il ciclo stimato 59,15 -> 53,38 ms e l'accettazione 67,2% -> 58,6%. Il ciclo è stimato dai millisecondi di generazione per gruppo di tre proposte. A e B usano lo stesso binario sperimentale e le stesse richieste D2 con seed 42. L'opzione cambia la quantizzazione delle attivazioni della verifica. La variante a 64 input è stata archiviata dopo 410/416 casi. La build candidata è in `build/bin` e si attiva con `GGML_VK_TRELLIS_VERIFY_I32=1`.
- **F4 accettazione**. Sui prefissi fissi, overlap A/B 0,437/0,441, 0,545/0,544, 0,306/0,300 per posizione (10/8/8 coppie). Con F4, bozza 0,8 rispetto a 1,0 porta la chat da 55,17 a 60,03 t/s, codice da 55,28 a 54,43 e 32K da 52,92 a 53,25. Le impostazioni di avvio restano quelle del riferimento. Nel profilo GPU seriale, mediana dei grafi di verifica 55,66 -> 51,58 ms. Le operazioni Trellis a quattro token pesano 47,72 ms con F4, con 18,06 ms per up e 10,39 ms per down. Le bande della testa della bozza pesano 2,00 ms. I tempi del profilo includono l'overhead delle misure. Il prototipo che riusa la scala ed elimina una barriera è scartato dopo il microbenchmark. Log build/f4-accept.
- **F4 conferma**. Bozza 0,8 confermata per chat breve nei prompt provati. 52,66 -> 54,21 t/s, accettazione 58,6% -> 61,7%, 6/9 coppie a favore di 0,8. Diagnostico TRL3 a 4 token, con output sintetici. Up 130,99 us con il kernel esatto, 72,52 us con hash e packing sostituiti e 60,11 us saltando anche le letture dei codici. Down 148,55 -> 93,59 -> 70,69 us nelle stesse prove. Il proxy senza WMMA è più lento e cambia il modo in cui il driver carica e usa i frammenti. Questi confronti localizzano il costo della decodifica e non sono una scomposizione additiva. Pack esatto a coppie in F4. trl3 5120 → 17408 132,06 -> 123,88 us. trl3 17408 → 5120 149,44 -> 141,18 us. trl4 5120 → 17408 133,23 -> 125,22 us. trl4 17408 → 5120 156,08 -> 147,14 us. Vulkan 416/416 e CTest 4/4. Le 31 righe dei logits coincidono bit per bit con F4 prima della modifica. Log build/f4-confirm.
- **F4 pack chat**. F4 attiva in entrambe le build, bozza 0,8. Confronto server A-B-B-A su stessi prompt e seed. breve 55.13 -> 59.18 t/s (+7.35%), ciclo stimato 53.86 -> 50.17 ms, token DFlash2 1014/1552 in entrambi. 32K 50.24 -> 53.49 t/s (+6.46%), ciclo stimato 56.20 -> 52.79 ms, token DFlash2 494/816 in entrambi. Guadagno confermato sul server a 32K. Log build/f4-pack-server.
- **Prossimo passo**. Anthropic Claude segue G1 per generazione e verifica DFlash2 e P1 per il prefill, con confronto D2 contro Unsloth e profilo della build di riferimento.
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

## Piano per accelerare insieme generazione e verifica DFlash2

**Responsabile dell'esecuzione: Anthropic Claude.** Claude segue G1–G5 e aggiorna il registro dopo ogni prova.

Analisi da consultare per G4 e per gli spunti sul prefill. [Report originale di Qwen4 Max](Qwen4Max-Analisys.txt).

Il bersaglio comune è il decoder dei pesi Trellis. Il modello resta Qwen3.8-27B NariQ 3.0 bpw, con pesi compressi su Vulkan e confronto CPU sullo stesso GGUF. Il punto di partenza è la build 130, commit `6b782586a`, con i binari in `build/ref-bin`.

### Riferimenti e traguardi

| Misura | Punto di partenza | Primo traguardo |
| --- | --- | --- |
| Generazione senza bozza | 28,6 t/s breve, 27,7 t/s a 32K nei test precedenti | Almeno 30 t/s nel breve e guadagno ripetibile a 32K |
| Generazione DFlash2 con F4 | 59,18 t/s breve, 53,49 t/s a 32K | Almeno 60 t/s a 32K con la stessa resa in token per ciclo |
| Ciclo DFlash2 stimato a 32K | 52,79 ms | Circa 47 ms a pari resa, come riferimento per il traguardo da 60 t/s |

I numeri senza bozza provengono da una build precedente. Prima di sviluppare, rimisurare tutti e quattro i casi sulla build 130 nella stessa sessione. Unsloth resta il riferimento finale con il protocollo D2. I traguardi intermedi servono a decidere dove proseguire. Il limite di VRAM resta quello del progetto.

### Ordine delle prove

1. **G1. Fissare il riferimento.** Misurare `tg64` a contesto breve e con profondità 32.768. Nel server usare gli stessi prompt D2, gli stessi seed e tre proposte DFlash2. F4 attiva in entrambi i lati del confronto con bozza. Congelare la temperatura della bozza per scenario e usare gli stessi valori in A e B. Registrare anche prefill, throughput, VRAM, token proposti e accettati, token prodotti per ciclo e durata del ciclo.
2. **G2. Localizzare il costo del decoder.** Misurare separatamente, a uno e quattro token, le forme MLP `5120 → 17408` e `17408 → 5120` per TRL3/TRL4. Tenere un controllo sulle bande della testa. Confrontare l'ISA emessa dal driver e verificare letture dei codici, estrazione delle finestre e hash a 32 bit. Usare gli shader sintetici solo nei microbenchmark e interpretare le differenze come indizi, perché il compilatore può cambiare le altre istruzioni del kernel.
3. **G3. Provare prima una modifica esatta riutilizzabile.** Per TRL1–4, verificare nell’ISA dei kernel da uno e quattro token se indici delle parole e shift delle finestre sono già calcolati fuori dal ciclo delle tile. Se restano ricalcoli, provare a calcolarli una volta per lane usando `trellis_words8`, `trellis_load8` e `trellis_states8_from`, già usati nel prodotto grande. Misurare questa sola modifica. Se l’estrazione resta costosa, provare su TRL3 le sequenze di 32 stati dell’issue #6, con shift costanti e letture delle parole di confine. Mantenere il percorso attuale per gli altri tipi. Valutare separatamente un hash esatto alternativo se l’ISA conferma un cambio nel lavoro effettivo. [Nari Labs #6, sequenze di 32 stati](https://github.com/nari-labs/nari-llama.cpp/issues/6#4-32-step-runs-every-window-becomes-a-compile-time-constant)
4. **G4. Completare il percorso con più margine.** Se una modifica comune favorisce soprattutto un kernel, misurare separatamente il costo residuo dell'altro. A un token, esaminare il dot4 e le letture delle attivazioni. A quattro token, esaminare decodifica e preparazione delle tile per F4. Tenere separate implementazioni e misure quando i due kernel richiedono strategie diverse. I due guadagni devono convivere nella stessa build finale.
   - **Prova F4 dopo G3.** Nei casi in cui è già attiva la fusione d’ingresso, unire la scrittura delle attivazioni ruotate FP32 e la quantizzazione F4 con scala comune su 32 input. Partire dalle due proiezioni MLP a quattro token. Confrontare byte delle attivazioni impacchettate e scale con il percorso attuale, poi eseguire i controlli del protocollo C. Misurare il tempo delle proiezioni e il ciclo DFlash2 con tre proposte sugli stessi prompt e seed. Registrare separatamente il risultato nel percorso con bozza.
5. **G5. Verificare nel server.** Solo dopo i controlli numerici, ripetere `tg64` nel breve e a 32K, poi D2 nel breve e a 32K con F4. Usare l'ordine A-B-B-A per risolvere variazioni piccole e considerare separatamente prompt e seed. Se la candidata supera le prove, controllare prefill, VRAM e generazione con DFlash2 a 65K prima di proporla per OpenClaw.

### Criteri di qualità e promozione

- Per modifiche al solo decoder dei pesi, verificare che i valori interi decodificati coincidano con la CPU su tutte le posizioni della tile per TRL1–6. Includere finestre che attraversano una parola e il bordo ciclico, con valori che dipendono dalle ultime parole. Verificare che il nuovo schema copra tutte le 40 parole in TRL5 e le 48 in TRL6. Controllare gli offset effettivi prima di usare un funnel shift con shift a cinque bit. A parità di ordine del calcolo, richiedere anche logits bit per bit uguali alla build di riferimento sui prefissi forzati. Confrontare a microbatch 1 e 4, includendo una coda e un contesto lungo. [Nari Labs #6, caricamento delle tile](https://github.com/nari-labs/nari-llama.cpp/issues/6#1-bug-the-coded-tile-can-need-more-words-than-one-wave-has-lanes)
- Eseguire il protocollo C, con suite Vulkan Trellis 416/416, CTest 4/4 e confronto `test-trellis-model` contro CPU. Misurare KL, top-1 e token greedy contro il riferimento. Verificare NaN, offload e percorso Vulkan nei log. Una variante che cambia scale o ordine di accumulo richiede un controllo aggiuntivo su KL e perplexity WikiText-2 sugli stessi campioni, oltre al confronto di accettazione su più prompt e seed. Applicare le tolleranze del protocollo esistente.
- Per una modifica esatta nel percorso DFlash2, controllare proposte e accettazione a seed identico. Se divergono, acquisire logits e distribuzioni della bozza e del target sugli stessi prefissi prima di attribuire il cambiamento alla velocità. Per modifiche numeriche, confrontare anche accettazione, token prodotti per ciclo e andamento delle richieste su più seed.
- Accettare una modifica comune se il guadagno è ripetibile almeno su uno dei due percorsi e l'altro resta entro la variabilità misurata. Il traguardo della fase richiede un miglioramento su entrambi i percorsi nella stessa build e assenza di regressioni riproducibili su qualità, prefill e VRAM. Registrare ogni prova, con preflight GPU e log, secondo le regole di lavoro.

## Piano per avvicinare il prefill a Unsloth

**Responsabile dell'esecuzione (per ora): Anthropic Claude.** Claude segue P1–P5 e aggiorna il registro dopo ogni prova.

Il confronto principale è il tempo di prefill di una richiesta chat a cache fredda su RX 9070. Usare gli stessi messaggi a 32K e 65K del D2, con thinking e bozza configurati allo stesso modo per ogni A/B, Flash Attention, KV `q4_0`, batch e microbatch target 512 e contesto allocato a 114.688. Misurare anche il tempo al primo token, il prefill del seguito a cache calda e il picco VRAM dedicata e condivisa. Confrontare NariQ e Unsloth sulla stessa build quando il binario supporta entrambi, con lo stesso microbatch target e registrando l'eventuale differenza nella bozza.

| D2 a cache fredda | NariQ, build 128 | Unsloth, build 128 | Primo traguardo | Traguardo successivo |
| --- | --- | --- | --- | --- |
| 32.145 token | 726 t/s, 44,27 s | 882 t/s, 36,44 s | Almeno 90% del throughput Unsloth | Arrivare entro il 5% |
| 65.145 token | 644 t/s, 101,10 s | 751 t/s, 86,77 s | Almeno 90% del throughput Unsloth | Arrivare entro il 5% |

Le velocità della build 128 sono il confronto storico. Il primo A/B rifissa i valori con il riferimento corrente. Valutare i traguardi sulla nuova sessione e a qualità, impostazioni e VRAM comparabili.

### Ordine delle prove prefill

1. **P1. Misurare riferimento e profilo.** Misurare NariQ e Unsloth nello stesso D2 a 32K e 65K. Per NariQ, acquisire anche `pp512`, `pp8192` e `pp512` con 32K già in cache per distinguere costo del prodotto e costo del contesto. Ripetere un campione con il profiler Vulkan attivo e separare preparazione delle attivazioni, rotazioni, prodotto Trellis e attenzione. Usare i run senza profiler per il throughput.
2. **P2. Prima variante F5 sul prodotto 128 × 256.** Il kernel grande scrive già due FP16 per parola LDS e usa il *magic bias* FP32. Provare il bias FP16 dell’issue #6 costruendo il valore `1024 + 4 × somma` dalla somma dei quattro campi a 6 bit. Sottrarre esattamente 1528 per ottenere `4 × peso`, con il fattore già compensato nella preparazione delle attivazioni. Verificare tutte le somme intere da 0 a 252 e confrontare i valori FP16 delle tile prima di misurare. Controllare nell’ISA se la conversione a coppie riduce istruzioni o registri e verificare il costo delle scritture LDS. Misurare prima le due proiezioni MLP e poi il modello completo con `pp512` e `pp8192`. [Nari Labs #6, bias FP16](https://github.com/nari-labs/nari-llama.cpp/issues/6#3-cuts-to-the-per-value-cost-kept-the-prs-kernel-structure)
3. **P3. Ridurre sincronizzazioni o traffico delle tile.** Sullo stesso kernel provare un passo K su quattro tile, con controlli sul limite LDS del driver, registri, occupazione e code K. Come prova separata, verificare se letture coalescenti dei codici comprimono il costo osservato. Includere almeno le due forme MLP grandi e una forma stretta. Tenere la variante solo se passa il caso con coda e migliora il modello a batch 512 e a contesto lungo. Usare le soglie `MM_BIG_MIN` e `MM_BIG_SPLIT` esistenti per confronti mirati quando il profilo mostra un problema di distribuzione del lavoro.
4. **P4. Verificare Flash Attention a contesto lungo.** Verificare C4, cioè che l'attenzione con KV `q4_0` selezioni il percorso atteso anche con NariQ. Misurare quanto pesa a 32K e 65K prima di modificare il dispatch. Tenere comparabili i risultati D2 con e senza cache del prompt e annotare i token effettivamente elaborati.
5. **P5. Qualificare e confrontare con Unsloth.** Ripetere D2 con la variante candidata prima a 32K, poi a 65K. Per guadagni sotto il 5% usare A-B-B-A. Riportare throughput, secondi di prefill e tempo al primo token, VRAM del processo, seguito a cache calda e controllo della generazione con e senza bozza. Ripetere con almeno un altro prompt lungo prima di proporre la build per OpenClaw.

### Vincoli di qualità del prefill

- Applicare il protocollo C, con Vulkan Trellis 416/416 e CTest 4/4. Per la conversione esatta delle coppie controllare i valori FP16 della tile. Per modifiche al passo K, confrontare logits e KL contro la CPU e il riferimento Vulkan. Controllare batch pieni e code, compreso il campione da 385 token, e una finestra lunga.
- Ripetere la perplexity WikiText-2 sulle stesse 40 finestre, usare la stessa cache e confrontare gli stessi token nei test sui logits. Verificare offload, NaN, picchi VRAM dedicati e condivisi. Le varianti con attivazioni INT8 o altri cambi numerici restano sperimentali finché passano le tolleranze già previste. Il prototipo C6 a un solo termine INT8 ha fallito casi numerici, quindi la priorità va alle modifiche FP16.
- Registrare ogni prova nel piano con il preflight. Se i lavori di generazione e prefill si svolgono in parallelo, coordinare una sola misura GPU e usare un riferimento annotato per ciascun confronto.

## Attività

Stato: `da fare`, `in corso (chi)`, `fatto`, `scartato`. Le stime di guadagno sono ipotesi da verificare.

### Fase A: generazione (un token alla volta)

| ID | Attività | Stato | Note e guadagno atteso |
| --- | --- | --- | --- |
| A1 | Baseline con B2 e B3 per NariQ e Unsloth con lo stesso binario; profilo di un token; tipi di rotazione usati dal modello | fatto | Vedi registro del 24/11 sera |
| A2 | Rotazioni in un solo passaggio | scartato | Il modello usa solo rotazioni a blocchi (128-4096) già in un passaggio in memoria condivisa; saltare del tutto la rotazione d'ingresso dà solo +4% (17,6 → 18,3 t/s) |
| A3 | Riuso della rotazione d'ingresso fra proiezioni con lo stesso input | bassa priorità | Vale al massimo ~1-2 ms per token |
| A4 | Fondere input, rotazione e uscita con il prodotto | bassa priorità | Tutto ciò che non è prodotto vale 5,2 ms per token |
| A5a | **fatto** (`565d38c7f`, +33%). **Prodotto con attivazioni int8 e `dot4`**: un passaggio prima del prodotto quantizza le attivazioni ruotate in int8 per gruppi da 16 (scala float e somma intera per gruppo), nell'ordine in cui le leggono i lane (colonne 2q, 2q+1, 2q+8, 2q+9 di ogni tile). Nel prodotto, per ogni peso: `b = (h >> 2) & 0x3f3f3f3f` e `acc += dot4(b, a_rep)` con `a_rep` = byte dell'attivazione replicato 4 volte (`v_perm`), meno `126 * somma_attivazioni` per tile; scala tile × scala attivazioni in float una volta per tile | da fare | Da ~15 a ~9 istruzioni per peso e una lettura di attivazioni invece di due. Atteso: prodotto da ~52 a ~30-35 ms, generazione ~25-28 t/s. Cambia la numerica (come q8_1 in llama.cpp): controllare logits CPU/Vulkan e KL |
| A5b | Riuso delle parole Trellis fra le due finestre di una lane | fatto | Due letture dirette nel kernel da un token TRL1–4. `tg64` +0,9% nel breve, stabile a 32K. Shuffle scartato dopo A/B sulla RX 9070 |
| A5c | Moltiplicazione dello stato (`state * 0x83dcd12d`, 32 bit, un quarto di velocità su RDNA): cercare una forma che il compilatore non ricomponga | **fatto** (`d02cd6d2e`: metà e shift opachi, `v_mul_u32_u24`; tg64 +3,7%, pp7 +6,5%) | Spezzarla in due prodotti a 16 bit non serve: il compilatore li ricompone in una sola moltiplicazione a 32 bit |

### Fase B: verifica DFlash2 (2-4 colonne)

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| B1 | Misurare DFlash2 con 3 token anticipati su NariQ | fatto | Dopo B2: senza bozza 24,0, bozza 2/3/4 token 38,7 / 42,2 / 17,1 t/s (prompt ripetitivo). Ciclo ~87 ms |
| B4 | Profilo di un ciclo DFlash2 (modello a 4 token, bozza, teste di uscita) | fatto | Vedi "Stato attuale" e il registro del 25/11. Corrette le proiezioni da 48 righe (`902513980`) |
| B5 | Estendere la verifica a 5–8 token (4–7 token draft) | **fatto** (`4d45ef0ca`, F4 a 5-16 token; predefinita da `96d164cfd`) | Riutilizzare la tile dei pesi decodificata in due prodotti WMMA e conservare la stessa somma FP32 per gruppo. Confrontare separatamente forma 5120 → 17408 e D2, partendo da 4 token draft. |
| B2 | Prodotto a più colonne: decodificare ogni peso una volta per tutte le colonne | fatto (`b14dae893`) | 4 token: TRL3 407 → 216 µs; 1 token invariato grazie a una variante dedicata | Con A5a ogni colonna in più costa un solo `dot4` per peso. Oggi `trellis_matvec` (batch ≤ 4) lancia un gruppo per token (`gl_WorkGroupID.y`) e decodifica i pesi una volta per token |
| B3 | Scegliere i token anticipati (2-4) in base ai tempi misurati | **fatto**: 6 token (63,5 t/s di media contro 60,4 con 5 e 63,9 più irregolare con 8) | |
| B6 | Verifica a 4 token più economica | Idee 1 e 2 validate, opt-in | Rapporti 4/1 1,35 e 1,38. D2 37,23/40,32/39,90 t/s. Il traguardo 1,3 resta aperto. |

### Fase C: prefill

| ID | Attività | Stato | Note |
| --- | --- | --- | --- |
| C1 | Profilo del prefill a 512 token per fase | da fare | Il 24/11: prodotto 89% del tempo |
| C2 | Preparare `act_hi`/`act_lo` una volta per proiezione invece che per gruppo | fatto in C3 | `trellis_prep16.comp`; il residuo è opzionale (`GGML_VK_TRELLIS_MM_RESIDUAL=1`, 28% più lento, usato dal test elemento per elemento) |
| C3 | Prodotto cooperativo con tile più grandi entro i 32 KB di memoria condivisa del driver Windows; decodifica dei pesi una volta per tile | fatto (`35e3c9cde`) | 32 uscite × 128 token per gruppo, 256 thread che decodificano 8 tile ogni 64 input, attivazioni FP16 preparate una volta per pezzo con una potenza di 2 per token. pp512 150 → ~490, 32K 147 → 395 |
| C5 | Prodotto cooperativo: forma del gruppo di lavoro | fatto (`1df83dfb5`) | 64 righe × 32 input per passo, migliore di 2×4, 4×4, 4×1, 8×1, 8×2 (vedi registro); k diviso per le proiezioni da 48 righe |
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
| F4 | Verifica a quattro token e costo della testa della bozza | INT32 su 32 input validato come candidato opt-in | Guadagno ripetuto a 32K sullo stesso prompt. La testa della bozza e la verifica su più prompt restano aperte. |
| F5 | Prefill: conversione diretta a coppie FP16 e passo K su quattro tile | da fare | P2 e P3. Il kernel attuale scrive già due FP16 per parola LDS. |

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


### Hash riscritti il 26/11/2026

Il 26/11 l'email d'autore dei commit non ancora pubblicati è stata portata all'indirizzo noreply di GitHub: gli hash sono cambiati e il testo usa quelli nuovi. I binari compilati prima (per esempio `build/ref-bin`) riportano in `--version` quelli vecchi.

| Vecchio | Nuovo | Commit |
| --- | --- | --- |
| `12fed4772` | `3805d1cf8` | vulkan: Trellis grid byte sum without a 32-bit multiply; GGML_VK_TRELLIS_DEBUG prints the  |
| `a66f18956` | `565d38c7f` | vulkan: Trellis decode product on int8 activations with dot4 for up to 4 tokens |
| `aeeecc701` | `80925fd2b` | docs: NARIQ_PIANO.md after the int8 decode product |
| `3b8cc30e7` | `b14dae893` | vulkan: Trellis int8 decode product for up to 4 tokens in one workgroup |
| `6d5efbdd4` | `f03482f19` | docs: NARIQ_PIANO.md after the multi-token int8 product and the DFlash2 test |
| `7a5cae586` | `902513980` | vulkan: split k for Trellis int8 products with few outputs; Trellis shapes in the perf log |
| `be382e0e0` | `d57b31957` | docs: NARIQ_PIANO.md after the DFlash2 cycle profile; prefill first |
| `af07eaa13` | `35e3c9cde` | vulkan: Trellis prompt-batch product on FP16 cooperative matrices, 32 outputs x 128 tokens |
| `e9b2bc3b3` | `45a58fd05` | docs: NARIQ_PIANO.md after the cooperative prompt-batch product |
| `71b71362e` | `1df83dfb5` | vulkan: Trellis cooperative prompt product on 64 outputs x 32 inputs per step; split k for |
| `07e82bc00` | `fd49b1e84` | docs: NARIQ_PIANO.md after the 64 x 32 cooperative product; HIP as an option to evaluate |
| `d4d3eb6e3` | `7835d4f05` | common: block drafts (DFlash, DSpark) use a micro-batch of at most 64 tokens |
| `073ab7d8f` | `a289c93b6` | docs: NARIQ_PIANO.md after the OpenClaw-style comparison; operational plan for the HIP eva |
| `d650f4beb` | `8ac54082a` | docs: NARIQ_PIANO.md, HIP evaluation handed to Codex |
| `bb09a6076` | `b83a70b47` | docs: NARIQ_PIANO.md, E1 toolchain on ROCm 10.2.0 (TheRock) in a separate venv |
| `fa1b47f2f` | `2d95f55e6` | docs: NARIQ_PIANO.md, HIP evaluation (E1) results from Codex, HIP closed; B6 started |
| `75848a838` | `8bc27c23a` | docs: NARIQ_PIANO.md, B6 handed to Codex with a starting analysis |
| `522f29491` | `06d57e2fe` | vulkan: pack Trellis dot4 verification batches |
| `92e2b6c0a` | `b7ddb5132` | vulkan: reuse Trellis code words for single-token decode |
| `10339c849` | `ee969e9c3` | vulkan: accelerate Trellis verification with INT8 WMMA |
| `60b4d242d` | `1eca433a3` | vulkan: reuse Trellis prefill weights across 256 tokens |
| `09b633ca2` | `439e39c60` | vulkan: accelerate Trellis Hadamard with subgroup shuffles |
| `48739f77c` | `805178e04` | vulkan: reuse Trellis state words in FP16 prefill |
| `4886574cd` | `65fe64270` | docs: record NariQ C8 comparison with OpenClaw production |
| `c1b1b50ed` | `5ddc20e00` | docs: NARIQ_PIANO.md, phase F (generation first, then prefill) |
| `5af77d26f` | `cf0d63784` | vulkan: Trellis INT8 verification product for 5-16 tokens (one token per cooperative-matri |
| `fced3af46` | `7503533c7` | docs: NARIQ_PIANO.md, community RDNA4 HIP and INT8 prefill pull requests in the Nari Labs  |
| `cc5807441` | `7b90766a4` | vulkan: Trellis prompt product with 128 outputs x 256 tokens per workgroup (GGML_VK_TRELLI |
| `2f9e5627e` | `13e79b729` | docs: NARIQ_PIANO.md, 128x256 prompt product and HIP PR #8 measured on the RX 9070 |
| `d7ddf5dd9` | `d9f47ce4b` | docs: NARIQ_PIANO.md, one-token product experiment and per-operation comparison with HIP |
| `72fdbdce7` | `4287d52b7` | vulkan: fuse the input and output stages of Trellis products for batches up to 15 tokens ( |
| `5ebe5f011` | `ea6dbac44` | vulkan: cheaper one-token Trellis int8 product (masked state bytes, chained dot4, precompu |
| `862bc72c6` | `c9046d52d` | docs: NARIQ_PIANO.md, day summary with the Vulkan and HIP comparison |
| `7e6212058` | `de886a1e1` | vulkan: Trellis packed int8 verification, wide/128x256 prompt product, block shuffles and  |

## Registro delle prove

Voci più recenti in alto. Formato: data, chi, commit o "non committato", cosa, numeri (riferimento → candidato), esito.

- **27/11/2026, Claude**, commit di questa voce (solo registro): HIP PR #8 contro Vulkan attuale, stessa sessione.
  - **Ramo `nariq/experimental-amd`** (`84a9af8c5`): PR #3 più un solo commit (`trellis-hip-rdna4.cuh`, percorso RDNA4); la PR #8 (`057395dec`, worktree `D:\Progetti\NariQ HIP RDNA4\pr8`, già compilata il 26/11 con ROCm 10.2) contiene anche la #7 ed è la versione HIP più avanzata.
  - **llama-bench** (cache q4_0, FA, b/ub 512; Vulkan `d127cb754`, HIP PR #8, ambiente di `e1-logs/sdk-paths.json`): pp4 97,6 / 62,0; pp7 168,8 / 69,8; pp512 1130 / 1230; tg64 30,1 / 31,8. A 32K: pp7 120,0 / 57,7; pp512 764 / 811; tg64 28,7 / 27,8. HIP avanti nel prefill (+6-9%) e a 1 token a contesto vuoto (+6%); Vulkan avanti nella verifica della bozza (×2,1-2,4) e a 1 token a 32K (+3%). HIP non ha l'integrazione DFlash2 con la testa Trellis condivisa.
  - **Stima di un porting completo su HIP**: generazione con bozza ~+5% (verifica portata al rapporto di Vulkan, 1,25 volte il costo di un token, sul token HIP da 31,4 ms), prefill +6-9%; costo: riscrivere in HIP F4, catena MLP, trasformate nei registri, lettura anticipata e integrazione DFlash, più il runtime ROCm per OpenClaw su Windows.

- **27/11/2026, Claude**, commit di questa voce (solo registro; patch e sorgenti in `build/p1/archive`): traccia CPU del prefill con bozza, microbenchmark HIP della tabella da 64 KB.
  - **Traccia del prefill** (`LLAMA_PREFILL_TRACE`, temporanea; prompt da 32.145 token, NariQ bozza 6): 64 iniezioni dalla RAM di ~503 righe: attesa del target 32,83 s (il calcolo del modello), copie 0,19 s, decodifiche della bozza 0,99 s, attesa della bozza 0,06 s; chiamate `llama_decode` del target ~50 ms di CPU per blocco da 512 (Unsloth ~44). Unsloth non passa dalla RAM: le sue feature vanno alla bozza direttamente dalla GPU (`dev_inject`), perché il blocco del target (512) entra nel micro-batch della bozza (512); per NariQ (64) no.
  - **Iniezione diretta in pezzi da 64** (`dflash-dev-inject-parts.patch`: API `llama_dflash_inject_features_rows_async`, soglia di copia in RAM al micro-batch del target): nessuna copia in RAM, accettazione e VRAM identiche, prefill 32K 39,4 → 38,9 s. Con pezzi della bozza da 128 / 256 (VRAM +0,23 / +0,70 GB): 38,8 / 38,5 s. Né le copie, né la sincronizzazione, né il numero di chiamate spiegano i ~3 s di GPU ferma che restano; ipotesi non verificata: caricamenti sincroni degli input della bozza nel backend Vulkan. Archiviata.
  - **HIP, tabella da 64 KB in LDS** (`build/p1/archive/hipbench/lutbench.hip`, ROCm 10.2 del venv della cartella HIP, gfx1201, 28 WGP, LDS 65.536 per gruppo; prodotto a 1 token TRL3 5120 → 17408 con codici sintetici, risultati uguali fra le varianti): hash a 32 bit 148,6 µs, hash con `__umul24` 156,4 µs, tabella in LDS 262-702 µs (al massimo due gruppi da 64 KB per WGP: occupazione troppo bassa, più il caricamento della tabella). Il kernel Vulkan ottimizzato fa 88 µs.
  - **Porting su HIP**: non conviene. Il vantaggio dei 64 KB (tabella) è negativo; il prefill Vulkan è già al 97-98% di Unsloth; i numeri HIP RDNA4 pubblici (PR #8, Linux, RX 9070 XT: pp512 1356, tg32 32,8) riportati alla RX 9070 sono vicini ai nostri (pp512 1128, tg64 30,0); servirebbe riscrivere in HIP tutti i kernel e l'integrazione DFlash.

- **27/11/2026, Claude**, commit di questa voce (solo registro): verifica dell'ipotesi sul costo della bozza nel prefill, rete.
  - **Profilo** (suite `build/g1` con `GGML_VK_PERF_LOGGER=1`, prompt da 32.145 token, bozza 6, build `d127cb754`): con bozza 41,01 s reali, GPU del target 33,78 s (61 blocchi da 512), GPU della bozza 0,53 s (504 iniezioni da 64 token: MUL_MAT, RMS_NORM_MUL, ROPE, SET_ROWS); senza bozza 36,55 s reali, GPU del target 33,92 s. Ipotesi confermata: i ~3,9 s in più sono attesa della GPU, non calcolo. Il server, in sequenza, scarica in RAM le feature di 5 strati del target (25.600 valori per token, ~3,3 GB per 32K token), le copia nel lotto della bozza e le ricarica in 504 decodifiche; il target riparte solo dopo. Non isolato quale delle tre parti pesi di più (serve una traccia della CPU).
  - **Prompt brevi**: chat-42 (153 token) 0,46 s senza bozza contro 1,97-1,98 con bozza; anche Unsloth 0,34 contro ~1,2 s. Ogni richiesta con bozza paga ~1-1,5 s fissi di prompt: nell'uso con Jarvis (un turno, un prompt) pesa più del 7% dei prompt lunghi.
  - **Rete** (27/11): nessuna novità per Vulkan. Nari Labs: `nariq/b12431` fermo a `2644f7de4` (25/11), `nariq/experimental-amd` a `84a9af8c5`, `nariq/act-kv-quant` senza commit nuovi; PR #4 (HIP RX 9070 XT) con un nuovo test su RX 7900 XTX (gfx1100, WSL2, ROCm 7.2), nessun codice nuovo; la nostra PR #9 senza commenti. Upstream llama.cpp: nessuna PR Vulkan RDNA4 nuova trovata.

- **27/11/2026, Claude**, commit di questa voce (solo registro): costo della bozza nel prefill del server.
  - **Server senza bozza** (suite `build/g1`, nuova opzione `-NoDraft`; build `d127cb754`): prefill a 32K NariQ 905,1 t/s (35,52 s), Unsloth 937,8 (34,28 s), 96,5%; prompt da 153 token 333 contro 455 t/s; tg 29,8 contro 39,0. Con la bozza (n-max 6 / 3): NariQ 816-818 t/s (39,4 s, +3,8 s, +11%), Unsloth 898 (35,8 s, +1,5 s, +4%). Il divario del prefill nel server è il costo della bozza, maggiore per NariQ.
  - **Testa della bozza**: i 65.536 id della bozza sono sparsi su tutte le 16 bande della testa Trellis (13.936 nella prima, 449-754 nelle ultime): ritagliare le bande toglierebbe ~15% del vocabolario della bozza. Durante il prompt la bozza non calcola la testa (ramo di iniezione: solo K/V), quindi una testa più piccola non toccherebbe il prefill.
  - **Limite del micro-batch della bozza** (variabile provvisoria, non committata): 128 contro 64, prefill a 32K 813,8 contro 816,5 t/s, VRAM 14,73 contro 14,50 GB, accettazione cambiata. Neanche la dimensione dei pezzi è la causa.
  - **Ipotesi non verificata**: con la bozza il target copia in RAM le feature di 5 strati a ogni blocco da 512 token e si sincronizza; il grafo NariQ, con più lanci per strato, paga di più la registrazione sulla CPU esposta. Da verificare con un profilo CPU/GPU del prefill con bozza.

- **27/11/2026, Claude**, commit di questa voce (solo registro; patch e note in `build/p1/archive`): altre prove su generazione e prefill, tutte scartate.
  - **Prodotto a 1 token dopo l'hash a 24 bit** (ISA): per 8 pesi e lane ~14 istruzioni di estrazione, 24 di hash (`v_mul_u32_u24`, shift, `v_mad_u32_u24`), 8 `and`, 8 `dot4`, ~20 di chiusura del tile e ciclo; il kernel gira in wave64. Due tile per iterazione: 88 → 89,4 µs. Wave32 forzato: 88 → 99,6 µs.
  - **Prefill, modello contro Unsloth** (llama-bench, stessa build `d127cb754`, cache q4_0, FA): pp512 1127,6 contro 1159,7 (97%); pp512 a 32K 777,8 contro 796,2 (98%). Il divario del server a 32K (818 contro 898, 91%) non è nei prodotti: stimando il solo modello, NariQ ha ~4,9 s di lavoro in più nel server, Unsloth ~2,3. Indiziata la passata della bozza sul prompt (micro-batch della bozza limitato a 64 per i target Trellis, `70dc70ae1`).
  - **Riserva della bozza su `n_outputs_max` e micro-batch 512** (`dflash-reserve-ubatch512.patch`): VRAM 14,5 → 15,6 GB + 408 MB condivisi, prefill 32K 818 → 273 t/s, accettazione cambiata. L'allocatore cresce comunque in esecuzione. Scartata.
  - **Iniezione del prompt nella bozza a blocchi di `n_batch`** (`dflash-inject-batch.patch`, con il controllo non causale rilassato per i lotti di feature DFlash): generazione, accettazione e VRAM identiche, prefill 32K 818 → 805 (rumore). Il costo fisso per chiamata non è la causa. Scartata.
  - **Esito**: build invariata (`d02cd6d2e` in OpenClaw). Resta aperto il ~7% di prefill del server, probabilmente nel lavoro della bozza sul prompt a micro-batch 64: per toglierlo serve un grafo della bozza che calcoli la testa solo sulle righe di uscita e che non allarghi la memoria.

- **27/11/2026, Claude**, `d02cd6d2e` e commit di questa voce: prova reale con Jarvis, report di Jarvis, narrow-N, hash a 24 bit (A5c / G3 risolti).
  - **Prova reale OpenClaw** (`logs\local\llama.err.log`, build `c3c2bf7eb`, bozza 6): 70 richieste, generazione 120.793 token in 2.459 s = 49,1 t/s (per richiesta 35-78, mediana 51); accettazione della bozza 35,3% (suite 40-46%); prefill dei blocchi ≥ 1.000 token 563 t/s su contesti già lunghi. Nessun errore. Con Chrome aperto llama-server aveva 477 MB in memoria condivisa (normale 110-150): chiuso Chrome, 154 MB. Bozza ferma a 6 per il margine di VRAM.
  - **Composizione del ciclo**: pp7 (verifica di 7 token) 156 t/s = 44,9 ms su un ciclo di 51-52 ms (suite, chat): bozza e resto ~13%.
  - **Report di Jarvis** (`Jarvis-analisys.md`): utile la diagnosi (calcolo, non banda). Da non seguire: generazione base "24,56 per entrambi" (dato vecchio: 28,9 contro 39,4), proiettore su GPU (è su CPU), #27952 presente (non lo è), N1 come leva per i prodotti Trellis (narrow-N tocca solo i quant standard), N2 / N3 (VRAM al limite).
  - **Narrow-N** ([ggml-org/llama.cpp#31990](https://github.com/ggml-org/llama.cpp/pull/31990)): non portato. Con coopmat1 (AMD) le tile strette valgono per 9 ≤ N ≤ 32; fino a 8 colonne (`mul_mat_vec_max_cols`) si usa il prodotto matrice-vettore, e la bozza ne usa 6-7.
  - **Hash a 24 bit** (`d02cd6d2e`): `state * 0x83dcd12d` come `s * 0xd12d + (s * 0x83dc) << 16`, con metà e shift ricavati in ogni kernel da un bit sempre zero di una costante push (`trellis_hash_init(p.k >> 31)` in `trellis_decode.glsl`). Con le costanti visibili il compilatore ricomponeva `v_mul_lo_u32`; con la sola costante opaca spostava lo shift nel prodotto (ancora `v_mul_lo_u32`, 6 token 124,6 → 138-143 µs); con anche lo shift opaco l'ISA ha `v_mul_u32_u24` / `v_mad_u32_u24` e nessuna `v_mul_lo_u32` nel ciclo. Logits e KLD identici. TRL3 up / down: 1 token 92 → 88 / 110 → 103,5 µs; 4 token 122 → 117 / 140 → 136; 6 token 124,6 → 119,7 / 137,4 → 132,8. Modello: tg64 28,9 → 30,0, pp7 156 → 166, pp512 ~1106 → ~1124 t/s. Server (bozza 6, due run): 71,9 / 68,5 / 66,4 t/s, media 67,0 → 68,9; prefill a 32K 818 t/s.
  - **OpenClaw**: build `runtime\backends\llama.cpp-nariq-vulkan-win-x64-d02cd6d2e`.

- **26/11/2026 notte, Claude**, `c3c2bf7eb` e commit di questa voce: generazione, perché la base è lenta, tabella dei valori, lettura anticipata.
  - **Generazione base** (tg64, stessa build): NariQ 28,9 t/s (35,1 ms per token, di cui TRELLIS 32,7), Unsloth 39,4 (25,4 ms). Nessuno dei due è al limite di banda (~68 t/s). La stessa matrice 5120 → 17408 a 1 token costa ~95 µs a TRL2, TRL3 e TRL4: il prodotto è limitato dal calcolo della decodifica, non dalla memoria. Con la bozza Unsloth verifica 4 token in 25,5 ms (come 1), NariQ 6 token in ~47 ms (+37% su 1 token).
  - **Tabella dei valori** (`build/p1/archive/lut-1token.patch`): 64 KiB con la somma delle cifre di ogni stato a 16 bit al posto di hash, maschera e somma, nel prodotto a 1 token con attivazioni impacchettate. 92 → 778 µs: le letture di byte sparse dalla memoria globale si serializzano; in memoria condivisa non entra (limite 32 KiB del driver). Scartata.
  - **Verifica F4 a 5-16 token, lettura anticipata** (`c3c2bf7eb`): parole codice e attivazioni int8 del tile successivo lette mentre si decodifica il corrente. 6 token: TRL3 down 151,6 → 137,4 µs, TRL4 down 160 → 142,5, TRL3 up 127,3 → 124,6; pp6 128,7 → 139,7 t/s (con l'impacchettamento di `029811524`). Logits identici. Scartate: due tile di righe per gruppo (up 127 → 133, down 152 → 172) e la lettura anticipata nel prodotto a 1 token (92 → 94-104). Dettagli in `build/p1/archive/README.txt`.
  - **Server** (suite `build/g1`, `c3c2bf7eb`, ordine 6-8-7-7-8-6, medie di due run; chat-42 / chat-173 / long-42): n-max 6 69,9 / 66,5 / 64,6 t/s (media 67,0; prima 63,5); n-max 7 65,4 / 67,6 / 71,2 (68,1); n-max 8 65,1 / 67,3 / 71,0 (67,8). 7 e 8 hanno la stessa accettazione. Resta 6, più uniforme.
  - **OpenClaw**: build `runtime\backends\llama.cpp-nariq-vulkan-win-x64-c3c2bf7eb`, bozza 6.
  - **Esito**: generazione con bozza 60,4 → 67,0 t/s in questa sessione (+11%), ~63% di Unsloth. La base resta ~29 t/s: per scendere serve meno lavoro per peso (moltiplicazione dell'hash, estrazione degli stati), non memoria.

- **26/11/2026 notte, Claude**, `029811524` e commit di questa voce: generazione, profilo della verifica, bozza da 6 token.
  - **Profilo della verifica** (llama-bench pp6 con `GGML_VK_PERF_LOGGER`): NariQ 48,0 ms per 6 token, di cui TRELLIS 44,6 ms (93%); Unsloth 25,5 ms per un passaggio a 4 token (MUL_MAT_VEC 13,8, MUL_MAT_ADD 6,9). Il ciclo DFlash2 di NariQ (~55 ms) è per ~85% la verifica.
  - **ISA di `trellis_verify16_i32_cm`** (`GGML_VK_PIPELINE_ISA`): per tile da 256 pesi e lane 2 letture, ~10 istruzioni per gli 8 stati, 8 `v_mul_lo_u32`, 8 `and`, 8 `dot4`, ~14 istruzioni di impacchettamento, 2 scritture LDS, una WMMA int8. Le barriere di sottogruppo non generano istruzioni. Memoria condivisa ~6,5 KB per gruppo e ~40 VGPR: l'occupazione non limita. Il kernel è limitato dalle istruzioni di decodifica.
  - **Impacchettamento** (`029811524`): `trellis_quant_i32` salva anche la somma degli int8 di ogni blocco da 32; la verifica scrive somma − 128 (solo xor), parte dall'accumulatore 2 × somma e somma il primo stato di ogni coppia nell'operando di accumulo del `dot4` del secondo. Logits identici a u4/u6/u8/u16; TRL3 up a 6 token 130,4 → 127,3 µs, down 153,5 → 151,6.
  - **Lunghezza della bozza** (suite `build/g1`, build `22939f800`, ordine 5-6-8-8-6-5, medie di due run; chat-42 / chat-173 / long-42): n-max 5 61,5 / 60,9 / 58,7 t/s (media 60,4); n-max 6 66,9 / 62,5 / 61,2 (63,5, +5%); n-max 8 61,2 / 63,3 / 67,1 (63,9, più irregolare). Accettazione 46-49% → 40-46% → 35-45%.
  - **OpenClaw**: profilo `qwen38-nariq` a 6 token, build `runtime\backends\llama.cpp-nariq-vulkan-win-x64-029811524`, backup in `runtime\migrations\2026-11-26-nariq-n6\backup`.
  - **Esito**: generazione NariQ ~63 t/s contro ~107 di Unsloth (~60%). Il limite è strutturale: ogni peso richiede ~9 istruzioni di decodifica (hash a 32 bit a un quarto di velocità incluso) a ogni passaggio, mentre i quant IQ di Unsloth sono limitati dalla banda. Leve rimaste, tutte piccole: moltiplicazione a 24 bit se il compilatore la emettesse (G3: no), meno istruzioni per estrarre gli stati, testa della bozza non Trellis (~3-4 ms per ciclo, VRAM in più).

- **26/11/2026 sera, Claude**, `b77ecda3b`, `f049540aa`, `22939f800` e commit di questa voce: prefill, catena MLP fusa e rotazioni da 128 nei registri. Tutto predefinito; `GGML_VK_TRELLIS_CHAIN=0`, `GGML_VK_TRELLIS_PREP_IN=0`, `GGML_VK_TRELLIS_OUT128=0` spengono le singole parti.
  - **Catena MLP fusa** (`b77ecda3b`, `trellis_chain.comp`, come in [nari-labs/nari-llama.cpp#5](https://github.com/nari-labs/nari-llama.cpp/pull/5)): gate e up scrivono il prodotto grezzo; un kernel per blocco da 1024 e token fa le due rotazioni d'uscita con le scale, SwiGLU, la scala e la rotazione d'ingresso di down; down parte dal prodotto. Pianificata per grafo (`ggml_vk_trellis_plan_chains`): nel grafo up, gate, SwiGLU e down sono separati da nodi della cache (VIEW, SCALE), quindi non è una fusione di nodi consecutivi. pp512 947 → 996-1040, pp4096 929 → 969-973; tempo GPU Trellis per 512 token 458 → 431 ms. KLD 385 token invariata (massima 3,6e-5, top-1 100%); logits u32 KL massima 5,8e-4 (solo sull'ultimo token generato, righe del prompt uguali).
  - **Ingresso, rotazione da 128 e preparazione FP16 in un kernel** (`trellis_prep16_in.comp`, un gruppo per token, blocchi trasformati nei registri del sottogruppo) e **rotazione da 128 + uscita in un kernel** (`trellis_out128.comp`), `f049540aa`: stessi arrotondamenti, KLD e PPL identiche in tutte le cifre. Ingresso + rotazione 52 → 7 ms, rotazione d'uscita 20 → 9 ms per 512 token; pp512 1041 → 1106, pp4096 1012 → 1079 (A-B-B-A, deviazioni ±2).
  - **Correzione** (`22939f800`): la pianificazione contava gli usi di tutti i nodi a ogni grafo, −0,9% in generazione (tg64 28,92 → 28,67); ora esce subito senza SwiGLU con più di 16 righe: tg64 28,93 / 28,94, pp6 128,7 / 128,7, pp128 671 → 697.
  - **Porting int8 coopmat upstream** ([ggml-org/llama.cpp#27952](https://github.com/ggml-org/llama.cpp/pull/27952)): non fatto. Il prefill di Unsloth UD-Q2_K_XL è iq3_xxs 33%, iq2_xxs 16%, iq2_s 15%, iq2_xs 10%, iq3_s 9%, iq4_xs 7%, iq1_s 6%, q8_0 3%: la PR copre solo q8_0 fra questi, guadagno massimo ~1%.
  - **Server** (suite `build/g1`, stessa build `22939f800`; NariQ bozza 5 a 0,8, Unsloth bozza 3 a 1,0 come nei profili OpenClaw; un run per caso): prefill a 32K NariQ 826,7 t/s (38,9 s) contro Unsloth 897,7 (35,8 s), 92% (prima 82%); a 65K 720,5 (90,4 s) contro 765,5 (85,1 s), 94% (prima 86%). Generazione NariQ 60,2 / 60,5 / 58,7 / 50,3 t/s (chat-42, chat-173, 32K, 65K), ciclo 55,3 / 54,1 / 59,8 / 63,0 ms; Unsloth 106,6 / 109,3 / 106,0 / 90,1 t/s, ciclo 26,6 / 26,3 / 29,1 / 32,0 ms. VRAM dedicata NariQ 14,49 / 14,63 GB, Unsloth 13,54 / 13,68 GB.
  - **OpenClaw**: build `runtime\backends\llama.cpp-nariq-vulkan-win-x64-22939f800`, backup degli script in `runtime\migrations\2026-11-26-nariq-prefill\backup`.
  - **Esito**: primo traguardo del prefill (≥ 90% di Unsloth) raggiunto a 32K e 65K. Il divario ora è nella generazione: ciclo DFlash2 ~55 ms contro ~27 di Unsloth, generazione al 55%. Il prodotto INT8 del prefill con scala ogni 32 input resta da valutare: la riconversione int32 → float ogni 32 input costa circa quanto le WMMA int8 risparmiate, e la qualità a scala per riga della PR #5 (top-1 98,2%) è fuori tolleranza.

- **26/11/2026, Claude**, commit di questa voce (solo registro; le due prove sono archiviate in `build/p1/archive/p2-pair-fusebig.patch`): verifica a 65K, P1 e prime prove P2.
  - **65K** (suite `build/g1`, nuovo modo `long65`, 65.145 token, F4, bozza 0,8, ordine 3-5-5-3): generazione n-max 3 47,52 / 47,34 t/s, n-max 5 51,01 / 50,66 (+7%); prefill 661,7 / 658,6 t/s, 98,4 / 98,9 s; picco VRAM dedicata 14,612 / 14,641 GB, condivisa 107 / 115 MB. Bozza 5 confermata anche a 65K.
  - **P1, profilo pp512** (llama-bench, cache q4_0, FA, b/ub 512, stessa build `608ac1dfb`, `GGML_VK_PERF_LOGGER`): NariQ 926 t/s, Unsloth 1131 t/s. Somma delle operazioni GPU: NariQ 549 ms, di cui TRELLIS 465 ms (84,7%); Unsloth 469 ms, di cui MUL_MAT 389 ms (83,0%). Le altre operazioni coincidono (delta-net 15-16, concat 14, GLU 11-12, norme ~18 ms). Divario tutto nei prodotti: ~76 ms.
  - **P1, fasi Trellis** (`GGML_VK_TRELLIS_PROFILE` insieme al perf logger, due blocchi da 256 token): ingresso 18,8, rotazione in ingresso 52-56, prodotto 329-346, rotazione in uscita 34-37, uscita 23 ms. Il prodotto da solo è già sotto il MUL_MAT di Unsloth (389); il divario sono i ~130 ms di ingresso, rotazioni e uscita. Rotazioni tutte Hadamard a blocchi (`GGML_VK_TRELLIS_DEBUG`): 128 in ingresso alle proiezioni da 5120, 1024 per 17408, 2048 per 6144 e 10240, 4096 per 12288.
  - **P2, coppie FP16 nel prodotto 128 × 256** (`1024 + 4 × somma` nei bit di 0x6400 meno 1528 con una sottrazione FP16 a coppie, senza conversioni float): esatta (380/380, KLD 385 token identica bit per bit), ma pp512 943 / 943 contro 943 / 988 (una sola misura alta, rumore), pp4096 invariato. Scartata.
  - **Fusione ingresso + rotazione e rotazione in uscita + uscita anche nel prefill** (kernel `trellis_fused.comp` esteso oltre 15 token): 380/380; il tempo Trellis passa da 460 / 471 a 453 / 465 ms (−4 ms per blocco), llama-bench entro il rumore. Il kernel fuso usa una farfalla in LDS con una barriera per passo, lento quanto i passaggi separati; in uscita non si attiva per la maggior parte dei prodotti. Scartata.
  - **Rete** ([nari-labs/nari-llama.cpp#5](https://github.com/nari-labs/nari-llama.cpp/pull/5), CUDA, aperta il 25/11): prefill INT8 a scala per riga, pp8192 675 → 1424 su RTX 5060 Ti, ma stesso top-1 98,2% (fuori dalle nostre tolleranze); catena MLP fusa al prefill (prodotti gate/up, rotazioni d'uscita, SwiGLU e rotazione d'ingresso di down in un kernel), +9,5% anche sul percorso FP16, rotazioni da 10,3% a 3,0% del profilo. Upstream llama.cpp ha un prodotto int8 coopmat1 per RDNA3/4 (#27952), utile solo ai quant standard.
  - **Esito**: le leve per arrivare a Unsloth sono (1) catena MLP fusa, esatta, stima −35-50 ms su 465; (2) prodotto del prefill INT8 con una scala ogni 32 input e accumulo int32 per due tile (lo schema di F4), stima −80-120 ms se la qualità regge; (3) rotazioni a blocchi più veloci con shuffle di sottogruppo anche nei passaggi fusi, stima −15-25 ms.

- **26/11/2026, Claude**, commit di questa voce: F4 predefinita da 2 a 16 token (`GGML_VK_TRELLIS_VERIFY_I32=0` la spegne). Con le impostazioni predefinite: test-backend-ops TRELLIS 380/380, ctest 4/4, logits contro CPU u1 / u4 / u6 / u32 KL massima 2,0e-4 / 1,4e-4 / 2,7e-4 / 6,2e-5, top-1 100%, KLD 385 token massima 3,6e-5, stesso top-1 100%. Profilo OpenClaw `qwen38-nariq`: bozza DFlash2 da 3 a 5 token e `--spec-draft-temp 0.8`. VRAM dedicata nella suite G1 a 112K: 14,46 GB con bozza 3, 14,49 GB con bozza 5 (+27 MB).

- **26/11/2026, Claude**, commit di questa voce: F4 (accumulo int32, una scala ogni 32 input) anche per la verifica da 5 a 16 token (`trellis_verify16_i32_cm`, variante `TRELLIS_VERIFY_I32` di `trellis_verify16_cm.comp`). Ancora opt-in in questo commit (predefinita dalla voce sopra).
  - **Prove preliminari**: hash con moltiplicazione a 16 bit (`uint16_t`): corretta ma il compilatore la allarga a `v_mul_lo_u32`, 1 token 92 → 105 µs, scartata. LDS da 64 KB sul driver AMD Windows: non forzabile (`maxComputeSharedMemorySize` = 32768 nel driver 26.11.2, nessuna estensione, chiave di registro o opzione LLPC nota; 64 KB solo con RADV su Linux).
  - **Correttezza**: test-backend-ops TRELLIS 380/380 con e senza F4 (comprende t = 8); ctest 4/4; logits contro CPU, KL massima senza / con F4: u4 2,1e-4 / 1,4e-4, u6 e u8 4,4e-4 / 2,7e-4, u16 9,3e-5 / 1,6e-4, top-1 100%.
  - **Tempi** (`test-backend-ops perf`, A-B-B-A, TRL3 up `5120 → 17408`): 5 token 151,5 → 127,9 µs, 6 token 153,3 → 130,4, 8 token 157,0 → 133,0, 16 token 176,0 → 148,5; down `17408 → 5120` a 6 token 177,1 → 154,5. TRL4 simile (6 token 156 → 135). Nuovi casi perf a 5, 6, 8 token (e down a 6).
  - **Server** (suite `build/g1`, F4 attiva, bozza 0,8, ordine 3-4-5-5-4-3, somme di due run; chat-42 / chat-173 / long-42): n-max 3: 63,5 / 55,9 / 53,6 t/s, ciclo 49,5 / 49,4 / 52,3 ms (invariato). n-max 4: 51,1 / 61,7 / 55,9, ciclo 53,6 / 53,2 / 56,8 ms (prima 60-65). n-max 5: 64,9 / 65,4 / 59,7 t/s, ciclo 54,3 / 53,3 / 58,0 ms (prima 62-66), 3,46-3,52 token per ciclo. Prefill long-42 750-762 t/s invariato.
  - **Esito**: con F4 la bozza da 5 token batte quella da 3 in tutti i casi, media 57,7 → 63,3 t/s (+10%). Da decidere: F4 predefinita e n-max 5 nel profilo OpenClaw; da provare n-max 6-8.

- **26/11/2026, Claude**, commit di questa voce (solo il nuovo caso perf `17408 → 5120` a 1 token): G2 e G3.
  - **G2, tempi** (`test-backend-ops perf`, F4 attiva, TRL3 / TRL4): up `5120 → 17408` a 1 token 91,9 / 93,5 µs, a 4 token 122,7 / 124,5; down `17408 → 5120` a 1 token 110,2 / 110,9, a 4 token 139,6 / 147,2.
  - **G2, ISA** (`trellis_matvec_q8` a 1 token e `trellis_verify_i32_cm` a 4 token, TRL3): per 8 pesi per lane 2 letture `b32` dei codici, finestra di bit con `lshrrev` + `lshlrev` + `lshl_or` + `alignbit`, 8 stati con 6 `bfe` + 4 `mov_b16`, 8 `v_mul_lo_u32` (hash, un quarto di velocità: ~32 cicli), 8 `and`, 8 `dot4`. Indici delle parole e shift delle finestre sono già fuori dal ciclo delle tile (registri costanti per lane): la prima parte di G3 è già fatta dal compilatore.
  - **G2, esclusioni** (shader diagnostici `TRELLIS_DIAG`, risultati volutamente errati, poi rimossi), TRL3 up / down: senza letture né estrazione dei codici 1 token 107,0 / 117,6 µs, 4 token 136,8 / 149,5 (più lento: il compilatore riordina il resto); senza la moltiplicazione dell'hash 1 token 79,9 / 90,2 (−13 / −18%), 4 token 110,3 / 129,2 (−10 / −8%). Il costo della decodifica è la moltiplicazione a 32 bit, non le letture o l'estrazione.
  - **G3, moltiplicazione spezzata in due a 24 bit** (`state * 0xd12d + ((state * 0x83dc) << 16)`), con uno zero opaco preso da una costante push (`p.k >> 31`) per impedire al compilatore di rifattorizzare: corretta (416/416), ma 1 token 92 → 105 µs, 4 token 122 → 137. L'ISA mostra `v_mad_u32_u24` per la prima metà e ancora `v_mul_lo_u32` per la seconda, anche con maschere esplicite (`& 0xffffff`, `& 0xffff`): dopo lo shift servono solo i 16 bit bassi, il compilatore toglie le maschere e perde la prova dei 24 bit. In GLSL manca un equivalente di `__umul24`. Scartata. Le sequenze da 32 stati con shift costanti (seconda parte di G3) non sono state provate: G2 mostra che estrazione e letture non sono il collo di bottiglia.
  - **Esito**: nessuna modifica del kernel. Il margine più concreto resta F4 a 5-8 token (G1: bozza da 5 token, +20% di token per ciclo).

- **26/11/2026, Claude**, build 130 (`6b782586a`): G1, riferimento, e bozza da 4-5 token con F4.
  - **Disturbo**: con Steam aperto (GPU in uso, 2 GiB di VRAM occupati) pp4 63 ± 5 e tg64 a 32K 24,2: misure scartate. Prima di ogni prova controllare anche i programmi in background che usano la GPU.
  - **llama-bench** (cache q4_0, FA, b/ub 512), A-B-B-A senza / con F4 (`GGML_VK_TRELLIS_VERIFY_I32`): pp4 79,05 / 79,16 → 92,87 (+17%), tg64 28,96 / 28,98 → 28,97. Con F4: pp512 947, a 32K pp512 695, tg64 27,70, pp4 71,14.
  - **Server** (suite `build/g1`, copia di `build/f4-pack-server` con `--n-max`; F4 attiva, bozza 0,8, casi tune chat-42 / chat-173 / long-42, 384 token, ordine 3-4-5-5-4-3, somme di due run): n-max 3: 63,5 / 55,9 / 53,8 t/s, ciclo 49,9 / 49,9 / 52,5 ms, 3,17 / 2,79 / 2,82 token per ciclo. n-max 4: 48,4 / 50,2 / 50,6, ciclo 60,6 / 60,6 / 65,2 ms. n-max 5: 54,9 / 58,2 / 52,2, ciclo 62,3 / 61,7 / 66,2 ms, 3,42 / 3,59 / 3,46 token per ciclo. Prefill long-42 750-762 t/s.
  - **Esito**: resta n-max 3. Con 4-5 token di bozza la verifica ha 5-6 token e F4 non si applica (`batch <= VK_TRELLIS_Q8_TOKENS`): il ciclo sale di ~11 ms. Con n-max 5 i token per ciclo crescono di ~20% (3,42-3,59 contro 2,79-3,17); se la verifica a 6 token costasse quanto quella a 4 con F4 (~50 ms), la generazione salirebbe a ~65-70 t/s. Prossimo passo: estendere F4 a 5-8 token (scala per 32 input e accumulo int32 anche in `trellis_verify16_cm`), poi ripetere questa suite.

- **26/11/2026, Codex**, F4 pack chat sintesi server. F4 attiva in entrambe le build, bozza 0,8. Confronto server A-B-B-A su stessi prompt e seed. breve 55.13 -> 59.18 t/s (+7.35%), ciclo stimato 53.86 -> 50.17 ms, token DFlash2 1014/1552 in entrambi. 32K 50.24 -> 53.49 t/s (+6.46%), ciclo stimato 56.20 -> 52.79 ms, token DFlash2 494/816 in entrambi. Guadagno confermato sul server a 32K. Log build/f4-pack-server.

- **26/11/2026, Codex**, F4 pack chat tune-A-t0.8-2 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.577 GiB. Log build/f4-pack-server/tune-A-t0.8-2.

- **26/11/2026, Codex**, F4 pack chat tune-A-t0.8-2-long-42. Seed 42, prefill 32145 token, generazione 384 a 50.66 t/s, accettazione 60.5%, ciclo stimato 55.59 ms. VRAM prima della richiesta 1.218 GiB. Log build/f4-pack-server/tune-A-t0.8-2-long-42.json.

- **26/11/2026, Codex**, F4 pack chat tune-A-t0.8-2-chat-173. Seed 173, prefill 4 token, generazione 384 a 52.70 t/s, accettazione 59.3%, ciclo stimato 52.79 ms. VRAM prima della richiesta 1.264 GiB. Log build/f4-pack-server/tune-A-t0.8-2-chat-173.json.

- **26/11/2026, Codex**, F4 pack chat tune-A-t0.8-2-chat-42. Seed 42, prefill 153 token, generazione 384 a 59.98 t/s, accettazione 72.2%, ciclo stimato 52.77 ms. VRAM prima della richiesta 1.381 GiB. Log build/f4-pack-server/tune-A-t0.8-2-chat-42.json.

- **26/11/2026, Codex**, F4 pack chat tune-B-t0.8-2 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.582 GiB. Log build/f4-pack-server/tune-B-t0.8-2.

- **26/11/2026, Codex**, F4 pack chat tune-B-t0.8-2-long-42. Seed 42, prefill 32145 token, generazione 384 a 53.29 t/s, accettazione 60.5%, ciclo stimato 52.84 ms. VRAM prima della richiesta 1.261 GiB. Log build/f4-pack-server/tune-B-t0.8-2-long-42.json.

- **26/11/2026, Codex**, F4 pack chat tune-B-t0.8-2-chat-173. Seed 173, prefill 4 token, generazione 384 a 55.66 t/s, accettazione 59.3%, ciclo stimato 49.98 ms. VRAM prima della richiesta 1.264 GiB. Log build/f4-pack-server/tune-B-t0.8-2-chat-173.json.

- **26/11/2026, Codex**, F4 pack chat tune-B-t0.8-2-chat-42. Seed 42, prefill 153 token, generazione 384 a 63.22 t/s, accettazione 72.2%, ciclo stimato 50.07 ms. VRAM prima della richiesta 1.381 GiB. Log build/f4-pack-server/tune-B-t0.8-2-chat-42.json.

- **26/11/2026, Codex**, F4 pack chat tune-B-t0.8-1 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.579 GiB. Log build/f4-pack-server/tune-B-t0.8-1.

- **26/11/2026, Codex**, F4 pack chat tune-B-t0.8-1-long-42. Seed 42, prefill 32145 token, generazione 384 a 53.41 t/s, accettazione 60.5%, ciclo stimato 52.73 ms. VRAM prima della richiesta 1.267 GiB. Log build/f4-pack-server/tune-B-t0.8-1-long-42.json.

- **26/11/2026, Codex**, F4 pack chat tune-B-t0.8-1-chat-173. Seed 173, prefill 4 token, generazione 384 a 55.67 t/s, accettazione 59.3%, ciclo stimato 49.98 ms. VRAM prima della richiesta 1.266 GiB. Log build/f4-pack-server/tune-B-t0.8-1-chat-173.json.

- **26/11/2026, Codex**, F4 pack chat tune-B-t0.8-1-chat-42. Seed 42, prefill 153 token, generazione 384 a 62.43 t/s, accettazione 72.2%, ciclo stimato 50.70 ms. VRAM prima della richiesta 1.382 GiB. Log build/f4-pack-server/tune-B-t0.8-1-chat-42.json.

- **26/11/2026, Codex**, F4 pack chat tune-A-t0.8-1 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.575 GiB. Log build/f4-pack-server/tune-A-t0.8-1.

- **26/11/2026, Codex**, F4 pack chat tune-A-t0.8-1-long-42. Seed 42, prefill 32145 token, generazione 384 a 49.57 t/s, accettazione 60.5%, ciclo stimato 56.81 ms. VRAM prima della richiesta 1.260 GiB. Log build/f4-pack-server/tune-A-t0.8-1-long-42.json.

- **26/11/2026, Codex**, F4 pack chat tune-A-t0.8-1-chat-173. Seed 173, prefill 4 token, generazione 384 a 49.15 t/s, accettazione 59.3%, ciclo stimato 56.61 ms. VRAM prima della richiesta 1.264 GiB. Log build/f4-pack-server/tune-A-t0.8-1-chat-173.json.

- **26/11/2026, Codex**, F4 pack chat tune-A-t0.8-1-chat-42. Seed 42, prefill 153 token, generazione 384 a 59.68 t/s, accettazione 72.2%, ciclo stimato 53.04 ms. VRAM prima della richiesta 1.380 GiB. Log build/f4-pack-server/tune-A-t0.8-1-chat-42.json.

- **26/11/2026, Codex**, F4 conferma sintesi. Bozza 0,8 confermata per chat breve nei prompt provati. 52,66 -> 54,21 t/s, accettazione 58,6% -> 61,7%, 6/9 coppie a favore di 0,8. Diagnostico TRL3 a 4 token, con output sintetici. Up 130,99 us con il kernel esatto, 72,52 us con hash e packing sostituiti e 60,11 us saltando anche le letture dei codici. Down 148,55 -> 93,59 -> 70,69 us nelle stesse prove. Il proxy senza WMMA è più lento e cambia il modo in cui il driver carica e usa i frammenti. Questi confronti localizzano il costo della decodifica e non sono una scomposizione additiva. Pack esatto a coppie in F4. trl3 5120 → 17408 132,06 -> 123,88 us. trl3 17408 → 5120 149,44 -> 141,18 us. trl4 5120 → 17408 133,23 -> 125,22 us. trl4 17408 → 5120 156,08 -> 147,14 us. Vulkan 416/416 e CTest 4/4. Le 31 righe dei logits coincidono bit per bit con F4 prima della modifica. Log build/f4-confirm.

- **26/11/2026, Codex**, F4 conferma perf-A-exact-2. Exit 0. VRAM libera 14.58 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 92.87 us/run - 32816 kB/run - [1;34m 336.99 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 131.90 us/run - 33080 kB/run - [1;34m 239.18 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 149.30 us/run - 33082 kB/run - [1;34m 211.32 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10752 runs - 93.32 us/run - 43696 kB/run - [1;34m 446.56 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 133.30 us/run - 43960 kB/run - [1;34m 314.51 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6876 runs - 155.28 us/run - 43962 kB/run - [1;34m 270.00 GB/s[0m. Log build/f4-confirm/perf-A-exact-2.log.

- **26/11/2026, Codex**, F4 conferma perf-B-exact-2. Exit 0. VRAM libera 14.58 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 92.90 us/run - 32816 kB/run - [1;34m 336.87 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 123.62 us/run - 33080 kB/run - [1;34m 255.20 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 141.78 us/run - 33082 kB/run - [1;34m 222.53 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10752 runs - 93.45 us/run - 43696 kB/run - [1;34m 445.95 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8404 runs - 125.43 us/run - 43960 kB/run - [1;34m 334.25 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6876 runs - 147.94 us/run - 43962 kB/run - [1;34m 283.39 GB/s[0m. Log build/f4-confirm/perf-B-exact-2.log.

- **26/11/2026, Codex**, F4 conferma perf-B-exact-1. Exit 0. VRAM libera 14.576 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 92.50 us/run - 32816 kB/run - [1;34m 338.34 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 124.14 us/run - 33080 kB/run - [1;34m 254.14 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 8120 runs - 140.57 us/run - 33082 kB/run - [1;34m 224.44 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10752 runs - 93.38 us/run - 43696 kB/run - [1;34m 446.29 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8404 runs - 125.02 us/run - 43960 kB/run - [1;34m 335.33 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6876 runs - 146.35 us/run - 43962 kB/run - [1;34m 286.47 GB/s[0m. Log build/f4-confirm/perf-B-exact-1.log.

- **26/11/2026, Codex**, F4 conferma perf-A-exact-1. Exit 0. VRAM libera 14.576 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 92.41 us/run - 32816 kB/run - [1;34m 338.68 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 132.22 us/run - 33080 kB/run - [1;34m 238.60 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 149.57 us/run - 33082 kB/run - [1;34m 210.94 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10752 runs - 94.43 us/run - 43696 kB/run - [1;34m 441.31 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 133.16 us/run - 43960 kB/run - [1;34m 314.83 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6876 runs - 156.88 us/run - 43962 kB/run - [1;34m 267.24 GB/s[0m. Log build/f4-confirm/perf-A-exact-1.log.

- **26/11/2026, Codex**, F4 conferma hash logits pack esatto. Logits F4 identici bit per bit sul campione di 31 righe.

- **26/11/2026, Codex**, F4 conferma logits-exact-pack. Top-1 31/31, KL max 0.000139429648072006, stessi token greedy True. Exit 0. Processi concorrenti assenti e 14.573 GiB stimati liberi. Log build/f4-confirm/logits-exact-pack.

- **26/11/2026, Codex**, F4 conferma ctest-B-exact. Exit 0. VRAM libera 14.577 GiB. 100% tests passed, 0 tests failed out of 4. Log build/f4-confirm/ctest-B-exact.log.

- **26/11/2026, Codex**, F4 conferma ops-B-exact. Exit 0. VRAM libera 14.577 GiB. 416/416 tests passed. Log build/f4-confirm/ops-B-exact.log.

- **26/11/2026, Codex**, F4 conferma build pack esatto. Impacchettamento a coppie nel solo percorso F4. Exit 0. Log build/f4-confirm/build-exact.log.

- **26/11/2026, Codex**, F4 conferma probe-0-r2. Exit 0. VRAM libera 14.582 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 131.63 us/run - 33080 kB/run - [1;34m 239.68 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 148.51 us/run - 33082 kB/run - [1;34m 212.44 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 133.22 us/run - 43960 kB/run - [1;34m 314.70 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6876 runs - 155.60 us/run - 43962 kB/run - [1;34m 269.45 GB/s[0m. Log build/f4-confirm/probe-0-r2.log.

- **26/11/2026, Codex**, F4 conferma probe-1-r2. Exit 0. VRAM libera 14.582 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 14210 runs - 73.96 us/run - 33080 kB/run - [1;34m 426.57 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 11165 runs - 95.56 us/run - 33082 kB/run - [1;34m 330.17 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 12988 runs - 77.83 us/run - 43960 kB/run - [1;34m 538.64 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 11460 runs - 91.61 us/run - 43962 kB/run - [1;34m 457.64 GB/s[0m. Log build/f4-confirm/probe-1-r2.log.

- **26/11/2026, Codex**, F4 conferma probe-2-r2. Exit 0. VRAM libera 14.576 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 6090 runs - 168.77 us/run - 33080 kB/run - [1;34m 186.93 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6090 runs - 190.91 us/run - 33082 kB/run - [1;34m 165.26 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 6112 runs - 169.23 us/run - 43960 kB/run - [1;34m 247.74 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 5348 runs - 198.29 us/run - 43962 kB/run - [1;34m 211.44 GB/s[0m. Log build/f4-confirm/probe-2-r2.log.

- **26/11/2026, Codex**, F4 conferma probe-3-r2. Exit 0. VRAM libera 14.576 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 17255 runs - 59.78 us/run - 33080 kB/run - [1;34m 527.72 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 14210 runs - 70.88 us/run - 33082 kB/run - [1;34m 445.12 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 17572 runs - 58.35 us/run - 43960 kB/run - [1;34m 718.51 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 15280 runs - 68.33 us/run - 43962 kB/run - [1;34m 613.58 GB/s[0m. Log build/f4-confirm/probe-3-r2.log.

- **26/11/2026, Codex**, F4 conferma probe-4-r2. Exit 0. VRAM libera 14.58 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 128.37 us/run - 33080 kB/run - [1;34m 245.76 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 146.43 us/run - 33082 kB/run - [1;34m 215.46 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8404 runs - 124.98 us/run - 43960 kB/run - [1;34m 335.44 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6876 runs - 147.00 us/run - 43962 kB/run - [1;34m 285.21 GB/s[0m. Log build/f4-confirm/probe-4-r2.log.

- **26/11/2026, Codex**, F4 conferma probe-4-r1. Exit 0. VRAM libera 14.582 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 127.42 us/run - 33080 kB/run - [1;34m 247.59 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 147.29 us/run - 33082 kB/run - [1;34m 214.19 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 131.20 us/run - 43960 kB/run - [1;34m 319.54 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6876 runs - 156.09 us/run - 43962 kB/run - [1;34m 268.60 GB/s[0m. Log build/f4-confirm/probe-4-r1.log.

- **26/11/2026, Codex**, F4 conferma probe-3-r1. Exit 0. VRAM libera 14.576 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 17255 runs - 60.43 us/run - 33080 kB/run - [1;34m 522.08 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 14210 runs - 70.49 us/run - 33082 kB/run - [1;34m 447.55 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 17572 runs - 57.61 us/run - 43960 kB/run - [1;34m 727.71 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 14516 runs - 69.65 us/run - 43962 kB/run - [1;34m 601.92 GB/s[0m. Log build/f4-confirm/probe-3-r1.log.

- **26/11/2026, Codex**, F4 conferma probe-2-r1. Exit 0. VRAM libera 14.58 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7105 runs - 161.36 us/run - 33080 kB/run - [1;34m 195.52 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6090 runs - 182.55 us/run - 33082 kB/run - [1;34m 172.83 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 6876 runs - 161.08 us/run - 43960 kB/run - [1;34m 260.26 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 5348 runs - 188.33 us/run - 43962 kB/run - [1;34m 222.61 GB/s[0m. Log build/f4-confirm/probe-2-r1.log.

- **26/11/2026, Codex**, F4 conferma probe-1-r1. Exit 0. VRAM libera 14.58 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 14210 runs - 71.08 us/run - 33080 kB/run - [1;34m 443.85 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 11165 runs - 91.61 us/run - 33082 kB/run - [1;34m 344.40 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 12988 runs - 77.78 us/run - 43960 kB/run - [1;34m 539.01 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 11460 runs - 92.19 us/run - 43962 kB/run - [1;34m 454.77 GB/s[0m. Log build/f4-confirm/probe-1-r1.log.

- **26/11/2026, Codex**, F4 conferma probe-0-r1. Exit 0. VRAM libera 14.576 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 130.34 us/run - 33080 kB/run - [1;34m 242.04 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 148.59 us/run - 33082 kB/run - [1;34m 212.33 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 133.47 us/run - 43960 kB/run - [1;34m 314.10 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6876 runs - 155.89 us/run - 43962 kB/run - [1;34m 268.94 GB/s[0m. Log build/f4-confirm/probe-0-r1.log.

- **26/11/2026, Codex**, F4 conferma probe-4-ops. Exit 0. VRAM libera 14.574 GiB. 416/416 tests passed. Log build/f4-confirm/probe-4-ops.log.

- **26/11/2026, Codex**, F4 conferma probe-0-ops. Exit 0. VRAM libera 14.572 GiB. 416/416 tests passed. Log build/f4-confirm/probe-0-ops.log.

- **26/11/2026, Codex**, F4 conferma build diagnostica. Exit 0. Sorgenti temporanei e DLL ripristinati. Binario diagnostico isolato, solo test-backend-ops. Log build/f4-confirm/build-probe.log.

- **26/11/2026, Codex**, F4 conferma SPIR-V diagnostico. Compilazione e validazione di cinque modi diagnostici. Exit 0.

- **26/11/2026, Codex**, F4 conferma sintesi temperatura. F4, bozza 1,0 -> 0,8, target 1,0. Cinque prompt e tre seed nuovi. chat_totale 52.66 -> 54.21 t/s e 58.6 -> 61.7% accettazione, 6/9 coppie vinte. code 56.91 -> 58.01 t/s e 66.9 -> 68.7% accettazione, 2/3 coppie vinte. long 52.89 -> 52.68 t/s e 67.1 -> 64.4% accettazione, 1/3 coppie vinte.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.586 GiB. Log build/f4-confirm/tune-B-t0.8.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-long-911. Seed 911, prefill 4 token, generazione 192 a 51.35 t/s, accettazione 63.1%, ciclo stimato 56.36 ms. VRAM prima della richiesta 1.110 GiB. Log build/f4-confirm/tune-B-t0.8-long-911.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-long-173. Seed 173, prefill 4 token, generazione 192 a 49.26 t/s, accettazione 57.6%, ciclo stimato 55.39 ms. VRAM prima della richiesta 1.110 GiB. Log build/f4-confirm/tune-B-t0.8-long-173.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-long-57. Seed 57, prefill 32174 token, generazione 192 a 57.23 t/s, accettazione 74.0%, ciclo stimato 56.57 ms. VRAM prima della richiesta 1.272 GiB. Log build/f4-confirm/tune-B-t0.8-long-57.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-code-911. Seed 911, prefill 4 token, generazione 192 a 56.85 t/s, accettazione 67.2%, ciclo stimato 52.50 ms. VRAM prima della richiesta 1.265 GiB. Log build/f4-confirm/tune-B-t0.8-code-911.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-code-173. Seed 173, prefill 4 token, generazione 192 a 60.59 t/s, accettazione 73.3%, ciclo stimato 52.54 ms. VRAM prima della richiesta 1.266 GiB. Log build/f4-confirm/tune-B-t0.8-code-173.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-code-57. Seed 57, prefill 107 token, generazione 192 a 55.90 t/s, accettazione 65.8%, ciclo stimato 53.11 ms. VRAM prima della richiesta 1.266 GiB. Log build/f4-confirm/tune-B-t0.8-code-57.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-editor-911. Seed 911, prefill 4 token, generazione 192 a 54.18 t/s, accettazione 62.7%, ciclo stimato 52.61 ms. VRAM prima della richiesta 1.234 GiB. Log build/f4-confirm/tune-B-t0.8-editor-911.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-editor-173. Seed 173, prefill 4 token, generazione 192 a 53.38 t/s, accettazione 60.3%, ciclo stimato 52.62 ms. VRAM prima della richiesta 1.273 GiB. Log build/f4-confirm/tune-B-t0.8-editor-173.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-editor-57. Seed 57, prefill 107 token, generazione 192 a 51.94 t/s, accettazione 58.9%, ciclo stimato 52.78 ms. VRAM prima della richiesta 1.269 GiB. Log build/f4-confirm/tune-B-t0.8-editor-57.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-decision-911. Seed 911, prefill 4 token, generazione 192 a 59.63 t/s, accettazione 71.0%, ciclo stimato 52.51 ms. VRAM prima della richiesta 1.269 GiB. Log build/f4-confirm/tune-B-t0.8-decision-911.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-decision-173. Seed 173, prefill 4 token, generazione 192 a 51.30 t/s, accettazione 56.3%, ciclo stimato 52.44 ms. VRAM prima della richiesta 1.274 GiB. Log build/f4-confirm/tune-B-t0.8-decision-173.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-decision-57. Seed 57, prefill 95 token, generazione 192 a 55.88 t/s, accettazione 65.5%, ciclo stimato 52.85 ms. VRAM prima della richiesta 1.278 GiB. Log build/f4-confirm/tune-B-t0.8-decision-57.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-support-911. Seed 911, prefill 4 token, generazione 192 a 55.07 t/s, accettazione 63.1%, ciclo stimato 52.55 ms. VRAM prima della richiesta 1.278 GiB. Log build/f4-confirm/tune-B-t0.8-support-911.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-support-173. Seed 173, prefill 4 token, generazione 192 a 54.33 t/s, accettazione 62.2%, ciclo stimato 52.47 ms. VRAM prima della richiesta 1.284 GiB. Log build/f4-confirm/tune-B-t0.8-support-173.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t0.8-support-57. Seed 57, prefill 95 token, generazione 192 a 50.68 t/s, accettazione 56.8%, ciclo stimato 53.09 ms. VRAM prima della richiesta 1.387 GiB. Log build/f4-confirm/tune-B-t0.8-support-57.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.58 GiB. Log build/f4-confirm/tune-B-t1.0.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-long-911. Seed 911, prefill 4 token, generazione 192 a 56.87 t/s, accettazione 75.1%, ciclo stimato 56.92 ms. VRAM prima della richiesta 1.107 GiB. Log build/f4-confirm/tune-B-t1.0-long-911.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-long-173. Seed 173, prefill 4 token, generazione 192 a 50.17 t/s, accettazione 61.7%, ciclo stimato 56.82 ms. VRAM prima della richiesta 1.111 GiB. Log build/f4-confirm/tune-B-t1.0-long-173.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-long-57. Seed 57, prefill 32174 token, generazione 192 a 51.27 t/s, accettazione 65.3%, ciclo stimato 57.90 ms. VRAM prima della richiesta 1.267 GiB. Log build/f4-confirm/tune-B-t1.0-long-57.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-code-911. Seed 911, prefill 4 token, generazione 192 a 58.80 t/s, accettazione 70.4%, ciclo stimato 52.39 ms. VRAM prima della richiesta 1.267 GiB. Log build/f4-confirm/tune-B-t1.0-code-911.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-code-173. Seed 173, prefill 4 token, generazione 192 a 57.76 t/s, accettazione 69.1%, ciclo stimato 52.77 ms. VRAM prima della richiesta 1.267 GiB. Log build/f4-confirm/tune-B-t1.0-code-173.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-code-57. Seed 57, prefill 107 token, generazione 192 a 53.57 t/s, accettazione 61.6%, ciclo stimato 52.69 ms. VRAM prima della richiesta 1.271 GiB. Log build/f4-confirm/tune-B-t1.0-code-57.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-editor-911. Seed 911, prefill 4 token, generazione 192 a 52.15 t/s, accettazione 57.6%, ciclo stimato 52.32 ms. VRAM prima della richiesta 1.267 GiB. Log build/f4-confirm/tune-B-t1.0-editor-911.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-editor-173. Seed 173, prefill 4 token, generazione 192 a 52.11 t/s, accettazione 57.6%, ciclo stimato 52.36 ms. VRAM prima della richiesta 1.267 GiB. Log build/f4-confirm/tune-B-t1.0-editor-173.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-editor-57. Seed 57, prefill 107 token, generazione 192 a 49.93 t/s, accettazione 54.4%, ciclo stimato 52.88 ms. VRAM prima della richiesta 1.269 GiB. Log build/f4-confirm/tune-B-t1.0-editor-57.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-decision-911. Seed 911, prefill 4 token, generazione 192 a 50.02 t/s, accettazione 54.3%, ciclo stimato 52.31 ms. VRAM prima della richiesta 1.278 GiB. Log build/f4-confirm/tune-B-t1.0-decision-911.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-decision-173. Seed 173, prefill 4 token, generazione 192 a 51.42 t/s, accettazione 56.8%, ciclo stimato 52.32 ms. VRAM prima della richiesta 1.269 GiB. Log build/f4-confirm/tune-B-t1.0-decision-173.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-decision-57. Seed 57, prefill 95 token, generazione 192 a 58.61 t/s, accettazione 70.3%, ciclo stimato 52.84 ms. VRAM prima della richiesta 1.275 GiB. Log build/f4-confirm/tune-B-t1.0-decision-57.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-support-911. Seed 911, prefill 4 token, generazione 192 a 51.41 t/s, accettazione 56.3%, ciclo stimato 52.32 ms. VRAM prima della richiesta 1.273 GiB. Log build/f4-confirm/tune-B-t1.0-support-911.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-support-173. Seed 173, prefill 4 token, generazione 192 a 52.72 t/s, accettazione 59.5%, ciclo stimato 53.02 ms. VRAM prima della richiesta 1.253 GiB. Log build/f4-confirm/tune-B-t1.0-support-173.json.

- **26/11/2026, Codex**, F4 conferma tune-B-t1.0-support-57. Seed 57, prefill 95 token, generazione 192 a 54.10 t/s, accettazione 62.8%, ciclo stimato 53.23 ms. VRAM prima della richiesta 1.384 GiB. Log build/f4-confirm/tune-B-t1.0-support-57.json.

- **26/11/2026, Codex**, F4 accettazione sintesi accettazione e costo. Sui prefissi fissi, overlap A/B 0,437/0,441, 0,545/0,544, 0,306/0,300 per posizione (10/8/8 coppie). Con F4, bozza 0,8 rispetto a 1,0 porta la chat da 55,17 a 60,03 t/s, codice da 55,28 a 54,43 e 32K da 52,92 a 53,25. Le impostazioni di avvio restano quelle del riferimento. Nel profilo GPU seriale, mediana dei grafi di verifica 55,66 -> 51,58 ms. Le operazioni Trellis a quattro token pesano 47,72 ms con F4, con 18,06 ms per up e 10,39 ms per down. Le bande della testa della bozza pesano 2,00 ms. I tempi del profilo includono l'overhead delle misure. Il prototipo che riusa la scala ed elimina una barriera è scartato dopo il microbenchmark. Log build/f4-accept.

- **26/11/2026, Codex**, F4 accettazione riuso scala e barriera scartati. Risultato A-B-B-A, medie in microsecondi. trl3 5120->17408 136,95 -> 143,77. trl3 17408->5120 156,45 -> 164,50. trl4 5120->17408 138,39 -> 141,47. trl4 17408->5120 159,67 -> 168,33. Vulkan 416/416. Variante archiviata in build/f4-accept/scale-reuse.patch. Ripristinati lo shader e la DLL F4 precedente, copia verificata con SHA-256.

- **26/11/2026, Codex**, F4 accettazione perf-A-reuse-2. Exit 0. VRAM libera 14.569 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 96.32 us/run - 32816 kB/run - [1;34m 324.94 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 137.30 us/run - 33080 kB/run - [1;34m 229.78 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 156.39 us/run - 33082 kB/run - [1;34m 201.74 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10752 runs - 96.64 us/run - 43696 kB/run - [1;34m 431.21 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 137.12 us/run - 43960 kB/run - [1;34m 305.75 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6876 runs - 155.48 us/run - 43962 kB/run - [1;34m 269.65 GB/s[0m. Log build/f4-accept/perf-A-reuse-2.log.

- **26/11/2026, Codex**, F4 accettazione perf-B-reuse-2. Exit 0. VRAM libera 14.569 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 92.07 us/run - 32816 kB/run - [1;34m 339.92 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7105 runs - 143.01 us/run - 33080 kB/run - [1;34m 220.59 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6090 runs - 169.00 us/run - 33082 kB/run - [1;34m 186.68 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10752 runs - 96.53 us/run - 43696 kB/run - [1;34m 431.72 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 144.28 us/run - 43960 kB/run - [1;34m 290.57 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6112 runs - 171.89 us/run - 43962 kB/run - [1;34m 243.91 GB/s[0m. Log build/f4-accept/perf-B-reuse-2.log.

- **26/11/2026, Codex**, F4 accettazione perf-B-reuse-1. Exit 0. VRAM libera 14.57 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10230 runs - 98.05 us/run - 32816 kB/run - [1;34m 319.18 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7105 runs - 144.54 us/run - 33080 kB/run - [1;34m 218.27 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 159.99 us/run - 33082 kB/run - [1;34m 197.19 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10752 runs - 93.74 us/run - 43696 kB/run - [1;34m 444.53 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 138.65 us/run - 43960 kB/run - [1;34m 302.38 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6112 runs - 164.77 us/run - 43962 kB/run - [1;34m 254.45 GB/s[0m. Log build/f4-accept/perf-B-reuse-1.log.

- **26/11/2026, Codex**, F4 accettazione perf-A-reuse-1. Exit 0. VRAM libera 14.575 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 96.23 us/run - 32816 kB/run - [1;34m 325.23 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 136.60 us/run - 33080 kB/run - [1;34m 230.95 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 156.51 us/run - 33082 kB/run - [1;34m 201.59 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10752 runs - 96.80 us/run - 43696 kB/run - [1;34m 430.50 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 139.67 us/run - 43960 kB/run - [1;34m 300.18 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6112 runs - 163.86 us/run - 43962 kB/run - [1;34m 255.86 GB/s[0m. Log build/f4-accept/perf-A-reuse-1.log.

- **26/11/2026, Codex**, F4 accettazione ops-B-reuse. Exit 0. VRAM libera 14.561 GiB. 416/416 tests passed. Log build/f4-accept/ops-B-reuse.log.

- **26/11/2026, Codex**, F4 accettazione build riuso scala. Build Windows exit 0. Log build/f4-accept/build-cost.log.

- **26/11/2026, Codex**, F4 accettazione riuso scala SPIR-V. Scala scritta una volta per coppia di tile e una barriera in meno. Compilazione e validazione SPIR-V exit 0.

- **26/11/2026, Codex**, F4 accettazione profilo costo. Grafo GPU di verifica (mediane 21 e 21 grafi) 55.66 -> 51.58 ms. Trellis a 4 token 51.93 -> 47.72 ms. MLP up 19.53 -> 18.06 ms e down 12.20 -> 10.39 ms. bande della testa bozza 2.47 -> 2.00 ms. Profilazione seriale con overhead. Log build/f4-accept/cost-*.log.

- **26/11/2026, Codex**, F4 accettazione cost-B-t1.0 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.54 GiB. Log build/f4-accept/cost-B-t1.0.

- **26/11/2026, Codex**, F4 accettazione cost-B-t1.0-chat-42. Seed 42, prefill 153 token, generazione 64 a 43.04 t/s, accettazione 73.3%, ciclo stimato 73.19 ms. VRAM prima della richiesta 1.380 GiB. Log build/f4-accept/cost-B-t1.0-chat-42.json.

- **26/11/2026, Codex**, F4 accettazione cost-A-t1.0 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.581 GiB. Log build/f4-accept/cost-A-t1.0.

- **26/11/2026, Codex**, F4 accettazione cost-A-t1.0-chat-42. Seed 42, prefill 153 token, generazione 64 a 40.44 t/s, accettazione 73.3%, ciclo stimato 77.89 ms. VRAM prima della richiesta 1.388 GiB. Log build/f4-accept/cost-A-t1.0-chat-42.json.

- **26/11/2026, Codex**, F4 accettazione temperatura B 1.2. Chat, codice, 32K (3 seed per prompt). chat 56.27 t/s, accettazione 65.3%, ciclo stimato 52.76 ms. code 55.26 t/s, accettazione 63.2%, ciclo stimato 52.74 ms. long 51.67 t/s, accettazione 65.1%, ciclo stimato 57.46 ms.

- **26/11/2026, Codex**, F4 accettazione temperatura B 1.0. Chat, codice, 32K (3 seed per prompt). chat 55.17 t/s, accettazione 63.6%, ciclo stimato 52.82 ms. code 55.28 t/s, accettazione 64.1%, ciclo stimato 52.98 ms. long 52.92 t/s, accettazione 64.8%, ciclo stimato 55.73 ms.

- **26/11/2026, Codex**, F4 accettazione temperatura B 0.8. Chat, codice, 32K (3 seed per prompt). chat 60.03 t/s, accettazione 72.9%, ciclo stimato 52.72 ms. code 54.43 t/s, accettazione 62.2%, ciclo stimato 52.82 ms. long 53.25 t/s, accettazione 67.4%, ciclo stimato 56.84 ms.

- **26/11/2026, Codex**, F4 accettazione temperatura A 1.0. Chat, codice, 32K (3 seed per prompt). chat 51.38 t/s, accettazione 66.5%, ciclo stimato 58.39 ms. code 52.03 t/s, accettazione 67.4%, ciclo stimato 58.27 ms. long 46.88 t/s, accettazione 64.2%, ciclo stimato 62.47 ms.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.2 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.577 GiB. Log build/f4-accept/tune-B-t1.2.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.2-long-777. Seed 777, prefill 4 token, generazione 192 a 51.61 t/s, accettazione 64.6%, ciclo stimato 56.93 ms. VRAM prima della richiesta 1.107 GiB. Log build/f4-accept/tune-B-t1.2-long-777.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.2-long-123. Seed 123, prefill 4 token, generazione 192 a 51.25 t/s, accettazione 66.1%, ciclo stimato 58.23 ms. VRAM prima della richiesta 1.108 GiB. Log build/f4-accept/tune-B-t1.2-long-123.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.2-long-42. Seed 42, prefill 32145 token, generazione 192 a 51.33 t/s, accettazione 64.6%, ciclo stimato 57.24 ms. VRAM prima della richiesta 1.263 GiB. Log build/f4-accept/tune-B-t1.2-long-42.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.2-code-777. Seed 777, prefill 4 token, generazione 192 a 57.61 t/s, accettazione 67.7%, ciclo stimato 52.62 ms. VRAM prima della richiesta 1.263 GiB. Log build/f4-accept/tune-B-t1.2-code-777.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.2-code-123. Seed 123, prefill 4 token, generazione 192 a 53.25 t/s, accettazione 60.6%, ciclo stimato 53.00 ms. VRAM prima della richiesta 1.266 GiB. Log build/f4-accept/tune-B-t1.2-code-123.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.2-code-42. Seed 42, prefill 99 token, generazione 192 a 54.23 t/s, accettazione 61.7%, ciclo stimato 52.57 ms. VRAM prima della richiesta 1.256 GiB. Log build/f4-accept/tune-B-t1.2-code-42.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.2-chat-777. Seed 777, prefill 4 token, generazione 192 a 54.15 t/s, accettazione 61.7%, ciclo stimato 52.65 ms. VRAM prima della richiesta 1.265 GiB. Log build/f4-accept/tune-B-t1.2-chat-777.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.2-chat-123. Seed 123, prefill 4 token, generazione 192 a 54.94 t/s, accettazione 63.6%, ciclo stimato 52.68 ms. VRAM prima della richiesta 1.271 GiB. Log build/f4-accept/tune-B-t1.2-chat-123.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.2-chat-42. Seed 42, prefill 153 token, generazione 192 a 59.10 t/s, accettazione 71.0%, ciclo stimato 52.98 ms. VRAM prima della richiesta 1.380 GiB. Log build/f4-accept/tune-B-t1.2-chat-42.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.0 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.578 GiB. Log build/f4-accept/tune-B-t1.0.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.0-long-777. Seed 777, prefill 4 token, generazione 192 a 50.62 t/s, accettazione 60.6%, ciclo stimato 55.76 ms. VRAM prima della richiesta 1.112 GiB. Log build/f4-accept/tune-B-t1.0-long-777.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.0-long-123. Seed 123, prefill 4 token, generazione 192 a 54.58 t/s, accettazione 68.6%, ciclo stimato 55.84 ms. VRAM prima della richiesta 1.112 GiB. Log build/f4-accept/tune-B-t1.0-long-123.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.0-long-42. Seed 42, prefill 32145 token, generazione 192 a 52.87 t/s, accettazione 65.6%, ciclo stimato 55.58 ms. VRAM prima della richiesta 1.264 GiB. Log build/f4-accept/tune-B-t1.0-long-42.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.0-code-777. Seed 777, prefill 4 token, generazione 192 a 52.55 t/s, accettazione 60.0%, ciclo stimato 53.19 ms. VRAM prima della richiesta 1.259 GiB. Log build/f4-accept/tune-B-t1.0-code-777.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.0-code-123. Seed 123, prefill 4 token, generazione 192 a 54.97 t/s, accettazione 64.3%, ciclo stimato 53.18 ms. VRAM prima della richiesta 1.259 GiB. Log build/f4-accept/tune-B-t1.0-code-123.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.0-code-42. Seed 42, prefill 99 token, generazione 192 a 57.70 t/s, accettazione 68.3%, ciclo stimato 52.55 ms. VRAM prima della richiesta 1.265 GiB. Log build/f4-accept/tune-B-t1.0-code-42.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.0-chat-777. Seed 777, prefill 4 token, generazione 192 a 51.07 t/s, accettazione 57.1%, ciclo stimato 52.92 ms. VRAM prima della richiesta 1.261 GiB. Log build/f4-accept/tune-B-t1.0-chat-777.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.0-chat-123. Seed 123, prefill 4 token, generazione 192 a 58.50 t/s, accettazione 69.9%, ciclo stimato 52.66 ms. VRAM prima della richiesta 1.267 GiB. Log build/f4-accept/tune-B-t1.0-chat-123.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t1.0-chat-42. Seed 42, prefill 153 token, generazione 192 a 55.60 t/s, accettazione 64.6%, ciclo stimato 52.85 ms. VRAM prima della richiesta 1.385 GiB. Log build/f4-accept/tune-B-t1.0-chat-42.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t0.8 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.582 GiB. Log build/f4-accept/tune-B-t0.8.

- **26/11/2026, Codex**, F4 accettazione tune-B-t0.8-long-777. Seed 777, prefill 4 token, generazione 192 a 50.25 t/s, accettazione 61.1%, ciclo stimato 56.17 ms. VRAM prima della richiesta 1.108 GiB. Log build/f4-accept/tune-B-t0.8-long-777.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t0.8-long-123. Seed 123, prefill 4 token, generazione 192 a 56.71 t/s, accettazione 76.1%, ciclo stimato 57.41 ms. VRAM prima della richiesta 1.108 GiB. Log build/f4-accept/tune-B-t0.8-long-123.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t0.8-long-42. Seed 42, prefill 32145 token, generazione 192 a 52.35 t/s, accettazione 66.1%, ciclo stimato 57.01 ms. VRAM prima della richiesta 1.259 GiB. Log build/f4-accept/tune-B-t0.8-long-42.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t0.8-code-777. Seed 777, prefill 4 token, generazione 192 a 51.06 t/s, accettazione 57.1%, ciclo stimato 52.94 ms. VRAM prima della richiesta 1.255 GiB. Log build/f4-accept/tune-B-t0.8-code-777.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t0.8-code-123. Seed 123, prefill 4 token, generazione 192 a 52.57 t/s, accettazione 59.7%, ciclo stimato 52.91 ms. VRAM prima della richiesta 1.264 GiB. Log build/f4-accept/tune-B-t0.8-code-123.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t0.8-code-42. Seed 42, prefill 99 token, generazione 192 a 59.52 t/s, accettazione 71.0%, ciclo stimato 52.60 ms. VRAM prima della richiesta 1.261 GiB. Log build/f4-accept/tune-B-t0.8-code-42.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t0.8-chat-777. Seed 777, prefill 4 token, generazione 192 a 57.65 t/s, accettazione 69.3%, ciclo stimato 52.59 ms. VRAM prima della richiesta 1.261 GiB. Log build/f4-accept/tune-B-t0.8-chat-777.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t0.8-chat-123. Seed 123, prefill 4 token, generazione 192 a 59.48 t/s, accettazione 72.7%, ciclo stimato 52.64 ms. VRAM prima della richiesta 1.272 GiB. Log build/f4-accept/tune-B-t0.8-chat-123.json.

- **26/11/2026, Codex**, F4 accettazione tune-B-t0.8-chat-42. Seed 42, prefill 153 token, generazione 192 a 62.20 t/s, accettazione 77.0%, ciclo stimato 52.94 ms. VRAM prima della richiesta 1.380 GiB. Log build/f4-accept/tune-B-t0.8-chat-42.json.

- **26/11/2026, Codex**, F4 accettazione tune-A-t1.0 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.576 GiB. Log build/f4-accept/tune-A-t1.0.

- **26/11/2026, Codex**, F4 accettazione tune-A-t1.0-long-777. Seed 777, prefill 4 token, generazione 192 a 45.86 t/s, accettazione 60.6%, ciclo stimato 61.55 ms. VRAM prima della richiesta 1.112 GiB. Log build/f4-accept/tune-A-t1.0-long-777.json.

- **26/11/2026, Codex**, F4 accettazione tune-A-t1.0-long-123. Seed 123, prefill 4 token, generazione 192 a 49.32 t/s, accettazione 71.0%, ciclo stimato 62.47 ms. VRAM prima della richiesta 1.112 GiB. Log build/f4-accept/tune-A-t1.0-long-123.json.

- **26/11/2026, Codex**, F4 accettazione tune-A-t1.0-long-42. Seed 42, prefill 32145 token, generazione 192 a 44.95 t/s, accettazione 61.7%, ciclo stimato 63.41 ms. VRAM prima della richiesta 1.257 GiB. Log build/f4-accept/tune-A-t1.0-long-42.json.

- **26/11/2026, Codex**, F4 accettazione tune-A-t1.0-code-777. Seed 777, prefill 4 token, generazione 192 a 49.66 t/s, accettazione 63.6%, ciclo stimato 58.27 ms. VRAM prima della richiesta 1.262 GiB. Log build/f4-accept/tune-A-t1.0-code-777.json.

- **26/11/2026, Codex**, F4 accettazione tune-A-t1.0-code-123. Seed 123, prefill 4 token, generazione 192 a 54.66 t/s, accettazione 72.8%, ciclo stimato 58.24 ms. VRAM prima della richiesta 1.262 GiB. Log build/f4-accept/tune-A-t1.0-code-123.json.

- **26/11/2026, Codex**, F4 accettazione tune-A-t1.0-code-42. Seed 42, prefill 99 token, generazione 192 a 51.20 t/s, accettazione 66.1%, ciclo stimato 58.29 ms. VRAM prima della richiesta 1.265 GiB. Log build/f4-accept/tune-A-t1.0-code-42.json.

- **26/11/2026, Codex**, F4 accettazione tune-A-t1.0-chat-777. Seed 777, prefill 4 token, generazione 192 a 47.50 t/s, accettazione 59.4%, ciclo stimato 58.28 ms. VRAM prima della richiesta 1.259 GiB. Log build/f4-accept/tune-A-t1.0-chat-777.json.

- **26/11/2026, Codex**, F4 accettazione tune-A-t1.0-chat-123. Seed 123, prefill 4 token, generazione 192 a 54.60 t/s, accettazione 72.8%, ciclo stimato 58.30 ms. VRAM prima della richiesta 1.270 GiB. Log build/f4-accept/tune-A-t1.0-chat-123.json.

- **26/11/2026, Codex**, F4 accettazione tune-A-t1.0-chat-42. Seed 42, prefill 153 token, generazione 192 a 51.73 t/s, accettazione 68.3%, ciclo stimato 58.60 ms. VRAM prima della richiesta 1.379 GiB. Log build/f4-accept/tune-A-t1.0-chat-42.json.

- **26/11/2026, Codex**, F4 accettazione diagnosi prefissi. Primo ciclo, ancora e prefisso della proposta identici nelle coppie. posizione 1 (10 coppie), overlap A/B 0.4372/0.4405, TV target 0.0078, TV bozza 0.0313; posizione 2 (8 coppie), overlap A/B 0.5445/0.5436, TV target 0.0110, TV bozza 0.0167; posizione 3 (8 coppie), overlap A/B 0.3060/0.3000, TV target 0.0185, TV bozza 0.0375. Log build/f4-accept/fixed-comparison.json.

- **26/11/2026, Codex**, F4 accettazione fixed-B-t1.0 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.574 GiB. Log build/f4-accept/fixed-B-t1.0.

- **26/11/2026, Codex**, F4 accettazione fixed-B-t1.0-long-0-1. Prefisso fisso, ancora 6054, proposte [28527, 2920, 501], sovrapposizione p/q [0.3565460142, 0.691888306, 0.56168348]. VRAM prima della richiesta 1.113 GiB. Log build/f4-accept/fixed-B-t1.0-long-0-1.json.

- **26/11/2026, Codex**, F4 accettazione fixed-B-t1.0-long-0-0. Prefisso fisso, ancora 16, proposte [13, 2972, 49], sovrapposizione p/q [0.9483143399999999, 0.301308917, 0.286045211]. VRAM prima della richiesta 1.264 GiB. Log build/f4-accept/fixed-B-t1.0-long-0-0.json.

- **26/11/2026, Codex**, F4 accettazione fixed-B-t1.0-code-0-3. Prefisso fisso, ancora 4931, proposte [2920, 3413, 450], sovrapposizione p/q [0.03923937, 0.01889149, 0]. VRAM prima della richiesta 1.259 GiB. Log build/f4-accept/fixed-B-t1.0-code-0-3.json.

- **26/11/2026, Codex**, F4 accettazione fixed-B-t1.0-code-0-2. Prefisso fisso, ancora 3687, proposte [1433, 25, 271], sovrapposizione p/q [0.5654554580000001, 0.44578983, 0.390712666]. VRAM prima della richiesta 1.264 GiB. Log build/f4-accept/fixed-B-t1.0-code-0-2.json.

- **26/11/2026, Codex**, F4 accettazione fixed-B-t1.0-code-0-1. Prefisso fisso, ancora 177198, proposte [11, 585, 500], sovrapposizione p/q [0.6495292300000001, 0.04249488, 0.234934]. VRAM prima della richiesta 1.259 GiB. Log build/f4-accept/fixed-B-t1.0-code-0-1.json.

- **26/11/2026, Codex**, F4 accettazione fixed-B-t1.0-code-0-0. Prefisso fisso, ancora 464, proposte [2857, 271, 727], sovrapposizione p/q [0.9759696, 0.5924392310000001, 0.645082665]. VRAM prima della richiesta 1.262 GiB. Log build/f4-accept/fixed-B-t1.0-code-0-0.json.

- **26/11/2026, Codex**, F4 accettazione fixed-B-t1.0-chat-0-3. Prefisso fisso, ancora 4931, proposte [1542, 384, 378], sovrapposizione p/q [0.45675306, 0.22820281199999998, 0.05848543]. VRAM prima della richiesta 1.268 GiB. Log build/f4-accept/fixed-B-t1.0-chat-0-3.json.

- **26/11/2026, Codex**, F4 accettazione fixed-B-t1.0-chat-0-2. Prefisso fisso, ancora 3687, proposte [829, 7320, 13], sovrapposizione p/q [0.41331399, 0.9969922, 0.646086294]. VRAM prima della richiesta 1.266 GiB. Log build/f4-accept/fixed-B-t1.0-chat-0-2.json.

- **26/11/2026, Codex**, F4 accettazione fixed-B-t1.0-chat-0-1. Prefisso fisso, ancora 271, proposte [17, 13, 2025], sovrapposizione p/q [0, 0.9999872, 0.02597712]. VRAM prima della richiesta 1.289 GiB. Log build/f4-accept/fixed-B-t1.0-chat-0-1.json.

- **26/11/2026, Codex**, F4 accettazione fixed-B-t1.0-chat-0-0. Prefisso fisso, ancora 271, proposte [16, 13, 198], sovrapposizione p/q [0, 0.70491896, 0]. VRAM prima della richiesta 1.381 GiB. Log build/f4-accept/fixed-B-t1.0-chat-0-0.json.

- **26/11/2026, Codex**, F4 accettazione fixed-A-t1.0 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.575 GiB. Log build/f4-accept/fixed-A-t1.0.

- **26/11/2026, Codex**, F4 accettazione fixed-A-t1.0-long-0-1. Prefisso fisso, ancora 6054, proposte [28527, 2920, 501], sovrapposizione p/q [0.3722746503, 0.6981725409999999, 0.60270732]. VRAM prima della richiesta 1.113 GiB. Log build/f4-accept/fixed-A-t1.0-long-0-1.json.

- **26/11/2026, Codex**, F4 accettazione fixed-A-t1.0-long-0-0. Prefisso fisso, ancora 16, proposte [13, 2972, 49], sovrapposizione p/q [0.95077162, 0.30799988300000003, 0.28829199]. VRAM prima della richiesta 1.259 GiB. Log build/f4-accept/fixed-A-t1.0-long-0-0.json.

- **26/11/2026, Codex**, F4 accettazione fixed-A-t1.0-code-0-3. Prefisso fisso, ancora 4931, proposte [2920, 3413, 450], sovrapposizione p/q [0.06415065, 0.02215113, 0]. VRAM prima della richiesta 1.259 GiB. Log build/f4-accept/fixed-A-t1.0-code-0-3.json.

- **26/11/2026, Codex**, F4 accettazione fixed-A-t1.0-code-0-2. Prefisso fisso, ancora 3687, proposte [2972, 1789, 632], sovrapposizione p/q [0.5370358670000001, 0, 0.54806977]. VRAM prima della richiesta 1.266 GiB. Log build/f4-accept/fixed-A-t1.0-code-0-2.json.

- **26/11/2026, Codex**, F4 accettazione fixed-A-t1.0-code-0-1. Prefisso fisso, ancora 177198, proposte [11, 585, 25], sovrapposizione p/q [0.61650281, 0.04937131, 0.2284746]. VRAM prima della richiesta 1.259 GiB. Log build/f4-accept/fixed-A-t1.0-code-0-1.json.

- **26/11/2026, Codex**, F4 accettazione fixed-A-t1.0-code-0-0. Prefisso fisso, ancora 464, proposte [2857, 271, 727], sovrapposizione p/q [0.9688834, 0.5859926010000001, 0.6445583934]. VRAM prima della richiesta 1.266 GiB. Log build/f4-accept/fixed-A-t1.0-code-0-0.json.

- **26/11/2026, Codex**, F4 accettazione fixed-A-t1.0-chat-0-3. Prefisso fisso, ancora 4931, proposte [1870, 384, 378], sovrapposizione p/q [0.46198344, 0.14040981, 0.09991499000000001]. VRAM prima della richiesta 1.262 GiB. Log build/f4-accept/fixed-A-t1.0-chat-0-3.json.

- **26/11/2026, Codex**, F4 accettazione fixed-A-t1.0-chat-0-2. Prefisso fisso, ancora 3687, proposte [829, 7320, 13], sovrapposizione p/q [0.400422115, 0.9959804, 0.657113817]. VRAM prima della richiesta 1.270 GiB. Log build/f4-accept/fixed-A-t1.0-chat-0-2.json.

- **26/11/2026, Codex**, F4 accettazione fixed-A-t1.0-chat-0-1. Prefisso fisso, ancora 271, proposte [17, 13, 2025], sovrapposizione p/q [0, 0.9999875, 0.02670928]. VRAM prima della richiesta 1.283 GiB. Log build/f4-accept/fixed-A-t1.0-chat-0-1.json.

- **26/11/2026, Codex**, F4 accettazione fixed-A-t1.0-chat-0-0. Prefisso fisso, ancora 271, proposte [16, 13, 198], sovrapposizione p/q [0, 0.696594469, 0]. VRAM prima della richiesta 1.379 GiB. Log build/f4-accept/fixed-A-t1.0-chat-0-0.json.

- **26/11/2026, Codex**, F4 INT32 sintesi 32 input. Scala su 32 input e accumulo INT32 su due tile, opt-in. TRL3 a quattro token, up 150,94 -> 131,86 us (12,6% in meno), down 174,53 -> 149,22 us (14,5% in meno). TRL4 up 148,92 -> 133,17 us (10,6% in meno), down 172,68 -> 155,10 us (10,2% in meno). DFlash2 A-B-B-A sullo stesso seed: breve 50,99 -> 51,26 t/s, a 32K 46,50 -> 51,60 t/s. Ciclo stimato a 32K 62,41 -> 56,24 ms, accettazione 64,1% -> 64,1%. Vulkan 416/416, CTest 4/4, CPU top-1 31/31 e stessi token greedy. KL massima 0,00013943. Log build/f4-int32. Sul breve il ciclo stimato 59,15 -> 53,38 ms e l'accettazione 67,2% -> 58,6%. Il ciclo è stimato dai millisecondi di generazione per gruppo di tre proposte. A e B usano lo stesso binario sperimentale e le stesse richieste D2 con seed 42. L'opzione cambia la quantizzazione delle attivazioni della verifica. La variante a 64 input è stata archiviata dopo 410/416 casi. La build candidata è in `build/bin` e si attiva con `GGML_VK_TRELLIS_VERIFY_I32=1`.

- **26/11/2026, Codex**, F4 INT32 logits-grouped32. Top-1 31/31, KL max 0.000139429648072006, stessi token greedy True. Exit 0. Processi concorrenti assenti e 14.576 GiB stimati liberi. Log build/f4-int32/logits-grouped32.

- **26/11/2026, Codex**, F4 INT32 ctest-B-final. Exit 0. VRAM libera 14.576 GiB. 100% tests passed, 0 tests failed out of 4. Log build/f4-int32/ctest-B-final.log.

- **26/11/2026, Codex**, F4 INT32 ops-B-final. Exit 0. VRAM libera 14.576 GiB. 416/416 tests passed. Log build/f4-int32/ops-B-final.log.

- **26/11/2026, Codex**, F4 INT32 ops-A-final. Exit 0. VRAM libera 14.568 GiB. 416/416 tests passed. Log build/f4-int32/ops-A-final.log.

- **26/11/2026, Codex**, F4 INT32 build finale. Build Windows con registrazione della pipeline limitata al supporto GLSL coopmat e dot. Exit 0. Log build/f4-int32/build-final.log.

- **26/11/2026, Codex**, F4 INT32 server-A2 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.573 GiB. Log build/f4-int32/server-A2.

- **26/11/2026, Codex**, F4 INT32 server-A2-cold32k. Prefill 32145 token a 740.13 t/s in 43.43 s, cache 0. Generazione 384 token a 47.13 t/s, accettazione DFlash2 64.1%. Primo token 43.91 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.260 GiB con il solo server di prova. Log build/f4-int32/server-A2-cold32k.json.

- **26/11/2026, Codex**, F4 INT32 server-A2-short. Prefill 153 token a 113.62 t/s in 1.35 s, cache 0. Generazione 384 token a 51.10 t/s, accettazione DFlash2 67.2%. Primo token 1.36 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.377 GiB con il solo server di prova. Log build/f4-int32/server-A2-short.json.

- **26/11/2026, Codex**, F4 INT32 server-B2 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.574 GiB. Log build/f4-int32/server-B2.

- **26/11/2026, Codex**, F4 INT32 server-B2-cold32k. Prefill 32145 token a 738.08 t/s in 43.55 s, cache 0. Generazione 384 token a 50.82 t/s, accettazione DFlash2 64.1%. Primo token 44.10 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.206 GiB con il solo server di prova. Log build/f4-int32/server-B2-cold32k.json.

- **26/11/2026, Codex**, F4 INT32 server-B2-short. Prefill 153 token a 115.57 t/s in 1.32 s, cache 0. Generazione 384 token a 51.86 t/s, accettazione DFlash2 58.6%. Primo token 1.34 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.377 GiB con il solo server di prova. Log build/f4-int32/server-B2-short.json.

- **26/11/2026, Codex**, F4 INT32 server-B1 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.574 GiB. Log build/f4-int32/server-B1.

- **26/11/2026, Codex**, F4 INT32 server-B1-cold32k. Prefill 32145 token a 733.15 t/s in 43.84 s, cache 0. Generazione 384 token a 52.38 t/s, accettazione DFlash2 64.1%. Primo token 44.41 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.256 GiB con il solo server di prova. Log build/f4-int32/server-B1-cold32k.json.

- **26/11/2026, Codex**, F4 INT32 server-B1-short. Prefill 153 token a 113.44 t/s in 1.35 s, cache 0. Generazione 384 token a 50.65 t/s, accettazione DFlash2 58.6%. Primo token 1.35 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.373 GiB con il solo server di prova. Log build/f4-int32/server-B1-short.json.

- **26/11/2026, Codex**, F4 INT32 server-A1 server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.566 GiB. Log build/f4-int32/server-A1.

- **26/11/2026, Codex**, F4 INT32 server-A1-cold32k. Prefill 32145 token a 733.77 t/s in 43.81 s, cache 0. Generazione 384 token a 45.86 t/s, accettazione DFlash2 64.1%. Primo token 44.34 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.254 GiB con il solo server di prova. Log build/f4-int32/server-A1-cold32k.json.

- **26/11/2026, Codex**, F4 INT32 server-A1-short. Prefill 153 token a 107.74 t/s in 1.42 s, cache 0. Generazione 384 token a 50.88 t/s, accettazione DFlash2 67.2%. Primo token 1.42 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.375 GiB con il solo server di prova. Log build/f4-int32/server-A1-short.json.

- **26/11/2026, Codex**, F4 INT32 logits-grouped32. Top-1 31/31, KL max 0.000139429648072006, stessi token greedy True. Exit 0. Processi concorrenti assenti e 14.571 GiB stimati liberi. Log build/f4-int32/logits-grouped32.

- **26/11/2026, Codex**, F4 INT32 ctest-B-32. Exit 0. VRAM libera 14.574 GiB. 100% tests passed, 0 tests failed out of 4. Log build/f4-int32/ctest-B-32.log.

- **26/11/2026, Codex**, F4 INT32 perf-A-32-2. Exit 0. VRAM libera 14.564 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 94.23 us/run - 32816 kB/run - [1;34m 332.13 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7105 runs - 155.52 us/run - 33080 kB/run - [1;34m 202.85 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6090 runs - 178.73 us/run - 33082 kB/run - [1;34m 176.52 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10752 runs - 97.13 us/run - 43696 kB/run - [1;34m 429.02 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 6876 runs - 152.93 us/run - 43960 kB/run - [1;34m 274.14 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6112 runs - 177.59 us/run - 43962 kB/run - [1;34m 236.08 GB/s[0m. Log build/f4-int32/perf-A-32-2.log.

- **26/11/2026, Codex**, F4 INT32 perf-B-32-2. Exit 0. VRAM libera 14.571 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 92.26 us/run - 32816 kB/run - [1;34m 339.22 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 131.58 us/run - 33080 kB/run - [1;34m 239.77 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 149.73 us/run - 33082 kB/run - [1;34m 210.72 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10752 runs - 93.35 us/run - 43696 kB/run - [1;34m 446.42 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 133.18 us/run - 43960 kB/run - [1;34m 314.78 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6876 runs - 155.14 us/run - 43962 kB/run - [1;34m 270.25 GB/s[0m. Log build/f4-int32/perf-B-32-2.log.

- **26/11/2026, Codex**, F4 INT32 perf-B-32-1. Exit 0. VRAM libera 14.571 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 92.54 us/run - 32816 kB/run - [1;34m 338.21 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 132.13 us/run - 33080 kB/run - [1;34m 238.76 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 148.72 us/run - 33082 kB/run - [1;34m 212.13 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11520 runs - 92.88 us/run - 43696 kB/run - [1;34m 448.65 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 133.15 us/run - 43960 kB/run - [1;34m 314.87 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6876 runs - 155.06 us/run - 43962 kB/run - [1;34m 270.38 GB/s[0m. Log build/f4-int32/perf-B-32-1.log.

- **26/11/2026, Codex**, F4 INT32 perf-A-32-1. Exit 0. VRAM libera 14.571 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 91.84 us/run - 32816 kB/run - [1;34m 340.76 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7105 runs - 146.36 us/run - 33080 kB/run - [1;34m 215.56 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6090 runs - 170.33 us/run - 33082 kB/run - [1;34m 185.23 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11520 runs - 92.83 us/run - 43696 kB/run - [1;34m 448.91 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 144.90 us/run - 43960 kB/run - [1;34m 289.33 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6112 runs - 167.77 us/run - 43962 kB/run - [1;34m 249.90 GB/s[0m. Log build/f4-int32/perf-A-32-1.log.

- **26/11/2026, Codex**, F4 INT32 ops-B-32. Exit 0. VRAM libera 14.566 GiB. 416/416 tests passed. Log build/f4-int32/ops-B-32.log.

- **26/11/2026, Codex**, F4 INT32 build 32 input. Build Windows con due tile per accumulatore INT32 e scala. Exit 0. Log build/f4-int32/build-32.log.

- **26/11/2026, Codex**, F4 INT32 preparazione 32 input. Prima sostituzione interrotta prima della build perche il pattern trovava tre cicli. Resa selettiva sul ciclo delle tile.

- **26/11/2026, Codex**, F4 INT32 sintesi 64 input. A-B-B-A TRL3 a quattro token: up 146,01 -> 128,33 us, down 169,54 -> 147,50 us. TRL4 up 145,08 -> 128,16 us, down 167,89 -> 151,65 us. Correttezza 410/416: errore sopra 5e-5 in sei casi. Variante archiviata in grouped64.patch.

- **26/11/2026, Codex**, F4 INT32 perf-A2. Exit 0. VRAM libera 14.575 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 92.50 us/run - 32816 kB/run - [1;34m 338.35 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7105 runs - 146.28 us/run - 33080 kB/run - [1;34m 215.67 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6090 runs - 169.98 us/run - 33082 kB/run - [1;34m 185.61 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10752 runs - 94.44 us/run - 43696 kB/run - [1;34m 441.28 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 144.97 us/run - 43960 kB/run - [1;34m 289.20 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6112 runs - 167.95 us/run - 43962 kB/run - [1;34m 249.62 GB/s[0m. Log build/f4-int32/perf-A2.log.

- **26/11/2026, Codex**, F4 INT32 perf-B2. Exit 0. VRAM libera 14.572 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 92.47 us/run - 32816 kB/run - [1;34m 338.45 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 128.78 us/run - 33080 kB/run - [1;34m 244.98 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 147.97 us/run - 33082 kB/run - [1;34m 213.22 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11520 runs - 92.93 us/run - 43696 kB/run - [1;34m 448.43 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8404 runs - 128.11 us/run - 43960 kB/run - [1;34m 327.24 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6876 runs - 151.31 us/run - 43962 kB/run - [1;34m 277.09 GB/s[0m. Log build/f4-int32/perf-B2.log.

- **26/11/2026, Codex**, F4 INT32 perf-B1. Exit 0. VRAM libera 14.572 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 92.06 us/run - 32816 kB/run - [1;34m 339.95 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8120 runs - 127.88 us/run - 33080 kB/run - [1;34m 246.70 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 7105 runs - 147.03 us/run - 33082 kB/run - [1;34m 214.59 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10752 runs - 93.75 us/run - 43696 kB/run - [1;34m 444.51 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 8404 runs - 128.20 us/run - 43960 kB/run - [1;34m 327.03 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6876 runs - 151.99 us/run - 43962 kB/run - [1;34m 275.84 GB/s[0m. Log build/f4-int32/perf-B1.log.

- **26/11/2026, Codex**, F4 INT32 perf-A1. Exit 0. VRAM libera 14.572 GiB. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 92.11 us/run - 32816 kB/run - [1;34m 339.79 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7105 runs - 145.73 us/run - 33080 kB/run - [1;34m 216.48 GB/s[0m | TRELLIS(type=trl3,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6090 runs - 169.09 us/run - 33082 kB/run - [1;34m 186.58 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11520 runs - 92.46 us/run - 43696 kB/run - [1;34m 450.70 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 145.18 us/run - 43960 kB/run - [1;34m 288.77 GB/s[0m | TRELLIS(type=trl4,k=17408,n=5120,t=4,rows=[1,128,1],cols=[1,1024,1],parts=[0,0],gather=0): 6112 runs - 167.82 us/run - 43962 kB/run - [1;34m 249.82 GB/s[0m. Log build/f4-int32/perf-A1.log.

- **26/11/2026, Codex**, F4 INT32 ops-B. Exit 1. VRAM libera 14.568 GiB. [TRELLIS] ERR = 0.000052434 > 0.000050000   TRELLIS(type=trl1,k=256,n=256,t=3,rows=[1,2,1],cols=[1,2,1],parts=[0,0],gather=1): [1;31mFAIL[0m | [TRELLIS] ERR = 0.000051856 > 0.000050000   TRELLIS(type=trl1,k=512,n=256,t=3,rows=[1,128,1],cols=[1,128,1],parts=[0,0],gather=1,x_pad=16): [1;31mFAIL[0m | [TRELLIS] ERR = 0.000051519 > 0.000050000   TRELLIS(type=trl2,k=512,n=256,t=3,rows=[1,128,1],cols=[1,128,1],parts=[0,0],gather=1,x_pad=16): [1;31mFAIL[0m | [TRELLIS] ERR = 0.000054193 > 0.000050000   TRELLIS(type=trl4,k=256,n=256,t=3,rows=[1,2,1],cols=[1,2,1],parts=[0,0],gather=0): [1;31mFAIL[0m | [TRELLIS] ERR = 0.000051553 > 0.000050000   TRELLIS(type=trl4,k=8192,n=64,t=3,rows=[0,0,1],cols=[1,8192,1],parts=[0,0],gather=1): [1;31mFAIL[0m | [TRELLIS] ERR = 0.000052238 > 0.000050000   TRELLIS(type=trl5,k=512,n=256,t=3,rows=[1,128,1],cols=[1,128,1],parts=[0,0],gather=1,x_pad=16): [1;31mFAIL[0m | 410/416 tests passed | Failing tests: | Backend Vulkan0: [1;31mFAIL[0m | [1;31mFAIL[0m. Log build/f4-int32/ops-B.log.

- **26/11/2026, Codex**, F4 INT32 build. Build Windows del prototipo opt-in con scala condivisa su 64 input e accumulo INT32. Exit 0. Log build/f4-int32/build.log.

- **26/11/2026, Codex**, F4 INT32 spirv. Due shader della variante raggruppata, compilazione e validazione SPIR-V. Exit 0.

- **26/11/2026, Codex**, F4 INT32 baseline. Build 128. Exit 0. TRELLIS(type=trl3,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 11253 runs - 92.19 us/run - 32816 kB/run - [1;34m 339.47 GB/s[0m | TRELLIS(type=trl3,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7105 runs - 145.69 us/run - 33080 kB/run - [1;34m 216.55 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=1,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 10752 runs - 93.19 us/run - 43696 kB/run - [1;34m 447.19 GB/s[0m | TRELLIS(type=trl4,k=5120,n=17408,t=4,rows=[1,1024,1],cols=[1,128,1],parts=[0,0],gather=0): 7640 runs - 145.16 us/run - 43960 kB/run - [1;34m 288.81 GB/s[0m. Controllo GPU e log in build/f4-int32/baseline.

- **26/11/2026, Codex**, D2 build 128 confronto finale. 32K a cache fredda: NariQ 580,40 -> 726,08 t/s di prefill e 47,87 -> 45,70 t/s di generazione. Unsloth 878,22 -> 882,07 t/s di prefill e 100,26 -> 105,69 t/s di generazione. 65K a cache fredda: NariQ 531,11 -> 644,36 t/s di prefill e 44,00 -> 41,19 t/s di generazione. Unsloth 752,03 -> 750,75 t/s di prefill e 84,24 -> 87,35 t/s di generazione. Picco VRAM dedicata del processo 12,74 / 13,61 GiB (Unsloth / NariQ). Stessi prompt D2 e stessa build 128 per entrambi. Log build/d2-b128.

- **26/11/2026, Codex**, D2 build 128 nariq server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.585 GiB. Log build/d2-b128/nariq.

- **26/11/2026, Codex**, D2 build 128 nariq-cold65k. Prefill 65145 token a 644.36 t/s in 101.10 s, cache 0. Generazione 384 token a 41.19 t/s, accettazione DFlash2 55.3%. Primo token 101.16 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.109 GiB con il solo server di prova. Log build/d2-b128/nariq-cold65k.json.

- **26/11/2026, Codex**, D2 build 128 nariq-warm32k. Prefill 53 token a 31.88 t/s in 1.66 s, cache 32141. Generazione 256 token a 49.59 t/s, accettazione DFlash2 68.7%. Primo token 1.71 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.107 GiB con il solo server di prova. Log build/d2-b128/nariq-warm32k.json.

- **26/11/2026, Codex**, D2 build 128 nariq-cold32k. Prefill 32145 token a 726.08 t/s in 44.27 s, cache 0. Generazione 384 token a 45.70 t/s, accettazione DFlash2 64.1%. Primo token 44.83 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.251 GiB con il solo server di prova. Log build/d2-b128/nariq-cold32k.json.

- **26/11/2026, Codex**, D2 build 128 nariq-short. Prefill 153 token a 104.93 t/s in 1.46 s, cache 0. Generazione 384 token a 51.45 t/s, accettazione DFlash2 67.2%. Primo token 1.46 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 1.357 GiB con il solo server di prova. Log build/d2-b128/nariq-short.json.

- **26/11/2026, Codex**, D2 build 128 production server. Suite terminata con exit 0. VRAM stimata libera prima dell'avvio 14.63 GiB. Log build/d2-b128/production.

- **26/11/2026, Codex**, D2 build 128 production-cold65k. Prefill 65145 token a 750.75 t/s in 86.77 s, cache 0. Generazione 384 token a 87.35 t/s, accettazione DFlash2 63.4%. Primo token 86.86 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 2.023 GiB con il solo server di prova. Log build/d2-b128/production-cold65k.json.

- **26/11/2026, Codex**, D2 build 128 production-warm32k. Prefill 53 token a 47.36 t/s in 1.12 s, cache 32141. Generazione 256 token a 85.90 t/s, accettazione DFlash2 51.5%. Primo token 1.16 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 2.016 GiB con il solo server di prova. Log build/d2-b128/production-warm32k.json.

- **26/11/2026, Codex**, D2 build 128 production-cold32k. Prefill 32145 token a 882.07 t/s in 36.44 s, cache 0. Generazione 384 token a 105.69 t/s, accettazione DFlash2 70.5%. Primo token 36.74 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 2.263 GiB con il solo server di prova. Log build/d2-b128/production-cold32k.json.

- **26/11/2026, Codex**, D2 build 128 production-short. Prefill 153 token a 132.05 t/s in 1.16 s, cache 0. Generazione 384 token a 106.24 t/s, accettazione DFlash2 61.5%. Primo token 1.16 s, primo testo assente al limite di generazione. VRAM libera stimata al preflight 2.363 GiB con il solo server di prova. Log build/d2-b128/production-short.json.

- **26/11/2026 notte, Claude**, `70dc70ae1`: limite di 64 token al micro-batch della bozza DFlash/DSpark solo con target Trellis (chiave `trellis.version` del GGUF). Unsloth UD-Q2_K_XL con DFlash2 sulla build NariQ: prefill di 2,5K token 662 → 716 t/s (come la produzione `d4b007d`: 716), generazione identica alla produzione seed per seed (114,7 / 112,0 / 108,3). NariQ resta a 64 (verificato nel log del contesto della bozza). Con questo il ramo esegue sia Unsloth (prestazioni della produzione) sia NariQ.
  - **OpenClaw**: profilo `qwen38-nariq` in `scripts\Start-OpenClaw-Local.ps1` (stesse impostazioni di `qwen38`, GGUF e mmproj letti da questa cartella, build in `runtime\backends\llama.cpp-nariq-vulkan-win-x64-70dc70ae1`), voce in `profiles\openclaw-profiles.json`, server riconosciuto da `Stop-OpenClaw-Local.ps1`; launcher `Avvia OpenClaw Qwen38 NariQ.bat` sul desktop e nella cartella di OpenClaw. Copie dei file prima della modifica in `runtime\migrations\2026-11-26-nariq-profile\backup`. Non avviato da Claude.
  - **Ramo per la PR verso Nari Labs**: `nariq-vulkan-pr` (worktree `D:\Progetti\NariQ Vulkan PR`, sopra `nari-labs/nari-llama.cpp` `nariq/b12431`), 4 commit: operatori Trellis Vulkan, condivisione di embedding e testa Trellis con le bozze DFlash, limite del micro-batch della bozza, limite solo per i target Trellis. 416/416, ctest 4/4 (test-trellis anche su Vulkan0), logits contro CPU a micro-batch 1 / 4 / 32 KL massima 2,1e-4 / 1,7e-4 / 4,4e-5; llama-bench pp512 ~977, tg64 ~27,6; DFlash2 con NariQ (opzioni standard `draft-dflash` / `draft-dflash2`) 44 t/s. Per la build: `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF` e asset della UI copiati da una build precedente (il download da Hugging Face fallisce).

- **26/11/2026, Claude**, commit di questa voce: opzioni Vulkan Trellis ora predefinite: `GGML_VK_TRELLIS_Q8_PACKED`, `VERIFY_CM`, `MM_WIDE`, `BLOCK_SHUFFLE`, `MM_BIG`, `FUSE` (ognuna si spegne con `=0`; `MM_RESIDUAL` resta opzionale con `=1`). Senza variabili d'ambiente: 416/416, ctest 4/4, logits contro CPU a micro-batch 1 / 4 / 8 KL massima 2,0e-4 / 2,1e-4 / 4,4e-4 con top-1 31/31, KLD su 385 token 3,6e-5 con top-1 100%; llama-bench pp512 990, tg64 28,5, pp4 79,6. Gli script di prova che impostano ancora le variabili a 1 restano validi.

- **26/11/2026, Claude**: riepilogo della giornata, tutte le misure nella stessa sessione (VRAM libera 14,8 GB, nessun altro processo GPU). "Stamattina" = `build/ref-bin` (`805178e04`, C8) con `Q8_PACKED`, `VERIFY_CM`, `MM_WIDE`, `BLOCK_SHUFFLE`; "ora" = `ea6dbac44` con in più `MM_BIG` e `FUSE`; HIP = PR #8 (worktree `pr8`, ROCm 10.2).

  | llama-bench (cache q4_0, FA, b/ub 512) | Stamattina | Ora | HIP PR #8 |
  | --- | ---: | ---: | ---: |
  | pp512 | 720 | 946 | 1219 |
  | pp512 a 32K | 546 | 693 | 772 |
  | tg64 | 25,3 | 28,6 | 31,5 |
  | tg64 a 32K | 23,9 | 27,7 | 27,3 |
  | pp4 (batch della verifica a 4 token) | 75,2 | 76,6 | 62,4 |

  | Server stile OpenClaw (chat reale, temperatura 1,0, cache come OpenClaw) | Stamattina | Ora | HIP PR #8 |
  | --- | ---: | ---: | ---: |
  | Prefill 2502 token | 470-480 | 534-550 | 997 |
  | Prefill 27487 token | 596 | 742 | 1000 |
  | Generazione breve con DFlash2 3 token (media 3 seed) | 44,4 (716/1305 accettati) | 40,5 (677/1426) | non disponibile |
  | Ciclo DFlash2 | 59,7 ms | 59,8 ms | non disponibile |
  | Generazione a ~27K con DFlash2 (1 run) | 41,2 | 41,3 | non disponibile |
  | Generazione senza bozza (breve / ~27K) | | | 30,0 / 28,0 |

  La differenza di generazione con DFlash2 tra stamattina e ora è rumore di accettazione (il ciclo costa uguale: la verifica a 4 token e la testa della bozza non sono cambiate oggi). Prove scartate oggi, archiviate in `build/f6`: prodotto a 1 token stile #7 in FP32 (`trellis_mv2`), prodotto int8 a 128 righe con ingressi in memoria condivisa (`trellis_mv8`, al meglio 94,9 contro 92,7 µs anche con k diviso fino a 8), estrazione degli stati con `bitfieldExtract` (invariato).

- **26/11/2026, Claude**, commit di questa voce: prodotto int8 a 1 token dopo il confronto dell'ISA con HIP.
  - **Confronto ISA** (assembly gfx1201 di `trellis.cu` della PR #8, generato con il comando di build più `--offload-device-only -S`, in `scratchpad`): ciclo principale di `k_product_mv_v2<3,1>` 189 istruzioni VALU per 32 pesi per lane (32 `v_mul_lo_u32`, 34 `v_and`, 32 `v_sad_u8`, 24 `v_bfe`, ~32 FMA in parte doppie), cioè ~9 cicli per peso con la moltiplicazione a un quarto di velocità; il nostro `trellis_matvec_q8` ~11,6 cicli per peso: 9 `v_lshrrev` (lo spostamento `>> 2` dei byte), 10 addizioni per sommare i `dot4`, una moltiplicazione a 32 bit in più per il termine −126.
  - **Modifiche** (formato replicato, 1 token e q8x4 non impacchettato): byte dello stato mascherati con `0xfcfcfcfc` senza spostamento (`trellis_bytes4`, 4 × cifre) e fattore 1/4 nella scala; `dot4` concatenati nell'accumulatore (`dotPacked4x8AccSatEXT(int, uint, int)`, `v_dot4_i32_iu8` a segno misto); termine di correzione `504 × somma` scritto dalla quantizzazione (`trellis_quant.comp` e `trellis_fused.comp`, solo formato replicato) invece della moltiplicazione per tile.
  - **Misure** (opzioni C8 + verifica + `GGML_VK_TRELLIS_FUSE=1`): TRL3 5120 → 17408 a 1 token 106,4 → 97,0 → **93,3 µs**; banda della testa 92,7 → 82,7; tg64 26,06 → 28,32 → **29,06** (HIP 31,8). pp4 invariato (79). 416/416 con e senza opzioni, ctest 4/4; logits contro CPU a micro-batch 1 e 4, con e senza fusione: identici al riferimento (KL massima 2,0e-4 / 2,1e-4, top-1 31/31).
  - **DFlash2 su HIP**: il server della PR #8 non accetta `--spec-accept`, `-bs` e `--spec-draft-vocab` (opzioni del nostro fork) e con le sole opzioni standard (`draft-dflash` o `draft-dflash2`) si ferma: "dflash needs the target model's token embedding and output, but the target has no plain one". Manca la condivisione degli embedding e della testa Trellis con la bozza (`92f88a70e` nel nostro ramo). Senza bozza HIP genera 31,1 t/s sulla richiesta chat (prefill 1046).
  - **Esito**: accettato (il percorso a 1 token è predefinito). Restano rispetto a HIP: estrazione degli stati (4 `v_mov_b16` + 6 `v_bfe` per tile contro `v_alignbit` + `v_bfe`), letture delle attivazioni per tile (`b128` + `b64` + `u16` contro due `ds_load_b64` da memoria condivisa).

- **26/11/2026, Claude**, commit di questa voce: fusione delle fasi dell'operazione Trellis per batch fino a 15 token (`GGML_VK_TRELLIS_FUSE=1`, nuovo `trellis_fused.comp`). Con trasformate a blocchi (tipo 1, blocchi fino a 4096; nel modello sono tutte così): ingresso + trasformata d'ingresso + quantizzazione int8 (formato replicato o impacchettato) in un lancio, trasformata d'uscita + uscita in un altro; un gruppo per blocco e token. A 1 token da 6 a 3 lanci. TRL3 5120 → 17408: 1 token 109,6 → 106,4 µs, 4 token 151,6 → 150,1; TRL5 5120 → 48 a 1 token 7,2 → 6,0. llama-bench (due ripetizioni alternate): tg64 25,80 → 26,06, pp4 78,5 → 79,5 (+1%). 416/416, ctest 4/4; logits contro CPU a micro-batch 1 / 4: KL massima 2,0e-4 / 2,1e-4 (identica al percorso non fuso a 4 token), top-1 31/31.
  - **Profilo per fase a 1 token** (`GGML_VK_PERF_LOGGER=1 GGML_VK_TRELLIS_PROFILE=1`, 1023 operazioni): prodotto (con quantizzazione) ~111 µs, ingresso + rotazione ~5,4, rotazione d'uscita ~2,4, uscita ~1,3. Quindi le fasi separate costano ~9 µs per operazione, non i 26 del divario con HIP (83 contro 109 µs): il divario è nel prodotto a 1 token. Il port diretto di `k_product_mv_v2` (voce precedente) era però più lento: il codice generato dal compilatore HIP per quel kernel resta da confrontare (ISA del code object in `ggml-hip.dll`).
  - **Esito**: accettato, opt-in. Generazione con DFlash2: la verifica a 4 token resta più veloce su Vulkan (pp4 79,5 contro 62,9 di HIP).

- **26/11/2026, Claude**, non committato (archiviato in `build/f6/trellis_mv2.comp.txt`): prodotto a 1 token con la struttura di `k_product_mv_v2` della PR #7 (8 sottogruppi per 128 righe, ingressi FP32 condivisi in memoria, codici di 4 tile letti in modo coalescente e passati dalla memoria condivisa, lettura anticipata, valori magic bias, k diviso fino a ~256 gruppi, niente quantizzazione int8). Corretto (416/416), ma più lento: TRL3 5120 → 17408 a 1 token 109 → 119 µs, banda della testa 95 → 112; 67 VGPR, niente spill; obiettivo di gruppi 1 / 112 / 256 / 512 indifferente. Esito: scartato.
  - **Operazione Trellis isolata, HIP PR #8 contro Vulkan** (stessi casi perf copiati nel `test-backend-ops` del worktree `pr8`, modifica solo locale): TRL3 5120 → 17408 a 1 / 2 / 4 / 16 token HIP 83 / 112 / 143 / 224 µs, Vulkan 109 / 142 / 147 / 175; TRL5 5120 → 48 a 1 / 4 token HIP 1,3 / 121, Vulkan 7,1 / 7,8; banda della testa 5120 → 15488 a 1 / 4 token HIP 71 / 133, Vulkan 95 / 131. HIP è avanti solo a 1 token, di ~26 µs per operazione: con ~300 operazioni Trellis per token fanno ~7,8 ms, cioè tutta la differenza di tg64 (31,4 contro 38,9 ms per token). Il prodotto in sé non è più lento (vedi il tentativo sopra): a 1 token l'operazione Vulkan fa 6-7 lanci con barriera (ingresso, rotazione d'ingresso, quantizzazione, prodotto, eventuale somma degli split, rotazione d'uscita, uscita), HIP fonde ingresso e rotazione (`k_input_rht_128`), rotazione d'uscita da 128 e moltiplicazione finale nell'epilogo del prodotto (`fuse_mode` 2) e i gruppi di proiezioni con lo stesso ingresso.
  - **Prossimo passo per la generazione**: fondere le fasi a 1-16 token (ingresso + rotazione d'ingresso + quantizzazione int8 in un lancio; rotazione d'uscita a blocchi da 128 + uscita nell'epilogo di un prodotto a 128 righe per gruppo), poi misurare tg64 e il ciclo DFlash2.

- **26/11/2026, Claude**, `7b90766a4`: prefill con la struttura della PR HIP #8 e confronto HIP sulla nostra scheda.
  - **HIP, PR #8 (con #7) su RX 9070**, ROCm 10.2 del venv di Codex, worktree `D:\Progetti\NariQ HIP RDNA4\pr8`, build `pr8\build` (DLL `amdhip64_7`, `amd_comgr`, `rocm_kpack` copiate accanto ai binari come in E1): pp512 1229, tg64 31,8, pp4 62,9 (batch da 4 token: 63,6 ms), a 32K pp512 812 / tg64 27,9. Stessi argomenti di B2/B3.
  - **Vulkan stesso momento** (C8 + verifica, opzioni attive): pp512 755, tg64 25,7, pp4 77,6 (51,6 ms): HIP è avanti del 24% a 1 token e del 63% nel prefill, Vulkan del 23% sul batch da 4 token (la verifica DFlash2).
  - **`GGML_VK_TRELLIS_MM_BIG=1`** (nuovo `trellis_mm_big.comp`, sopra `GGML_VK_TRELLIS_MM_WIDE=1`): 128 uscite × 256 token per gruppo, 8 sottogruppi 2 × 4 con 4 × 4 tile di accumulatori ciascuno, pesi e attivazioni in memoria condivisa con doppio buffer (24 KB), lettura del passo successivo durante il prodotto. pp512 755 → 922. Con il valore del peso costruito come float sommando i byte alla mantissa di 2^23 (`grid_value4` della #7: `dotPacked4x8AccSatEXT(hash & 0xfcfcfcfc, 0x01010101, 0x4B000000)` − 8389112 = 4 × valore, esatto) e due FP16 per parola di memoria condivisa: **993**. A 32K: 499 → **691**. k diviso in due sotto 56 gruppi: +0,5% (soglia `GGML_VK_TRELLIS_MM_BIG_SPLIT`, predefinita 56). Usarlo anche per i prodotti da 5120 uscite (40 gruppi) conviene: con soglie minime 48 / 64 / 90 gruppi pp512 850 / 840 / 827. 416/416; KLD contro CPU su 385 token (WikiText-2, `build/c6/cpu-wt2.kld`): massima 3,6e-5 (riferimento 3,7e-5), top-1 100%.
  - **Esito**: accettato, opt-in. Restano da portare dalla #7 la struttura del prodotto a 1 token (8 sottogruppi per 128 righe, attivazioni condivise, letture coalescenti dei codici) e, nel prefill, la lettura coalescente dei codici in memoria condivisa.

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

- **25/11/2026, Codex**, C7 avvio. Riferimento C6 1eca433a3. Profilare il prefill con wide=1 e misurare le componenti MLP prima di scegliere la fusione.

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

- **25/11/2026, Codex**, C6 build Release. BUILD_OK con MSVC e Vulkan SDK. Inizio test e confronto con FP16 del commit ee969e9c3.

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

- **25/11/2026, Codex**, A5b non committato: B2 sul GGUF, A-B-C-C-B-A, riferimento `06d57e2fe`, stesso B6 opt-in su tutti i run. Prefill/tg64 medi A 594,75/23,87 t/s, B due shuffle 601,85/23,63, C letture dirette 606,68/24,33. `tg64` A 23,72 e 24,01, C 24,34 e 24,32 (+1,9% medio). Picco 11,11 GiB dedicati per tutti. Nessun processo concorrente, ~2,13 GiB occupati prima dei run. Esito: selezionato il solo kernel da un token con letture dirette per TRL1–4. La variante multispeculativa mantiene il suo decoder. Patch dei prototipi `build/a5b/two-words-prototypes.patch`, log `b2-*`.

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

- **25/11/2026 notte, Claude**: D2, confronto completo in stile OpenClaw (contesto 114688, cache q4_0, ubatch 512, visione su CPU, DFlash2 3 token con rejection, prompt sintetici dello script: breve, 32K, 65K). NariQ `build/bin` (con `7835d4f05`): gen 25,41 / 23,92 / 25,61 t/s, prefill 522,4 / 432,0, accettati 271/720, 130/373, 146/332, picco VRAM 14,962 GiB + 0,198 condivisa. Unsloth sul fork di produzione `d4b007d`: 86,96 / 90,03 / 78,31, prefill 910,8 / 661,1, accettati 291/661, 159/288, 154/303, 14,056 + 0,236. Ciclo di verifica NariQ 84-90 ms contro 27-32. Esito: con NariQ DFlash2 non aumenta la velocità; prefill al 57-65% di Unsloth; generazione circa 3,4 volte più lenta.
- **25/11/2026 notte, Claude**, `7835d4f05`: memoria della bozza DFlash2 con NariQ. Il contesto della bozza ereditava ubatch 512 e il suo grafo calcola la testa per ogni token: con la testa Trellis (tutti i 248.320 logit in bande, poi riordinati) la riserva era 1437 MiB contro 160 con Unsloth. Selezionare le sole righe di output prima della testa non si può (`dflash.cpp` richiede i logit di tutte le posizioni: assert); ubatch della bozza limitato a 64: 213 MiB, picco VRAM 15,03 + 0,91 GiB condivisa → 14,56 + 0,17 (verifica preliminare dello script). Esito: accettato.

- **25/11/2026 notte, Claude**, `1df83dfb5`: forma del prodotto cooperativo. pp512 per righe × input per passo: 2 × 4 519 t/s, 4 × 4 483, **4 × 2 602-626**, 4 × 1 601, 8 × 1 566-569, 8 × 2 520 (prove alternate, varianti temporanee poi rimosse; 380/380 test TRELLIS per ognuna). Con 4 × 2 e k diviso per le proiezioni da 48 righe: pp512 485 → 602 (con residuo 353 → 452), a 32K 395 → 472, tg64 24,4. KL sul prompt 6,6e-8, top-1 identico. Prima, con 2 × 4, la sola divisione di k aveva dato pp512 ~481 → ~504. 416/416, ctest 4/4. Esito: accettato.

- **25/11/2026 notte, Claude**, `35e3c9cde`: C3, nuovo prodotto cooperativo (`trellis_prep16.comp` + `trellis_matmul_cm.comp`). pp512 150,6 → 485-502 t/s; con il residuo 353; a 32K (`-d 32768`) 146,6 → 395,4; tg64 invariato (24,75). Perf 5120 → 17408 a 16 token: TRL3 1041 → 695 µs. Logits del modello completo con micro-batch da 32 token, sulle 25 posizioni del prompt: KL media 6,7e-8 (un termine FP16), 1,3e-8 (con residuo), 6,4e-9 (kernel precedente), top-1 identico. Un solo termine FP16 senza potenza di 2 per token non era stato provato; con un solo termine `test-trellis-vulkan-coopmat` fallisce di poco (TRL1: -10,3191 contro -10,3261, tolleranza 2e-4 × (1 + |atteso|)), quindi quel test usa la variante con residuo. 416/416 con e senza percorso cooperativo, ctest 4/4. Esito: accettato, percorso cooperativo predefinito.

- **25/11/2026 notte, Claude**, `902513980`: prodotti int8 con meno di 64 gruppi di lavoro divisi lungo k su fino a 16 gruppi, con somma finale (`trellis_reduce.comp`). TRL5 5120 → 48: 1 token 29,8 → 7,1 µs, 4 token 77,0 → 9,5 µs. tg64 24,09 → 24,92; server con DFlash2 3 token 39,47 → 40,75 t/s (accettati 180/225 e 182/228), ciclo ~86,5 → ~82,6 ms. Il profilatore ora nomina i nodi Trellis con tipo, k, n e token. 416/416, ctest 4/4. Esito: accettato.
- **25/11/2026 notte, Claude**: B4, profilo di un ciclo DFlash2 (`GGML_VK_PERF_LOGGER=1`, operazioni in serie). Grafo di verifica a 4 token ~94 ms: prodotti Trellis delle proiezioni grandi (5120 → 17408 TRL3 15,0 ms, TRL2 10,9, 17408 → 5120 TRL3 10,1, ...) e 7,2 ms per le proiezioni da 48 righe; grafo della bozza ~10 ms, di cui ~6 nelle bande della testa Trellis condivisa. In isolamento: banda 5120 → 15488 TRL2 102 µs (1 token) e 197 µs (4 token), normale; 5120 → 48 TRL5 30 / 77 µs, anomalo. Esito: corretto il caso da 48 righe; il resto del ciclo è il costo intrinseco della decodifica.

- **25/11/2026 notte, Claude**: B1 sul server (`-c 32768 -ub 512`, cache q4_0, prompt sintetico ripetitivo da 1527 token, 256 token generati, `GGML_VK_TRELLIS_COOPMAT=1`): senza bozza 23,96 t/s; DFlash2 2 token 38,68 (accettati 168/176); 3 token 42,18 (186/210); 4 token 17,06 (197/232, batch 5 fuori dal percorso int8). Esito: usare 3 token; il ciclo costa ~87 ms, da profilare (B4).
- **25/11/2026 notte, Claude**, `b14dae893`: B2, un gruppo di lavoro per 16 uscite e tutti i token del batch (fino a 4), con variante a 1 token (`trellis_matvec_q8`) e fino a 4 (`trellis_matvec_q8x4`). Nuovi casi perf 5120 → 17408 a 1/2/3/4/16 token. µs per prodotto: TRL3 1 token 111,5 → 112,5, 4 token 407,3 → 215,9; TRL4 105,7 → 107,2 e 386,6 → 217,3. Una prima versione con un solo kernel per tutti i casi rallentava il token singolo dell'11% (124 µs). tg64 24,06, 416/416, ctest 4/4. Esito: accettato.

- **24/11/2026 notte, Claude**, `565d38c7f`: A5a, prodotto int8. Nuovi `trellis_quant.comp` (attivazioni int8 per gruppi di 16, valori replicati per lane + somma e scala) e `trellis_matvec_q8` in `trellis_product.comp`; usato per batch ≤ 4 se il dispositivo ha `integer_dot_product`, `GGML_VK_TRELLIS_Q8=0` torna al prodotto float. Terza area di scratch da 6 byte per input e token (`ggml_vk_trellis_quant_size`). tg64 18,08 → 24,12 t/s; pp512 invariato (150,5). Logits del modello completo, `test-trellis-model --ubatch 1 --predict 6` contro CPU: KL media 4,0e-5, massima 2,2e-4, top-1 31/31 (prodotto float: KL media 1,1e-8). Soglie `test-backend-ops` per t ≤ 4: prodotto e gruppo d'ingresso 5e-5 (misurato ~2e-5), MLP codificato 2e-4 (misurato fino a 8,8e-5); `test-trellis-vulkan*` girano con `GGML_VK_TRELLIS_Q8=0` (controlli elemento per elemento del formato). 416/416, ctest 4/4. Esito: accettato.

- **24/11/2026 sera, Claude** (commit della sessione, vedi `git log`): somma dei 4 byte di `trellis_grid` con spostamenti e somme invece della moltiplicazione per `0x01010101`. Risultato identico, test 416/416, `test-trellis` superato; nel codice macchina di `trellis_matvec` le moltiplicazioni a 32 bit per ciclo passano da 16 a 8. Velocità invariata (tg64 18,07 → 18,05). Esito: tenuta, ma conferma che non basta togliere le moltiplicazioni.
- **24/11/2026 sera, Claude**: prove a esclusione su tg64 (risultati volutamente sbagliati, interruttori temporanei poi rimossi). Normale 17,60 t/s; senza rotazione d'ingresso 18,29; senza prodotto 193,8; senza prodotto né rotazione 230,3; prodotto con decodifica banale (`float(state & 255)`) 35,0. Quindi per token: prodotto ~52 ms (decodifica ~28, letture ~23), resto 5,2 ms, rotazione d'ingresso ~1 ms. `trellis_matvec`: 35 registri, 1 KB di memoria condivisa, ciclo principale ~110 istruzioni per 8 pesi per lane, 4-6 letture a 32 bit dei codici e 2 da 64 bit delle attivazioni. Esito: priorità ad A5a.
- **24/11/2026 sera, Claude**: rotazioni usate dal modello (nuova diagnostica `GGML_VK_TRELLIS_DEBUG=1`, stampa ogni forma una volta): solo `kind=1` a blocchi, lunghezze 128 (larghezze 5120 e 15488), 256 (15616), 1024 (17408 e 1024), 2048 (6144 e 10240), 4096 (12288), sempre `m=1` e senza parti fuse, quindi sempre il kernel a blocchi in memoria condivisa. Esito: A2 scartata.
- **24/11/2026 sera, Claude**: baseline A1 con `build/ref-bin` (`343fe2183`), stessi argomenti di B2. NariQ pp512 153,2, tg64 17,45; Unsloth pp512 1150,2, tg64 39,30. A 32K (`-d 32768 -r 1`): NariQ 146,6 / 17,44, Unsloth 765,8 / 37,27.

- **24/11/2026, Claude**: perplexity WikiText-2, 40 finestre da 512 token, stessa build per i due modelli. NariQ 6,2455 ± 0,147, Unsloth 6,2317 ± 0,148; NariQ meglio in 19 finestre su 40. Alla pari, coerente con le KL di Nari Labs (la perplexity non misura la fedeltà al BF16). Esito: la qualità resta un motivo valido per il cambio.
