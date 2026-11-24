# NariQ su Vulkan per Windows e RDNA4

Stato al 24 novembre 2026. Branch sperimentale `nariq-vulkan-rdna4`.

## Punto di partenza per chi riprende il lavoro

- Il commit `1493ad894` contiene il decoder ottimizzato e il profiler per fase. La build di riferimento per le nuove prove è quella ottenuta da questo commit o da un suo discendente con gli stessi kernel.
- Su questa macchina il binario ottimizzato è in `build/packaging-x64/bin`. Nei pacchetti GitHub, `BUILD_INFO.txt` indica il commit usato per compilare. Per provare il decoder aggiornato da un nuovo checkout, seguire la sezione [Riproduzione delle prove](#riproduzione-delle-prove).
- Baseline sulla RX 9070 con il GGUF 3.0 bpw e prefill cooperativo. 65/65 layer su Vulkan, cache Q4, 150,77 tok/s nel prefill da 256 token e 17,39 tok/s nella generazione da 32 token. Le variazioni vanno confrontate con la stessa configurazione e con una cattura dei logits.
- Per decidere il passo successivo, vedere [Profiling degli operatori Trellis](#profiling-degli-operatori-trellis) e [Priorità di ottimizzazione](#priorità-di-ottimizzazione).

## Confronto dei repository

| Sorgente | Revisione verificata | Ruolo |
| --- | --- | --- |
| [Filters83/rx7900-xtx](https://github.com/Filters83/rx7900-xtx/commit/d4b007d773fdf9052d95bcf6e26d4e719042a67f) | `d4b007d773fdf9052d95bcf6e26d4e719042a67f` | Base `windows-amd-proprietary` |
| [Nari Labs HEAD](https://github.com/nari-labs/nari-llama.cpp/commit/8e0e016e0cd13f153380bde2bec01add08eae85e) | `8e0e016e0cd13f153380bde2bec01add08eae85e` | Riferimento formato v1 e runtime |
| [Nari Labs tag b12431-nariv0](https://github.com/nari-labs/nari-llama.cpp/tree/b12431-nariv0) | `9d5c990a28410e8d8e7228da2d5727c6a653dca3` | Release originale verificata |
| [llama.cpp upstream](https://github.com/ggml-org/llama.cpp/commit/8bf2eae708e5bdaa782bbae9a8a8ceda97acaf1c) | `8bf2eae708e5bdaa782bbae9a8a8ceda97acaf1c` | Confronto della cronologia |

`git merge-base` restituisce `a946eb541a6d3051496bdcc451a0fa85772b8dfd` per Nari Labs e upstream. La cronologia del fork Filters83 ha una radice autonoma, `38f3b1150aa0e0d0b0778c72cc7b74fb272d38cf`, il cui messaggio identifica lo snapshot upstream `f95b0d95394d5e311ba8228689972843178c5e28` più modifiche MTP. `f95b0d9` è antenato di Nari Labs e del master upstream verificato. Il confronto del contenuto usa questi riferimenti perché Git non trova un antenato comune con la cronologia dello snapshot di Filters83.

Il fork Filters83 conserva il percorso AMD proprietario per Flash Attention, le correzioni dei crash LLPC con array di matrici cooperative, il tiling e lo split-K AMD, MMVQ a basso bit, il replay Vulkan e le modifiche DeltaNet e speculative decoding. L'importazione selettiva di NariQ ha richiesto di riconciliare gli enum, il grafo Qwen e il loader. Partire da Nari Labs avrebbe richiesto di riportare e riqualificare queste modifiche sul nuovo albero upstream.

| Modifica Nari Labs | Commit | Integrazione |
| --- | --- | --- |
| Tipi e operatori ggml, validazione, helper di decodifica e metadati Python | `9d40e3d2d` | Importati mantenendo gli ID GGUF 44–49 e la numerazione degli operatori esistenti |
| Riferimento CPU, decoder SIMD e scheduling | `0bc7d59e8` | Importato mantenendo gli operatori ricorrenti del fork |
| libtrellis (Apache-2.0), prodotti CUDA, conversioni e fusioni NariQ | `0fc1e9ae8`, `93fbb0a73` | Consultati come riferimento algoritmico |
| Kernel Metal per prodotti, Hadamard e permutazioni | `c739bb19e` | Consultati come riferimento algoritmico |
| Caricamento, tensori ausiliari, embedding e LM head in bande | `d42cfda6d` | Importato preservando la gestione Qwen/DeltaNet del fork |
| Converter per checkpoint `native_trellis` | `c5ffb70c5` | Analizzato per verificare il layout e le permutazioni del GGUF pubblicato |
| Test del formato, operatori e confronto del modello | `ca246f7e3` | Importati e adattati alle API del fork |
| CI, release, documentazione, launcher Linux/macOS/Windows e test di packaging | `ff7194ace`–`8e0e016e0` | Esaminati per provenienza e qualificazione. La build Windows di questo branch usa il toolchain Vulkan del fork |

La ricerca nei repository pubblici ha trovato il [port HIP sperimentale per RX 7900 XTX](https://github.com/nari-labs/nari-llama.cpp/pull/3), validato dai suoi autori su Linux/ROCm con un errore aperto in un caso sintetico TRL1. Dalla ricerca su GitHub e upstream non è emerso un port Vulkan NariQ da riusare. I sorgenti CUDA e Metal di Nari Labs si possono consultare nelle rispettive revisioni senza introdurre le loro dipendenze nella build Vulkan.

## Formato e integrazione

Il [GGUF NariQ 3.0 bpw](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF/blob/17886395feb8a752b5ca5f02ad0c25543950915e/Qwen3.8-27B-NariQ-3.0bpw.gguf) misura 10.123.626.592 byte e dichiara architettura `qwen35`, contesto di training 262.144 e 16 bande dell'output. Contiene tipi TRL2, TRL3, TRL4, TRL5 e TRL6. L'embedding è TRL2 e occupa 317.849.600 byte. Tutte le descrizioni dei pesi 3-bit esaminati usano identità o Hadamard a blocchi. I test sintetici coprono anche i casi a slice intera.

| File verificato | SHA-256 |
| --- | --- |
| `Qwen3.8-27B-NariQ-3.0bpw.gguf` | `1eaad934aa05579b8eba47ed9b26450392f9875fc2354efe8cfa8b29be8f4419` |
| `Qwen3.8-27B-NariQ-mmproj-BF16.gguf` | `8832425f593068d3278e08eda4c1591072c7a2d8dc90f21e4f86394e4b2eb002` |

Vedi la [specifica del formato](nariq/FORMAT.md) per la disposizione delle tile 16×16 e per i tensori F16, F32 e I32 di scala e permutazione. `GGML_OP_TRELLIS` esegue input gather, rotazione, prodotto con decodifica nella tile, rotazione inversa, scala e output gather. `GGML_OP_TRELLIS_GET_ROWS` estrae solo le righe selezionate e gestisce le scritture per banda. I kernel Vulkan usano i buffer GGUF originali e due aree riutilizzabili per le attivazioni. Il codice mantiene i pesi in forma compressa durante il prodotto.

L'implementazione NariQ importata è copyright Nari Labs e usa la licenza MIT del progetto. Il formato delle tile e la griglia riprendono [exllamav3](https://github.com/turboderp-org/exllamav3), [licenza MIT](licenses/LICENSE-exllamav3).

Il riferimento CPU decompone le finestre di stato a 16 bit e riusa le rotazioni FP32, con un decoder AVX2 sul Ryzen 7 5800X. CUDA usa libtrellis per preparare l'input e per i prodotti e aggiunge fusioni specializzate per il decode. Metal separa il gather e le rotazioni dal prodotto e ha percorsi distinti per simdgroup GEMV e matrici 8×8. Il port Vulkan riusa la disposizione delle tile e l'ordine delle trasformazioni e seleziona i kernel in base alle capacità Vulkan esposte dal driver. [Sorgenti Nari Labs](https://github.com/nari-labs/nari-llama.cpp/tree/8e0e016e0cd13f153380bde2bec01add08eae85e/ggml/src)

## Validazione numerica

Ambiente verificato. Windows x64, Radeon RX 9070, Adrenalin 26.11.2, driver Vulkan 2.0.416 LLPC, MSVC 19.51 e SDK Vulkan 1.4.363.0.

- `test-trellis` verifica TRL1–6, prodotti, input con stride, rotazioni anche Paley, gather, embedding e batch. La suite include un batch da 129 token per attraversare il limite dello scratch e un grafo con embedding in Vulkan.
- `test-trellis-metadata` copre descrittori e metadati malformati, permutazioni, embedding tied, pesi expert e vincoli MTP.
- `test-backend-ops test -b Vulkan0 -o TRELLIS,TRELLIS_GET_ROWS`, 416/416 superati. Le catture dei kernel comprendono percorso F32 e riscontri separati con il layer Khronos.
- GGUF completo, confronto F32 CPU/Vulkan con Flash Attention disattivata. 12 posizioni, 248.320 logits per posizione, 12/12 top-1, stessi token greedy, KL massima 1,91×10⁻⁸ ed errore assoluto massimo 0,00126. Risposta in entrambi i casi ` Paris.\nThe capital of Germany is`.
- GGUF completo, confronto CPU/Vulkan con cooperative matrix e residuo FP16. 31 posizioni, 31/31 top-1, stessi token greedy, KL massima 1,12×10⁻⁷ ed errore assoluto massimo 0,00445. Il prototipo con un solo termine FP16 falliva un confronto sintetico TRL1. La versione con due termini e accumulo F32 supera il confronto senza modificare la soglia del test.
- Con `-ngl 999` il log del modello riporta 65/65 layer e un buffer modello da 9.644,01 MiB su Vulkan0, embedding compreso. Le assegnazioni del grafo hanno collocato gli operatori Trellis del modello su Vulkan0. L'host conserva le pagine mappate del GGUF durante il caricamento.

## Benchmark sulla RX 9070

`llama-bench` con `-p 256 -n 32 -r 3 -b 512 -ub 256 -ctk q4_0 -ctv q4_0 -fa on -ngl 999 -dev Vulkan0`. La prima implementazione usa due ripetizioni. I picchi WDDM misurano la memoria GPU dedicata del processo a intervalli di 0,5 secondi. La RAM è il picco del working set Windows e include memoria mappata.

| Modello e kernel | Prefill tok/s | Decode tok/s | Picco VRAM GiB | Picco RAM GiB |
| --- | ---: | ---: | ---: | ---: |
| NariQ 3.0 bpw, prima implementazione F32 | 78,82 | 7,56 | 10,50 | 9,69 |
| NariQ 3.0 bpw, decode ottimizzato e prefill F32 | 78,37 | 14,74 | 10,50 | 9,66 |
| NariQ 3.0 bpw, decode ottimizzato e prefill cooperative matrix | 143,78 | 15,01 | 10,50 | 9,66 |
| Unsloth `Qwen3.8-27B-UD-Q2_K_XL.gguf`, modello installato | 1.108,37 | 38,87 | 9,04 | 9,02 |

Il decode ottimizzato riusa due parole contigue per quattro stati e quattro partizioni K per gruppo. Il prefill cooperative matrix usa quattro sottogruppi, wave32 sulla Radeon e riuso dei pesi decodificati via LDS. Le matrici dei pesi usano il valore intero esatto e l'attivazione è divisa in un termine FP16 e nel suo residuo scalato. Una tile K=64 ha ridotto il prefill a 51,92 tok/s. La build mantiene K=16. Il modello NariQ del confronto ha misurato 5,33 s di caricamento nell'acquisizione logits con mmap.

Perplexity con `llama-perplexity -c 256 -b 256 -ub 128 --chunks 1 -fa off -t 8`. La prova usa una finestra WikiText-2 identica per tutti. [Corpus PyTorch con revisione fissata](https://github.com/pytorch/examples/blob/d5678bc8ac0cdd79dbd5e44d4130271018bcec4e/word_language_model/data/wikitext-2/test.txt)

| Modello e backend | PPL |
| --- | ---: |
| NariQ CPU | 8,8386 |
| NariQ Vulkan F32 | 8,8384 |
| NariQ Vulkan cooperative matrix | 8,8387 |
| Unsloth UD-Q2_K_XL Vulkan | 7,7513 |

## Contesto e cache KV

Avvio di `llama-server` con 1 slot, `-b 512 -ub 256`, `--fit off`, Flash Attention, offload completo e una richiesta 5+4 token. I valori sono i picchi misurati in GiB.

| Contesto allocato | Q4 dedicata | Q4 condivisa | Q8 dedicata | Q8 condivisa |
| ---: | ---: | ---: | ---: | ---: |
| 8.192 | 9,88 | 0,01 | 9,99 | 0,01 |
| 32.768 | 10,31 | 0,01 | 10,80 | 0,01 |
| 65.536 | 10,89 | 0,01 | 11,89 | 0,01 |
| 131.072 | 12,05 | 0,01 | 14,05 | 0,01 |
| 262.144 | 13,61 | 0,76 | 14,19 | 4,18 |

Verifica con prompt sintetico effettivo e Flash Attention attiva.

| Contesto | Prompt effettivo | Prefill tok/s | Decode tok/s | VRAM dedicata GiB |
| ---: | ---: | ---: | ---: | ---: |
| 8.192 | 8.125 | 140,37 | 10,25 | 9,94 |
| 32.768 | 32.700 | 137,05 | 13,32 | 10,47 |
| 65.536 | 65.470 | 132,91 | 13,03 | 12,08 |

La richiesta da 65.470 token è terminata con 4 token generati dopo 492,60 s di prefill. Una build CPU separata era attiva durante parte di questa misura. A 128K e 256K sono stati verificati allocazione e risposta breve. A 256K WDDM ha riportato memoria GPU condivisa con entrambe le cache. Il branch di partenza espone `q4_0` e `q8_0` e non include TurboQuant/Turbo3. Un eventuale port deve assegnare ID separati e qualificare `SET_ROWS` e Flash Attention. Il [fork TheTom](https://github.com/TheTom/llama-cpp-turboquant/blob/feature/turboquant-kv-cache/docs/KV-cache-quantization.md) assegna a Turbo3 l'ID 44, usato dal GGUF NariQ per TRL1.

## Funzioni di llama-server

- Thinking verificato via API OpenAI con `enable_thinking=true`. La risposta ha fornito `reasoning_content` separato e risultato `132`.
- Tool calling verificato con `tool_choice=required` e un tool di prova. La risposta ha restituito `get_weather` con `{"city":"Rome"}`.
- DFlash2 verificato con il draft Q4_K_M installato, 32 token greedy, 23 proposte accettate su 36. Il draft condivide l'embedding Trellis e l'output in bande. Il limite di vocabolario si applica ai logits dopo l'output Trellis. La prova ha dato lo stesso testo senza draft, a 10,49 tok/s con DFlash2 e 14,73 tok/s senza draft. Il percorso DFlash2 con Unsloth ha accettato 24 proposte su 28.
- Speculative decoding `ngram-simple` verificato su una sequenza ripetitiva con 36 proposte accettate su 36.
- Web UI dal sorgente verificata con risposta HTTP 200 e bundle JavaScript servito tramite gzip.
- Proiettore BF16 NariQ su CPU verificato via API. Il modello ha identificato correttamente un'immagine rossa. Il log multimodale ha emesso avvisi sulle posizioni dei token durante l'elaborazione dell'immagine e la richiesta è terminata con successo.
- Venti richieste nello stesso contesto Q4 da 32K, 16 token generati per richiesta. Con la cache prompt dell'host attiva, il working set è cresciuto da 10.059 a 15.762 MiB e la VRAM è rimasta quasi costante. Con `--cache-ram 0` e lo stesso carico, working set 10.108–10.115 MiB, VRAM dedicata 10.558–10.571 MiB. Il limite predefinito della cache prompt su RAM è 8.192 MiB.

## Build e avvio

`scripts\build-vulkan-windows.cmd` compila la build Windows x64 nativa con MSVC e il Vulkan SDK, senza CUDA, HIP o WSL. Impostare `LLAMA_BUILD_UI=ON` per compilare gli asset della Web UI via Node.js. La build del pacchetto usa gli asset generati dal sorgente con il lockfile npm del fork. Il server resta compatibile con l'API OpenAI quando si sceglie una build solo API.

Esempio PowerShell dalla cartella del progetto. Il percorso cooperativo è attivato esplicitamente. I dispositivi che non espongono il formato cooperativo FP16 con accumulo F32 usano il prodotto F32.

```powershell
$env:GGML_VK_TRELLIS_COOPMAT = '1'
.\build\bin\llama-server.exe -m '.\models\Qwen3.8-27B-NariQ-3.0bpw.gguf' `
  -dev Vulkan0 -ngl 999 -fa on -ctk q4_0 -ctv q4_0 `
  -c 8192 -b 512 -ub 256 -np 1 --fit off --cache-ram 0 --jinja --host 127.0.0.1 --port 18080
```

Aggiungere `--mmproj '.\models\Qwen3.8-27B-NariQ-mmproj-BF16.gguf' --no-mmproj-offload` per usare il proiettore su CPU. Per OpenClaw il confronto corrente favorisce la build Unsloth installata sul throughput. I risultati di qualità qui coprono una finestra di perplexity e richieste brevi.

Con la cache Q4, `-c 32768` e `-c 65536` sono stati verificati anche con prompt vicini alla capienza. Per contesti più grandi, controllare la memoria GPU condivisa in Windows.

Nell'archivio Windows `llama-server.exe` è accanto alle DLL. Sostituire `./build/bin/llama-server.exe` con `./llama-server.exe` nel comando di avvio e indicare il percorso del GGUF scaricato a parte. La build completa include anche `llama-bench.exe`, `llama-perplexity.exe` e gli eseguibili di test.

Il pacchetto usa il loader Vulkan del driver AMD e il runtime Visual C++ x64. `dumpbin` mostra `vulkan-1.dll`, `MSVCP140.dll`, `VCRUNTIME140*.dll` e `VCOMP140.dll` tra le dipendenze dinamiche. [Runtime Visual C++ x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version)

Lo script imposta AVX2, FMA e F16C esplicitamente per il Ryzen 7 5800X. La sonda automatica MSVC nella configurazione Release aveva segnalato AVX-512, seguita da un'istruzione non supportata in `test-trellis`.

## Profiling degli operatori Trellis

Prove locali sulla RX 9070 con lo stesso GGUF 3.0 bpw, offload Vulkan, Flash Attention, cache Q4 e prefill cooperativo. Il profiler usa `GGML_VK_PERF_LOGGER=1` e `GGML_VK_TRELLIS_PROFILE=1` con il logger concorrente disattivato. I timestamp comprendono le attese fra dispatch. I benchmark A/B qui sotto sono eseguiti con il profiler disattivato.

Il dispatch in [`ggml-vulkan-trellis.cpp`](ggml/src/ggml-vulkan/ggml-vulkan-trellis.cpp) prepara le attivazioni con scala e permutazione, applica la Hadamard, lancia il prodotto e applica le trasformazioni d'uscita del tensore. La [decodifica](ggml/src/ggml-vulkan/vulkan-shaders/trellis_decode.glsl) estrae i valori dalle finestre di stato mentre il [kernel di prodotto](ggml/src/ggml-vulkan/vulkan-shaders/trellis_product.comp) lavora sulle tile. Le query Vulkan marcano inizio e fine dei passaggi. Il log `Trellis GPU us` somma gli intervalli di tutte le proiezioni del grafo. La voce `product` include decodifica, preparazione delle tile e moltiplicazione. Per differenziare il costo di banda, registri, LDS e barriere servono contatori GPU.

| Fase | Tempo GPU totale | Prodotto Trellis | Rotazione d'ingresso |
| --- | ---: | ---: | ---: |
| Prefill 256 token | 1.790 ms | 1.594 ms (89,0%) | 118 ms (6,6%) |
| Decode, mediana per token | 73,7 ms | 32,6 ms (44,3%) | 34,8 ms (47,2%) |

Una prova ha fuso la preparazione dell'ingresso e la rotazione dei blocchi Hadamard da 128 valori. Ha superato la parità numerica. A/B con tre ripetizioni ha misurato 14,73 e 15,11 tok/s in decode. Invertendo l'ordine ha misurato 15,08 e 15,11 tok/s. Il prefill è rimasto fra 144 e 146 tok/s. Il guadagno di decode è passato da circa 2,6% a 0,2% fra i due confronti. La fusione è stata rimossa dal candidato locale.

Il decoder locale allinea due parole compresse una volta per gruppo di quattro stati. Sia il prodotto vettoriale sia quello cooperativo riusano la stessa finestra allineata. Benchmark con prefill da 256 token, decode da 32 token e tre ripetizioni per misura.

| Confronto | Prefill prima / dopo (tok/s) | Decode prima / dopo (tok/s) |
| --- | ---: | ---: |
| Prima coppia | 144,09 / 150,60 | 15,22 / 17,49 |
| Ordine invertito | 143,91 / 150,94 | 15,19 / 17,29 |
| Media | 144,00 / 150,77 (+4,7%) | 15,21 / 17,39 (+14,4%) |

Superati 4/4 CTest e 416/416 casi Vulkan per Trellis e lookup delle righe. I 31 vettori di logits del GGUF reale coincidono bit per bit con la precedente cattura Vulkan. Il confronto col riferimento CPU dà 31/31 top-1 e KL massima 1,12×10⁻⁷.

`llama-server` ha completato una richiesta con un prompt breve e 12 token generati a 17,44 tok/s, con contesto configurato a 8.192 token, cache Q4 e 65/65 layer su Vulkan.

## Riproduzione delle prove

Da un nuovo checkout Windows x64 del branch, usare `scripts\build-vulkan-windows.cmd` con MSVC e Vulkan SDK. Lo script crea gli eseguibili in `build/bin`. Nel checkout locale qui sotto si usa la build già pronta in `build/packaging-x64/bin`. Sostituire `$build` con `'.\build'` dopo una compilazione tramite script. Eseguire dalla radice del repository, con il GGUF 3.0 bpw disponibile in `models`.

```powershell
$build = '.\build\packaging-x64'
$bin = Join-Path $build 'bin'
$model = '.\models\Qwen3.8-27B-NariQ-3.0bpw.gguf'
$env:GGML_VK_TRELLIS_COOPMAT = $null
$env:GGML_VK_PERF_LOGGER = $null
$env:GGML_VK_TRELLIS_PROFILE = $null
$env:GGML_VK_PERF_LOGGER_CONCURRENT = $null

ctest --test-dir $build --output-on-failure -R '^test-trellis(-vulkan(-coopmat)?|-metadata)?$'
& "$bin\test-backend-ops.exe" test -b Vulkan0 -o TRELLIS,TRELLIS_GET_ROWS
$env:GGML_VK_TRELLIS_COOPMAT = '1'
& "$bin\llama-bench.exe" -m $model -p 256 -n 32 -r 3 -b 512 -ub 256 `
  -fa on -ctk q4_0 -ctv q4_0 -ngl 999 -dev Vulkan0 -o json
```

Per il confronto numerico, catturare i logits CPU e Vulkan sugli stessi token, usando la sequenza CPU anche sulla GPU. Il test manuale imposta Flash Attention su OFF per la parità. Il confronto di prestazioni sopra la usa su ON.

```powershell
$prompt = 'The capital of France is Paris. Berlin is the capital of Germany. Madrid is the capital of Spain. The capital of Italy is'
$argsModello = @('--model', $model, '--prompt', $prompt, '--ubatch', '32', '--predict', '6', '--threads', '8')
$cpu = Join-Path $build 'handoff-cpu'
$gpu = Join-Path $build 'handoff-vulkan'
& "$bin\test-trellis-model.exe" @argsModello --gpu-layers 0 --output $cpu
& "$bin\test-trellis-model.exe" @argsModello --gpu-layers 999 --force "$cpu.json" --output $gpu
python .\nariq\compare_logits.py $cpu $gpu
```

Per una cattura diagnostica, impostare `$env:GGML_VK_PERF_LOGGER = '1'` e `$env:GGML_VK_TRELLIS_PROFILE = '1'` e rilanciare il benchmark raccogliendo `stderr`. Lasciare `GGML_VK_PERF_LOGGER_CONCURRENT` disattivata. Escludere il riscaldamento quando si confrontano i tempi per grafo. Rimuovere le due variabili prima di ripetere le misure di throughput.

Per il server, usare l'esempio in [Build e avvio](#build-e-avvio) con il binario della stessa build. Verificare nel log `file type = NariQ Trellis`, `offloaded 65/65 layers to GPU` e il buffer modello su `Vulkan0`. In una prova a 8K o 32K registrare il numero effettivo di token elaborati, il tempo al primo token, la velocità di generazione e la VRAM dedicata e condivisa, con lo stesso stato della cache prompt fra le build.

## Priorità di ottimizzazione

1. **Prefill.** In `TRELLIS_COOPMAT`, gruppi che producono righe diverse ricalcolano oggi gli stessi valori `act_hi` e `act_lo`. Provare a prepararli una volta per proiezione e microbatch, dopo la rotazione, riusandoli dal prodotto. Conservare la scala e il residuo esistenti con accumulo FP32. Misurare se il lavoro risparmiato supera il costo del dispatch e del buffer aggiuntivi. La variante con un solo termine FP16 fallisce un caso sintetico TRL1.
2. **Generazione.** Misurare di nuovo il decoder aggiornato e raccogliere contatori per il prodotto vettoriale e per la Hadamard a blocchi da 128. Le priorità da verificare sono il costo delle barriere, il traffico LDS e l'uso delle wave32. La prova che fonde solo preparazione e rotazione ha dato un guadagno quasi nullo nel secondo A/B.
3. **Accettazione.** Ripetere i test sintetici, confrontare il GGUF reale col riferimento CPU e misurare A-B-B-A con il profiler spento. Passare poi a prompt reali da 8K e 32K. Per confrontare Unsloth, usare lo stesso eseguibile e le stesse impostazioni del server e misurare anche una richiesta tipica di OpenClaw. Il confronto pubblicato sopra usa build diverse e la prova di perplexity copre una sola finestra.

## Lavoro aperto

- Estendere il benchmark con prompt reali da 128K in su, misure di stabilità più lunghe e un corpus di qualità più ampio.
- Valutare l'importazione separata di TurboQuant/Turbo3.
