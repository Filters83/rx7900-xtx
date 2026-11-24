# NariQ su Vulkan per Windows e RDNA4

Stato al 24 novembre 2026. Branch sperimentale `nariq-vulkan-rdna4`.

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

## Lavoro aperto

- Estendere il benchmark con prompt reali da 128K in su, misure di stabilità più lunghe e un corpus di qualità più ampio.
- Valutare l'importazione separata di TurboQuant/Turbo3.
