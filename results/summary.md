# Results summary

Needle retrieval accuracy, % correct [95% Wilson CI]. `wrong` = answered with a *different* needle's code.

| method | budget | 4K acc | 4K wrong |
|---|---|---|---|
| full | 100% | 100 [89–100] | 0 |
| topk_dynamic | 2.5% | 100 [89–100] | 0 |
| topk_dynamic | 5% | 100 [89–100] | 0 |
| topk_dynamic | 10% | 100 [89–100] | 0 |
| topk_dynamic | 20% | 100 [89–100] | 0 |
| snapkv_question_aware | 2.5% | 33 [19–51] | 0 |
| snapkv_question_aware | 5% | 77 [59–88] | 0 |
| snapkv_question_aware | 10% | 97 [83–99] | 0 |
| snapkv_question_aware | 20% | 100 [89–100] | 0 |
| snapkv | 2.5% | 13 [5–30] | 7 |
| snapkv | 5% | 27 [14–44] | 0 |
| snapkv | 10% | 30 [17–48] | 0 |
| snapkv | 20% | 30 [17–48] | 0 |
| streaming_llm | 2.5% | 7 [2–21] | 23 |
| streaming_llm | 5% | 17 [7–34] | 23 |
| streaming_llm | 10% | 27 [14–44] | 37 |
| streaming_llm | 20% | 30 [17–48] | 40 |
| random | 2.5% | 0 [0–11] | 0 |
| random | 5% | 0 [0–11] | 0 |
| random | 10% | 0 [0–11] | 0 |
| random | 20% | 0 [0–11] | 0 |
| kv_int4 | 31.25% bits | 100 [89–100] | 0 |
| kv_int2 | 18.75% bits | 90 [74–97] | 0 |

## Systems (Qwen3-0.6B, fp16, PyTorch MPS on an 8 GB M2)

| context | prefill tok/s | KV cache MB | decode ms/token: full | evicted to 10% | top-k 10% (oracle) |
|---|---|---|---|---|---|
| 1000 | 457 | 109 | 95 | 97 | 129 |
| 2000 | 422 | 219 | 195 | 101 | 279 |
| 4000 | 303 | 438 | 311 | 140 | 792 |
| 8000 | 95 | 875 | 622 | 172 | 894 |
