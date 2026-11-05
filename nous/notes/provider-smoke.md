# Provider smoke receipt

Machine: 4 cores / 8 threads, 6.6 GiB RAM, no GPU. Debug build
(`CARGO_TARGET_DIR=/home/lucas/.cache/apeiron-nous-target cargo build --manifest-path nous/Cargo.toml`),
Qwen2.5-0.5B-Instruct Q8_0, `--kernel q8act`, `--ctx 1024`, under `nice -n 19`.

## HTTP

```sh
nous-serve --model qwen2.5-0.5b-instruct-q8_0.gguf --tokenizer tokenizer.json \
  --addr 127.0.0.1:8096 --name qwen2.5-0.5b-test --ctx 1024 --kernel q8act
```

```text
GET /v1/health  -> {"status":"ok"}
GET /v1/models  -> {"object":"list","data":[{"id":"qwen2.5-0.5b-test",...}]}

POST /v1/chat/completions
{"model":"qwen2.5-0.5b-test","messages":[{"role":"user","content":"Reply with exactly: NOUS OK"}],"max_tokens":16}
-> {"choices":[{"finish_reason":"stop","message":{"content":"NOUS OK","role":"assistant"}}],...}

POST /v1/chat/completions (stream true)
-> data: {"choices":[{"delta":{"content":"Sun"},
   data: {"choices":[{"delta":{"content":"."},
   data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
   data: {"choices":[],"usage":{...}}
   data: [DONE]
```

The debug `q8act` build decoded at about 3.0 s/token in this run. The HTTP receipt is the request
and response shape, not a speed claim.

## OpenCode

OpenCode v2.0.22, `nous/opencode.json` copied to a scratch project with
`compaction.auto = false`, model id `qwen2.5-0.5b-test`, provider base URL
`http://127.0.0.1:8097/v1` for a mock and `http://127.0.0.1:8096/v1` for the real provider.

The mock provider returned `NOUS MOCK OK` from:

```sh
opencode run --standalone --auto --model nous/qwen2.5-0.5b-test \
  'Reply with exactly: NOUS MOCK OK'
```

The real provider received `chat prompt 530 tokens, max_new 494, stream true` through the same
config. The unoptimized local build was still decoding when the 240 s smoke-test timeout closed the
client; no completion was recorded. The mock receipt fixes the OpenCode URL/schema; the real
provider receipt fixes the request shape.
