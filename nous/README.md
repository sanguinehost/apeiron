# Nous: an OpenAI-compatible provider for apeiron

Nous is the AI-serving face of apeiron. The engine stays a library with no listener; this
separate crate is the local adapter that standard AI clients already know how to call.

It speaks the OpenAI Chat Completions shape over loopback:

- `GET /v1/models` and `GET /v1/models/{id}`
- `POST /v1/chat/completions` (streaming and non-streaming)
- `GET /v1/health`, `GET /health`, `GET /`
- `OPTIONS` for browser clients

The provider is deliberately small: one model, one request at a time, greedy decoding. That is
enough for OpenCode, the OpenAI SDKs, Continue, LibreChat, and a `curl` check.

## Build and run

```sh
cargo build --release --manifest-path nous/Cargo.toml

./nous/target/release/nous-serve \
  --model /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --fold /path/to/exact.fold \
  --tokenizer /path/to/tokenizer.json \
  --name qwen2.5-7b-e8-exact \
  --ctx 2048 \
  --kernel exact
```

`--fold` is optional; without it the GGUF is served dense. `--kernel exact` is the default and
the quality-correct path. `--kernel q8act` is much faster and quality-broken; use it only for
plumbing. The server refuses a non-loopback `--addr`. There is no authentication and no TLS.

## OpenCode

OpenCode v2 uses the `providers` key. Merge `nous/opencode.json` into the project's
`opencode.json` (or the global OpenCode config):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "providers": {
    "nous": {
      "name": "Nous",
      "package": "aisdk:@ai-sdk/openai-compatible",
      "settings": {
        "baseURL": "http://127.0.0.1:8095/v1",
        "apiKey": "local"
      },
      "models": {
        "qwen2.5-7b-e8-exact": {
          "name": "Qwen2.5-7B E8 exact",
          "limit": { "context": 4096, "output": 512 }
        }
      }
    }
  },
  "compaction": { "auto": false }
}
```

The model id in OpenCode is the `--name` passed to `nous-serve`. Select it with `/models` in
OpenCode, or set `"model": "nous/qwen2.5-7b-e8-exact"`:

```sh
opencode run --standalone --auto --model nous/qwen2.5-7b-e8-exact "Say hello in one sentence."
```

The example disables automatic compaction
because the local context is bounded; if you keep compaction on, set its `buffer` below the model's
`limit.context` (the local config here uses 848576). `limit.context` must not exceed the `--ctx`
passed to `nous-serve`; the example matches the default 4096.

## OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8095/v1", api_key="local")
r = client.chat.completions.create(
    model="qwen2.5-7b-e8-exact",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=32,
)
print(r.choices[0].message.content)
```

The same shape works with the Node SDK (`new OpenAI({ baseURL, apiKey })`), Continue
(`provider: "openai"`, `apiBase: "http://127.0.0.1:8095/v1"`), and any client that accepts a
custom OpenAI-compatible base URL.

## curl

```sh
curl -s http://127.0.0.1:8095/v1/models
curl -s http://127.0.0.1:8095/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-7b-e8-exact","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":32}'
```

Set `"stream": true` for server-sent events.

## Tests

```sh
cargo test --manifest-path nous/Cargo.toml --locked
cargo clippy --manifest-path nous/Cargo.toml --all-targets --locked -- -D warnings
cargo fmt --manifest-path nous/Cargo.toml -- --check
```

The tests cover ChatML construction, the OpenAI model card, and the provider name.

## Limits

- Greedy decoding only. `temperature`, `top_p`, `tools`, and string `stop` are ignored.
- One request at a time; the engine call is blocking.
- ChatML prompt template; a base model not trained on it will not behave like an instruct model.
- No authentication, TLS, multi-model routing, or token accounting beyond the response usage.
- The exact kernel is slow on CPU; the fast Q8Act kernel is not a quality path.
- `nous` is a local provider crate, not a hosted service or a registry entry.
- The provider serves the model and kernel you launch; it does not make the E8 pack match Q4. The
  measured exact 7B pack is 21.7069 PPL against 7.4719 for Q4, so treat it as a research path.
