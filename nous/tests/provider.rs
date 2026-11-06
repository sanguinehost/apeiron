use serde_json::json;

#[test]
fn chatml_accepts_text_parts_and_maps_developer() {
    let messages = vec![
        json!({"role": "developer", "content": [{"type": "text", "text": "be terse"}]}),
        json!({"role": "user", "content": "hello"}),
    ];
    let s = nous::chatml(&messages).expect("chatml");
    assert!(s.contains("<|im_start|>system\nbe terse<|im_end|>"));
    assert!(s.contains("<|im_start|>user\nhello<|im_end|>"));
    assert!(s.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn chatml_rejects_a_message_without_a_role() {
    assert!(nous::chatml(&[json!({"content": "hello"})]).is_err());
}

#[test]
fn model_card_is_openai_shaped_and_names_nous() {
    let card = nous::model_card("qwen2.5-7b-e8-exact", 4096);
    assert_eq!(card["id"], "qwen2.5-7b-e8-exact");
    assert_eq!(card["object"], "model");
    assert_eq!(card["owned_by"], "nous");
    assert_eq!(card["context_window"], 4096);
    let list = nous::models_json("qwen2.5-7b-e8-exact", 4096);
    assert_eq!(list["object"], "list");
    assert_eq!(list["data"][0]["id"], "qwen2.5-7b-e8-exact");
}

#[test]
fn help_is_a_successful_plain_text_exit() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_nous-serve"))
        .arg("--help")
        .output()
        .expect("run nous-serve --help");
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("OpenAI-compatible loopback provider"));
    assert!(text.contains("--kernel exact|q8act"));
}

#[test]
fn non_loopback_address_is_refused() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_nous-serve"))
        .args([
            "--model",
            "/nonexistent/model.gguf",
            "--tokenizer",
            "/nonexistent/tokenizer.json",
        ])
        .args(["--addr", "0.0.0.0:8095"])
        .output()
        .expect("run nous-serve with a public address");
    assert!(!out.status.success());
    let text = String::from_utf8_lossy(&out.stderr);
    assert!(text.contains("is not loopback"));
}
