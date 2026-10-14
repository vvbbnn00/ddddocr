use serde_json::{json, Value};
use std::sync::LazyLock;

pub fn tools() -> &'static Value {
    static TOOLS: LazyLock<Value> =
        LazyLock::new(|| serde_json::from_str(include_str!("../list.json")).expect("tool schemas"));
    &TOOLS
}

/// Generate HTTP documentation from the same request/response schemas used by MCP.
pub fn document() -> &'static Value {
    static DOCUMENT: LazyLock<Value> = LazyLock::new(|| {
        let mut paths = serde_json::Map::new();
        for tool in tools()["tools"].as_array().unwrap() {
            let name = tool["name"].as_str().unwrap();
            let pair = name.starts_with("slide_");
            let mut fields = tool["inputSchema"]["properties"]
                .as_object()
                .unwrap()
                .clone();
            let mut parameters = vec![
                json!({"in":"query","name":"beta","schema":{"type":"boolean"},"description":"Select common.onnx; defaults to the startup model."}),
                json!({"in":"query","name":"old","schema":{"type":"boolean"},"description":"Force common_old.onnx."}),
                json!({"in":"query","name":"use_gpu","schema":{"type":"boolean","default":false},"description":"CPU server; true is rejected."}),
                json!({"in":"query","name":"device_id","schema":{"type":"integer","minimum":0}}),
                json!({"in":"query","name":"show_ad","schema":{"type":"boolean","default":false},"description":"Advertising stays disabled."}),
            ];
            for (field, schema) in fields.iter_mut() {
                if field.contains("image") {
                    *schema = json!({"type":"string","format":"binary"});
                } else {
                    if !pair {
                        parameters.push(json!({"in":"query","name":field,"schema":{"type":"string"},"description":"Raw-body option; encode arrays and objects as JSON."}));
                    }
                    *schema = json!({"type":"string","description":"Boolean text, a plain string, or JSON for arrays and objects."});
                }
            }
            let required = if pair {
                vec!["target_image", "background_image"]
            } else {
                fields.remove("image");
                fields.insert("file".into(), json!({"type":"string","format":"binary"}));
                vec!["file"]
            };
            let mut content = json!({
                "application/json":{"schema":tool["inputSchema"]},
                "multipart/form-data":{"schema":{"type":"object","properties":fields,"required":required}}
            });
            if !pair {
                content["application/octet-stream"] =
                    json!({"schema":{"type":"string","format":"binary"}});
            }
            let mut responses = json!({"200":{"description":"Recognition result and elapsed seconds.","content":{"application/json":{"schema":tool["outputSchema"]}}}});
            for (code, description) in [
                (400, "Invalid input"),
                (408, "Upload timeout"),
                (413, "Body or image too large"),
                (415, "Unsupported content type"),
                (503, "Server busy; Retry-After: 1"),
            ] {
                responses[code.to_string()] = json!({"description":description,"content":{"application/json":{"schema":{"type":"object","required":["detail"],"properties":{"detail":{"type":"string"}}}}}});
            }
            let operation = json!({"summary":tool["description"],"parameters":parameters,"requestBody":{"required":true,"content":content},"responses":responses});
            for suffix in if pair {
                vec!["", "/file"]
            } else {
                vec!["", "/file", "/binary"]
            } {
                paths.insert(format!("/{name}{suffix}"), json!({"post":operation}));
            }
        }
        for name in ["health", "status", "config"] {
            paths.insert(format!("/{name}"), json!({"get":{"summary":name,"responses":{"200":{"description":"Service information","content":{"application/json":{"schema":{"type":"object"}}}}}}}));
        }
        paths.insert("/set_charset_range".into(), json!({"post":{
            "summary":"Set the selected model's default charset. An empty array clears it; prefer per-request charset_range for independent clients.",
            "parameters":[{"in":"query","name":"beta","schema":{"type":"boolean"}},{"in":"query","name":"old","schema":{"type":"boolean"}}],
            "requestBody":{"required":true,"content":{"application/json":{"schema":{"type":"object","required":["charset_range"],"properties":{"charset_range":tools()["tools"][0]["inputSchema"]["properties"]["charset_range"]}}}}},
            "responses":{"200":{"description":"Default charset updated"}}
        }}));
        paths.insert("/mcp".into(), json!({"post":{"summary":"Stateless MCP JSON-RPC; enable with --mcp.","requestBody":{"content":{"application/json":{"schema":{"type":"object"}}}},"responses":{"200":{"description":"JSON-RPC result"},"202":{"description":"Notification accepted"}}}}));
        json!({"openapi":"3.0.3","info":{"title":"ddddocr API","version":env!("CARGO_PKG_VERSION"),"description":"All inference responses contain result and processing_time. Images: 8 MiB, longest side 4096 px. Advertising disabled."},"paths":paths})
    });
    &DOCUMENT
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn http_and_mcp_share_response_contracts() {
        for tool in tools()["tools"].as_array().unwrap() {
            let path = format!("/{}", tool["name"].as_str().unwrap());
            assert_eq!(
                document()["paths"][path]["post"]["responses"]["200"]["content"]
                    ["application/json"]["schema"],
                tool["outputSchema"]
            );
            assert_eq!(
                tool["outputSchema"]["required"],
                json!(["result", "processing_time"])
            );
        }
        assert!(document()["paths"]["/ocr/binary"].is_object());
    }
}
