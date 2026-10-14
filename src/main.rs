mod api;
mod schema;

use clap::{ArgGroup, Parser};
use salvo::prelude::*;
use std::sync::OnceLock;
use tracing::info;

static ARGS: OnceLock<Args> = OnceLock::new();

#[derive(Parser, Debug, Clone)]
#[command(group(ArgGroup::new("ocr_model").args(["ocr", "old"]).multiple(false)),
    group(ArgGroup::new("mcp_mode").args(["mcp", "only_mcp"]).multiple(false)))]
struct Args {
    #[arg(long, default_value = "0.0.0.0:8000")]
    address: String,
    #[arg(long)]
    mcp: bool,
    #[arg(long)]
    only_mcp: bool,
    /// Enable OCR using common.onnx (upstream beta model).
    #[arg(long)]
    ocr: bool,
    /// Enable OCR using common_old.onnx (upstream default model).
    #[arg(long)]
    old: bool,
    #[arg(long)]
    det: bool,
    #[arg(long)]
    slide: bool,
    #[arg(long)]
    ocr_charset_range: Option<String>,
    /// Override the OCR model and its same-name .json charset, including inline builds.
    #[arg(long)]
    ocr_path: Option<std::path::PathBuf>,
    #[arg(long)]
    det_path: Option<std::path::PathBuf>,
    /// Maximum admitted requests, including upload, inference and response streaming. Excess gets 503.
    #[arg(long, default_value_t = default_concurrency(), value_parser = clap::value_parser!(u16).range(1..=256))]
    max_concurrency: u16,
    /// ONNX threads per shared session; 1 avoids oversubscription under concurrent traffic.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..=256))]
    inference_threads: u16,
    /// Maximum complete request body, including multipart or Base64 overhead.
    #[arg(long, default_value_t = 24, value_parser = clap::value_parser!(u16).range(1..=256))]
    max_body_mib: u16,
    /// Deadline for receiving a request body; running inference retains its admission slot.
    #[arg(long, default_value_t = 15, value_parser = clap::value_parser!(u16).range(1..=300))]
    upload_timeout_secs: u16,
    #[arg(long)]
    acme: Option<String>,
}

fn default_concurrency() -> u16 {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8) as u16)
        .unwrap_or(1)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = Args::parse();
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    if !args.ocr && !args.old && !args.det && !args.slide {
        args.old = true;
        args.det = true;
        args.slide = true;
    }
    ARGS.set(args.clone()).unwrap();
    api::initialize(&args)?;
    let service = api::service(&args);
    info!(address = %args.address, max_concurrency = args.max_concurrency,
        inference_threads = args.inference_threads, "ddddocr ready (advertising disabled)");
    let acceptor = TcpListener::new(args.address);
    if let Some(domain) = args.acme {
        Server::new(acceptor.acme().add_domain(domain).bind().await)
            .serve(service)
            .await;
    } else {
        Server::new(acceptor.bind().await).serve(service).await;
    }
    Ok(())
}
