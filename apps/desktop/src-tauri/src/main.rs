#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::Manager;

struct BackendProcess(Mutex<Option<Child>>);

fn resolve_python_executable() -> PathBuf {
    // Prefer local workspace virtualenv if present, otherwise fall back to python3/python.
    let candidates = [
        PathBuf::from("../../.venv/bin/python"),
        PathBuf::from("../../.venv/Scripts/python.exe"),
    ];
    for candidate in candidates {
        if candidate.is_file() {
            return candidate;
        }
    }
    if cfg!(target_os = "windows") {
        PathBuf::from("python")
    } else {
        PathBuf::from("python3")
    }
}

fn wait_for_backend(port: u16, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(200));
    }
    false
}

fn spawn_backend(port: u16) -> Result<Child, String> {
    let python = resolve_python_executable();
    Command::new(python)
        .args([
            "-m",
            "uvicorn",
            "backend.app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
        ])
        .env("FACEVISION_EXECUTION_MODE", "local_desktop")
        .env("FACEVISION_PORT", port.to_string())
        .spawn()
        .map_err(|e| format!("Failed to start FaceVision Python backend: {e}"))
}

fn main() {
    let port: u16 = 8000;
    let backend_child = match spawn_backend(port) {
        Ok(child) => {
            if !wait_for_backend(port, Duration::from_secs(15)) {
                eprintln!("Warning: FaceVision backend did not respond on 127.0.0.1:{port} within 15s.");
            }
            Some(child)
        }
        Err(err) => {
            eprintln!("{err}");
            None
        }
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(BackendProcess(Mutex::new(backend_child)))
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.state::<BackendProcess>();
                if let Ok(mut guard) = state.0.lock() {
                    if let Some(mut child) = guard.take() {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running FaceVision desktop application");
}
