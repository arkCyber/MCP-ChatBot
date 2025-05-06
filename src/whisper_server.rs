//! Whisper Server Module: Voice recognition using whisper-rs
//!
//! This module provides functionality for:
//! - Recording audio from microphone
//! - Converting speech to text using whisper-rs
//! - Managing audio streams and buffers
//!
//! Author: arkSong <arksong2018@gmail.com>
//! Version: 1.0.0
//! License: MIT

use crate::utils::print_recording_animation;
use anyhow::{Error, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{error, info};
use ringbuf::ring_buffer::RbBase;
use ringbuf::{HeapRb, Rb};
use std::sync::Arc;
use tokio::sync::Mutex;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct WhisperServer {
    context: Arc<Mutex<WhisperContext>>,
    is_recording: Arc<Mutex<bool>>,
    audio_buffer: Arc<Mutex<HeapRb<f32>>>,
}

impl WhisperServer {
    pub fn new() -> Result<Self> {
        // Initialize whisper context
        let params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params("models/ggml-base.en.bin", params)
            .map_err(|e| Error::msg(format!("Failed to load whisper model: {}", e)))?;

        // Create audio buffer with 30 seconds of audio at 16kHz
        let buffer = HeapRb::new(16000 * 30);

        Ok(Self {
            context: Arc::new(Mutex::new(ctx)),
            is_recording: Arc::new(Mutex::new(false)),
            audio_buffer: Arc::new(Mutex::new(buffer)),
        })
    }

    pub async fn start_recording(&self) -> Result<()> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| Error::msg("No input device available"))?;

        let config = device.default_input_config()?;
        let _sample_rate = config.sample_rate().0 as u32;
        let _channels = config.channels() as u32;

        let is_recording = Arc::clone(&self.is_recording);
        let audio_buffer = Arc::clone(&self.audio_buffer);

        // Start recording animation
        let stop_signal = Arc::new(Mutex::new(false));
        let stop_signal_clone = Arc::clone(&stop_signal);
        let _animation_handle = tokio::spawn(async move {
            print_recording_animation(stop_signal_clone).await;
        });

        // Start recording
        *is_recording.lock().await = true;
        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut buffer = audio_buffer.blocking_lock();
                for &sample in data {
                    if buffer.is_full() {
                        buffer.pop();
                    }
                    buffer.push(sample);
                }
            },
            move |err| {
                error!("Error in audio stream: {}", err);
            },
            None,
        )?;

        stream.play()?;
        Ok(())
    }

    pub async fn stop_recording(&self) -> Result<String> {
        *self.is_recording.lock().await = false;

        // Get audio data from buffer
        let audio_data = {
            let buffer = self.audio_buffer.lock().await;
            buffer.iter().copied().collect::<Vec<f32>>()
        };

        // Process audio with whisper
        let ctx = self.context.lock().await;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        let mut state = ctx.create_state()?;
        state.full(params, &audio_data)?;

        let num_segments = state.full_n_segments()?;
        let mut text = String::new();
        for i in 0..num_segments {
            if let Ok(segment) = state.full_get_segment_text(i) {
                text.push_str(&segment);
                text.push(' ');
            }
        }

        Ok(text.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_whisper_server_initialization() {
        let server = WhisperServer::new();
        assert!(server.is_ok());
    }
}
 