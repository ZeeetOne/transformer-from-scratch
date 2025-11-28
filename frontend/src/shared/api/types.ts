/**
 * Shared API Type Definitions
 */

export interface InferenceRequest {
  source_text: string;
  target_text?: string;
  generate: boolean;
  max_gen_len?: number;
}

export interface InferenceResponse {
  source_text: string;
  decoded_output: string;
  source_tokens: number[];
  target_tokens: number[];
  mode: string;
  visualization_data: any;
}

export interface AttentionHeatmap {
  head_index: number;
  weights: number[][][]; // [batch, seq, seq]
  entropy: number;
  focus_pattern: number[][];
}

export interface AttentionVisualization {
  layer: number;
  n_heads: number;
  heatmaps: AttentionHeatmap[];
  tokens?: {
    source: number[];
    text: string;
  };
}

export interface ModelInfo {
  architecture: string;
  d_model: number;
  n_encoder_layers: number;
  n_decoder_layers: number;
  vocab_size: number;
  device: string;
  total_parameters: number;
  trainable_parameters: number;
}
