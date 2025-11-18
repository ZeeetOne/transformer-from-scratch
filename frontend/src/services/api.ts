/**
 * API Service for Transformer Visualization Backend
 *
 * Provides typed interfaces for all API endpoints.
 */

import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Type definitions
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

class TransformerAPI {
  private client: AxiosInstance;

  constructor(baseURL: string = API_BASE_URL) {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Run transformer inference
   */
  async runInference(request: InferenceRequest): Promise<InferenceResponse> {
    const response = await this.client.post<InferenceResponse>('/inference', request);
    return response.data;
  }

  /**
   * Get attention visualization for specific layer/head
   */
  async getAttentionVisualization(
    sourceText: string,
    layerIdx: number = 0,
    headIdx?: number
  ): Promise<AttentionVisualization> {
    const response = await this.client.post<AttentionVisualization>('/attention', {
      source_text: sourceText,
      layer_idx: layerIdx,
      head_idx: headIdx,
    });
    return response.data;
  }

  /**
   * Get model architecture information
   */
  async getModelInfo(): Promise<ModelInfo> {
    const response = await this.client.get<ModelInfo>('/model/info');
    return response.data;
  }

  /**
   * Get embedding visualization
   */
  async getEmbeddingVisualization(sourceText: string): Promise<any> {
    const response = await this.client.post('/visualize/embeddings', null, {
      params: { source_text: sourceText },
    });
    return response.data;
  }

  /**
   * Get attention flow visualization
   */
  async getAttentionFlow(
    sourceText: string,
    layerIdx: number = 0,
    headIdx: number = 0
  ): Promise<any> {
    const response = await this.client.post('/visualize/flow', null, {
      params: {
        source_text: sourceText,
        layer_idx: layerIdx,
        head_idx: headIdx,
      },
    });
    return response.data;
  }

  /**
   * Get complete visualization data
   */
  async getCompleteVisualization(request: InferenceRequest): Promise<any> {
    const response = await this.client.post('/visualize/complete', request);
    return response.data;
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{ status: string }> {
    const response = await this.client.get('/health');
    return response.data;
  }

  /**
   * Predict next word (Mode 1: GPT-style)
   */
  async predictNextWord(inputText: string): Promise<any> {
    const response = await this.client.post('/predict-next-word', {
      input_text: inputText,
    });
    return response.data;
  }
}

// Export singleton instance
export const api = new TransformerAPI();
export default api;

// Convenience function for Mode 1
export async function predictNextWord(inputText: string): Promise<any> {
  return api.predictNextWord(inputText);
}
