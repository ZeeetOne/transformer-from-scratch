/**
 * Mode 1 Prediction API
 *
 * API functions for next word prediction.
 */

import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export interface NextWordRequest {
  input_text: string;
}

export interface NextWordResponse {
  input_text: string;
  predicted_token: string;
  predicted_word: string;
  confidence: number;
  top_predictions: Array<{ token: string; token_id: number; probability: number }>;
  steps: {
    tokenization: { tokens: string[]; token_ids: number[] };
    embeddings: { shape: number[]; sample_values: number[][] };
    attention: { num_heads: number; num_layers: number; attention_shape: number[] };
    feedforward: { hidden_dim: number; output_shape: number[] };
    output: { logits_shape: number[]; softmax_shape: number[] };
  };
  raw_visualization: any;
}

class PredictionAPI {
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
   * Predict next word (Mode 1: GPT-style)
   */
  async predictNextWord(inputText: string): Promise<NextWordResponse> {
    const response = await this.client.post<NextWordResponse>('/predict-next-word', {
      input_text: inputText,
    });
    return response.data;
  }
}

// Export singleton instance
export const predictionAPI = new PredictionAPI();

// Convenience function
export async function predictNextWord(inputText: string): Promise<NextWordResponse> {
  return predictionAPI.predictNextWord(inputText);
}

export default predictionAPI;
