/**
 * Shared API Client
 *
 * Base API client with common configuration.
 */

import axios, { AxiosInstance } from 'axios';
import { InferenceRequest, InferenceResponse } from './types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export class APIClient {
  protected client: AxiosInstance;

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
   * Health check
   */
  async healthCheck(): Promise<{ status: string }> {
    const response = await this.client.get('/health');
    return response.data;
  }
}

// Export singleton instance
export const apiClient = new APIClient();
export default apiClient;
