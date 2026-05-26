import type { AnalysisReport, Job, Mode } from './types';

const API_BASE = import.meta.env.VITE_SQUAT_API_URL ?? 'http://127.0.0.1:8000';

export function assetUrl(path: string | null | undefined): string {
  if (!path) return '';
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

export async function createJob(file: File, mode: Mode): Promise<Job> {
  const form = new FormData();
  form.append('video', file);
  form.append('mode', mode);

  const response = await fetch(`${API_BASE}/api/jobs`, {
    method: 'POST',
    body: form
  });
  return readJson<Job>(response);
}

export async function getJob(jobId: string): Promise<Job> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  return readJson<Job>(response);
}

export async function getReport(jobId: string): Promise<AnalysisReport> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}/report`);
  return readJson<AnalysisReport>(response);
}

async function readJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof payload.detail === 'string' ? payload.detail : response.statusText;
    throw new Error(detail);
  }
  return payload as T;
}
