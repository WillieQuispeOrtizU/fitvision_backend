export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed';
export type Mode = 'beginner' | 'pro';

export interface Job {
  job_id: string;
  status: JobStatus;
  mode: Mode;
  input_filename: string;
  error: string | null;
  report_url: string | null;
  original_video_url: string | null;
  annotated_video_url: string | null;
}

export interface AnalysisReport {
  video: {
    input_path: string;
    output_path: string;
    fps?: number;
    width?: number;
    height?: number;
    frame_count?: number;
    duration?: number;
  };
  stats: {
    correct: number;
    incorrect: number;
    total_reps: number;
  };
  reps: RepAnalysis[];
  assistant: {
    enabled?: boolean;
    status?: string;
    overall_summary?: string | null;
    latency_seconds?: number | null;
    model_names?: Record<string, string>;
    errors?: string[];
  };
}

export interface RepAnalysis {
  rep_id: number;
  start_time?: number;
  end_time?: number;
  duration?: number;
  correct: boolean;
  issue_tags: string[];
  worst_issue: string | null;
  angles: Record<string, { min: number; max: number }>;
  key_frames: Record<string, string>;
  assistant_feedback: {
    summary?: string;
    cues?: string[] | string;
    severity?: string;
  } | null;
}
