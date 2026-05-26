import { CheckCircle2, Clock3, FileVideo, Loader2, Upload, XCircle } from 'lucide-react';
import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';

import { assetUrl, createJob, getJob, getReport } from './api';
import type { AnalysisReport, Job, Mode, RepAnalysis } from './types';

const pollMs = 2500;

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<Mode>('beginner');
  const [job, setJob] = useState<Job | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const statusText = useMemo(() => {
    if (!job) return 'Waiting for upload';
    if (job.status === 'queued') return 'Queued';
    if (job.status === 'processing') return 'Processing video and assistant feedback';
    if (job.status === 'completed') return 'Completed';
    return 'Failed';
  }, [job]);

  useEffect(() => {
    if (!job || job.status === 'completed' || job.status === 'failed') return;

    const timer = window.setInterval(async () => {
      try {
        const nextJob = await getJob(job.job_id);
        setJob(nextJob);
        if (nextJob.status === 'completed') {
          const nextReport = await getReport(nextJob.job_id);
          setReport(nextReport);
          window.clearInterval(timer);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Could not fetch job status');
        window.clearInterval(timer);
      }
    }, pollMs);

    return () => window.clearInterval(timer);
  }, [job]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setError('Choose a video before processing.');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setReport(null);
    setJob(null);

    try {
      const created = await createJob(file, mode);
      setJob(created);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleFile(nextFile: File | undefined) {
    if (!nextFile) return;
    setFile(nextFile);
    setError(null);
  }

  return (
    <main className="min-h-screen bg-slate-50 text-ink">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-5 py-6 lg:px-8">
        <header className="flex flex-col gap-2 border-b border-line pb-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-normal">Squat Analysis</h1>
            <p className="mt-1 max-w-2xl text-sm text-slate-600">
              Upload a side-view squat video and review MediaPipe metrics with Falcon/Gemma coaching feedback.
            </p>
          </div>
          <div className="rounded border border-line bg-white px-3 py-2 text-sm text-slate-700">{statusText}</div>
        </header>

        <section className="grid gap-6 lg:grid-cols-[380px_1fr]">
          <form className="flex flex-col gap-4" onSubmit={handleSubmit}>
            <div
              className="flex min-h-52 cursor-pointer flex-col items-center justify-center gap-3 rounded-md border border-dashed border-slate-400 bg-white p-5 text-center transition hover:border-slate-700"
              onClick={() => inputRef.current?.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                handleFile(event.dataTransfer.files[0]);
              }}
            >
              <input
                ref={inputRef}
                className="hidden"
                type="file"
                accept="video/mp4,video/quicktime,video/x-msvideo"
                onChange={(event) => handleFile(event.target.files?.[0])}
              />
              <Upload className="h-8 w-8 text-slate-600" aria-hidden="true" />
              <div>
                <p className="text-sm font-medium">{file ? file.name : 'Drop or choose a squat video'}</p>
                <p className="mt-1 text-xs text-slate-500">MP4, MOV, or AVI</p>
              </div>
            </div>

            <div className="grid grid-cols-2 rounded-md border border-line bg-white p-1">
              {(['beginner', 'pro'] as Mode[]).map((option) => (
                <button
                  key={option}
                  type="button"
                  className={`rounded px-3 py-2 text-sm font-medium capitalize transition ${
                    mode === option ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'
                  }`}
                  onClick={() => setMode(option)}
                >
                  {option}
                </button>
              ))}
            </div>

            <button
              className="inline-flex h-11 items-center justify-center gap-2 rounded-md bg-slate-900 px-4 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
              type="submit"
              disabled={isSubmitting || Boolean(job && job.status !== 'failed' && job.status !== 'completed')}
            >
              {isSubmitting || (job && ['queued', 'processing'].includes(job.status)) ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              ) : (
                <FileVideo className="h-4 w-4" aria-hidden="true" />
              )}
              Process video
            </button>

            {job && <JobProgress job={job} />}
            {error && <Message tone="error" text={error} />}
            {job?.status === 'failed' && <Message tone="error" text={job.error ?? 'Processing failed'} />}
          </form>

          <section className="min-h-[520px]">
            {report ? <ResultView report={report} /> : <EmptyResult />}
          </section>
        </section>
      </div>
    </main>
  );
}

function JobProgress({ job }: { job: Job }) {
  const Icon = job.status === 'completed' ? CheckCircle2 : job.status === 'failed' ? XCircle : Clock3;
  return (
    <div className="rounded-md border border-line bg-white p-4">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Icon className="h-4 w-4 text-slate-700" aria-hidden="true" />
        <span className="capitalize">{job.status}</span>
      </div>
      <p className="mt-2 text-xs text-slate-500">Job {job.job_id}</p>
    </div>
  );
}

function EmptyResult() {
  return (
    <div className="flex min-h-[520px] items-center justify-center rounded-md border border-line bg-white">
      <div className="max-w-sm text-center">
        <FileVideo className="mx-auto h-10 w-10 text-slate-500" aria-hidden="true" />
        <p className="mt-3 text-sm font-medium">Results will appear here</p>
        <p className="mt-1 text-sm text-slate-500">The annotated video, counts, and per-rep feedback load after processing.</p>
      </div>
    </div>
  );
}

function ResultView({ report }: { report: AnalysisReport }) {
  return (
    <div className="flex flex-col gap-5">
      <video className="aspect-video w-full rounded-md bg-black" src={assetUrl(report.video.output_path)} controls />

      <div className="grid gap-3 sm:grid-cols-3">
        <Metric label="Total reps" value={report.stats.total_reps} />
        <Metric label="Correct" value={report.stats.correct} />
        <Metric label="Incorrect" value={report.stats.incorrect} />
      </div>

      <section className="rounded-md border border-line bg-white p-5">
        <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
          <h2 className="text-base font-semibold">Assistant Summary</h2>
          <span className="text-xs uppercase tracking-wide text-slate-500">{report.assistant.status ?? 'unknown'}</span>
        </div>
        <p className="mt-3 text-sm leading-6 text-slate-700">
          {report.assistant.overall_summary ?? 'No assistant summary was returned.'}
        </p>
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        {report.reps.map((rep) => (
          <RepCard key={rep.rep_id} rep={rep} />
        ))}
      </section>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-line bg-white p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-2 text-3xl font-semibold">{value}</p>
    </div>
  );
}

function RepCard({ rep }: { rep: RepAnalysis }) {
  const cues = normalizeCues(rep.assistant_feedback?.cues);
  return (
    <article className="grid gap-4 rounded-md border border-line bg-white p-4 sm:grid-cols-[128px_1fr]">
      <img
        className="aspect-square w-full rounded object-cover"
        src={assetUrl(rep.key_frames.bottom)}
        alt={`Rep ${rep.rep_id} bottom position`}
      />
      <div className="min-w-0">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold">Rep {rep.rep_id}</h3>
            <p className="mt-1 text-xs text-slate-500">{formatDuration(rep.duration)} seconds</p>
          </div>
          <span
            className={`rounded px-2 py-1 text-xs font-medium ${
              rep.correct ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'
            }`}
          >
            {rep.correct ? 'Correct' : 'Needs review'}
          </span>
        </div>

        <p className="mt-3 text-sm text-slate-700">{rep.assistant_feedback?.summary ?? 'No rep feedback returned.'}</p>

        {cues.length > 0 && (
          <ul className="mt-3 list-disc space-y-1 pl-4 text-sm text-slate-600">
            {cues.map((cue) => (
              <li key={cue}>{cue}</li>
            ))}
          </ul>
        )}

        <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
          {Object.entries(rep.angles).map(([name, angle]) => (
            <div key={name} className="rounded border border-line bg-panel px-2 py-2">
              <p className="font-medium capitalize">{name}</p>
              <p className="mt-1 text-slate-600">
                {Math.round(angle.min)}-{Math.round(angle.max)}
              </p>
            </div>
          ))}
        </div>
      </div>
    </article>
  );
}

function Message({ tone, text }: { tone: 'error'; text: string }) {
  return <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">{text}</div>;
}

function normalizeCues(cues: string[] | string | undefined): string[] {
  if (!cues) return [];
  return Array.isArray(cues) ? cues : [cues];
}

function formatDuration(value: number | undefined): string {
  return typeof value === 'number' ? value.toFixed(2) : '0.00';
}
