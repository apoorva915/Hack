import { useMemo, useState } from "react";

export default function UploadForm({ onAnalyze, loading }) {
  const [resumeFile, setResumeFile] = useState(null);
  const [jobDescriptionText, setJobDescriptionText] = useState("");
  const [jobDescriptionFile, setJobDescriptionFile] = useState(null);

  const canSubmit = useMemo(() => {
    if (!resumeFile) return false;
    const hasText = jobDescriptionText && jobDescriptionText.trim().length > 0;
    return hasText || !!jobDescriptionFile;
  }, [resumeFile, jobDescriptionText, jobDescriptionFile]);

  return (
    <form
      className="flex flex-col gap-4"
      onSubmit={(e) => {
        e.preventDefault();
        if (!canSubmit) return;
        onAnalyze({ resumeFile, jobDescriptionText, jobDescriptionFile });
      }}
    >
      <div className="grid grid-cols-1 gap-2">
        <label className="text-sm text-slate-300">Resume (PDF or DOCX)</label>
        <input
          type="file"
          accept=".pdf,.docx,.doc,.txt"
          className="block w-full text-sm file:mr-4 file:rounded file:border-0 file:bg-indigo-500 file:px-4 file:py-2 file:text-white hover:file:bg-indigo-400"
          onChange={(e) => setResumeFile(e.target.files?.[0] || null)}
        />
      </div>

      <div className="grid grid-cols-1 gap-2">
        <label className="text-sm text-slate-300">Job Description (paste text)</label>
        <textarea
          value={jobDescriptionText}
          onChange={(e) => setJobDescriptionText(e.target.value)}
          rows={7}
          className="w-full rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-sm outline-none focus:border-indigo-500"
          placeholder="Paste job description requirements here..."
        />
      </div>

      <div className="grid grid-cols-1 gap-2">
        <label className="text-sm text-slate-300">Or upload job description file</label>
        <input
          type="file"
          accept=".pdf,.docx,.doc,.txt"
          className="block w-full text-sm file:mr-4 file:rounded file:border-0 file:bg-emerald-500 file:px-4 file:py-2 file:text-white hover:file:bg-emerald-400"
          onChange={(e) => setJobDescriptionFile(e.target.files?.[0] || null)}
        />
      </div>

      <button
        type="submit"
        disabled={!canSubmit || loading}
        className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-slate-600"
      >
        {loading ? "Analyzing..." : "Generate Roadmap"}
      </button>
    </form>
  );
}

