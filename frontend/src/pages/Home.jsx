import { useMemo, useState } from "react";
import UploadForm from "../components/UploadForm.jsx";
import SkillDisplay from "../components/SkillDisplay.jsx";
import Roadmap from "../components/Roadmap.jsx";
import ReasoningPanel from "../components/ReasoningPanel.jsx";
import { analyze, uploadResumeAndJD } from "../services/api.js";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [analysisId, setAnalysisId] = useState(null);
  const [result, setResult] = useState(null);

  const missingCount = useMemo(() => {
    const gaps = result?.skill_gaps || [];
    return gaps.filter((g) => g.is_missing).length;
  }, [result]);

  async function handleAnalyze({ resumeFile, jobDescriptionText, jobDescriptionFile }) {
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const uploadRes = await uploadResumeAndJD({
        resumeFile,
        jobDescriptionText,
        jobDescriptionFile,
      });
      const id = uploadRes.analysis_id;
      setAnalysisId(id);

      const full = await analyze({ analysisId: id });
      setResult(full);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <div>
        <div className="text-2xl font-bold text-slate-50">AI Adaptive Onboarding Engine</div>
        <div className="mt-1 text-sm text-slate-400">
          Upload a resume + job description. Get a prerequisite-aware learning roadmap and a full reasoning trace.
        </div>
      </div>

      <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
        <UploadForm onAnalyze={handleAnalyze} loading={loading} />
        {analysisId ? (
          <div className="mt-3 text-xs text-slate-400">
            Analysis ID: <span className="text-slate-200">{analysisId}</span>
          </div>
        ) : null}
        {error ? <div className="mt-3 text-sm text-rose-300">{error}</div> : null}
      </div>

      {result ? (
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <SkillDisplay title="Resume: extracted skills" extractedSkills={result.resume} />
            <SkillDisplay title="Job Description: extracted requirements" extractedSkills={result.job_description} />
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
            <div className="mb-2 text-sm font-bold text-slate-100">Skill gaps</div>
            <div className="text-sm text-slate-400">
              Missing skills detected: {missingCount}
            </div>
            <div className="mt-3 flex flex-col gap-2">
              {(result.skill_gaps || []).map((g) => (
                <div
                  key={g.jd_skill_id}
                  className={
                    "rounded-md border p-3 text-sm " +
                    (g.is_missing ? "border-rose-900 bg-rose-950/30" : "border-slate-800 bg-slate-900")
                  }
                >
                  <div className="font-semibold text-slate-100">
                    {g.jd_skill_name}{" "}
                    <span className="text-xs text-slate-400">
                      (required: {g.required_level})
                    </span>
                  </div>
                  <div className="mt-1 text-xs text-slate-300">
                    Similarity: {g.similarity.toFixed(2)}{" "}
                    {g.matched_resume_skill_name ? (
                      <> | Resume match: {g.matched_resume_skill_name}</>
                    ) : null}
                  </div>
                  {g.is_missing ? (
                    <div className="mt-1 text-xs text-rose-200">Missing: recommend in roadmap</div>
                  ) : (
                    <div className="mt-1 text-xs text-emerald-200">Partially covered</div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <Roadmap roadmap={result.roadmap} />
          <ReasoningPanel reasoningTrace={result.reasoning_trace} />
        </div>
      ) : null}
    </div>
  );
}

