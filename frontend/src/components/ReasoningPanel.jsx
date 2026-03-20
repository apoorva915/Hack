export default function ReasoningPanel({ reasoningTrace }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
      <div className="mb-3 text-sm font-bold text-slate-100">Reasoning Trace (Why each skill was recommended)</div>

      {(reasoningTrace || []).length === 0 ? (
        <div className="text-sm text-slate-400">No reasoning available yet.</div>
      ) : (
        <div className="flex flex-col gap-3">
          {reasoningTrace.map((r) => (
            <div key={r.skill_id} className="rounded-md border border-slate-800 bg-slate-900 p-3">
              <div className="text-sm font-semibold text-slate-100">
                {r.skill_name} <span className="text-xs text-slate-400">(target: {r.target_level})</span>
              </div>
              <div className="mt-2 flex flex-col gap-1">
                {(r.reasons || []).map((reason, idx) => (
                  <div key={idx} className="text-xs text-slate-300">
                    {reason}
                  </div>
                ))}
              </div>
              {r.evidence && Object.keys(r.evidence).length ? (
                <details className="mt-2">
                  <summary className="cursor-pointer text-xs text-indigo-300 hover:text-indigo-200">Evidence (structured)</summary>
                  <pre className="mt-2 overflow-auto rounded border border-slate-700 bg-slate-950 p-2 text-[11px] text-slate-200">
                    {JSON.stringify(r.evidence, null, 2)}
                  </pre>
                </details>
              ) : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

