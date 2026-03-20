function PhaseHeader({ phase }) {
  const label =
    phase === "beginner" ? "Beginner Phase" : phase === "intermediate" ? "Intermediate Phase" : "Advanced Phase";
  return (
    <div className="mb-2 text-sm font-bold text-slate-100">
      {label}
    </div>
  );
}

export default function Roadmap({ roadmap }) {
  const phases = ["beginner", "intermediate", "advanced"];
  const byPhase = {};
  for (const p of phases) byPhase[p] = [];
  for (const step of roadmap || []) {
    if (!byPhase[step.phase]) byPhase[step.phase] = [];
    byPhase[step.phase].push(step);
  }

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
      <div className="mb-3 text-sm font-bold text-slate-100">Adaptive Learning Roadmap</div>

      {(roadmap || []).length === 0 ? (
        <div className="text-sm text-slate-400">Roadmap will appear after analysis.</div>
      ) : (
        <div className="flex flex-col gap-4">
          {phases.map((phase) =>
            (byPhase[phase] || []).length > 0 ? (
              <div key={phase}>
                <PhaseHeader phase={phase} />
                <div className="flex flex-col gap-2">
                  {byPhase[phase].map((step) => (
                    <div key={step.skill_id} className="rounded-md border border-slate-800 bg-slate-900 p-3">
                      <div className="text-sm font-semibold text-slate-100">
                        {step.step_index + 1}. {step.skill_name}
                      </div>
                      <div className="mt-1 text-xs text-slate-400">
                        Prerequisites: {(step.prerequisites || []).length ? step.prerequisites.join(", ") : "None"}
                      </div>
                      {step.notes ? <div className="mt-1 text-xs text-indigo-300">{step.notes}</div> : null}
                      {(step.resources || []).length ? (
                        <div className="mt-2 flex flex-col gap-1">
                          <div className="text-xs font-semibold text-slate-200">Recommended resources</div>
                          {(step.resources || []).map((r) => (
                            <a
                              key={r.link + r.title}
                              href={r.link}
                              target="_blank"
                              rel="noreferrer"
                              className="text-xs text-emerald-300 hover:text-emerald-200 underline"
                            >
                              {r.title} ({r.difficulty}, {r.estimated_time})
                            </a>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              </div>
            ) : null
          )}
        </div>
      )}
    </div>
  );
}

