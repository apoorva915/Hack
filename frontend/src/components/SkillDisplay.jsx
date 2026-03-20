function SkillPill({ label, sublabel }) {
  return (
    <div className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1">
      <div className="text-sm font-semibold text-slate-100">{label}</div>
      {sublabel ? <div className="text-xs text-slate-400">{sublabel}</div> : null}
    </div>
  );
}

export default function SkillDisplay({ title, extractedSkills }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
      <div className="mb-3 text-sm font-bold text-slate-100">{title}</div>
      <div className="flex flex-wrap gap-2">
        {(extractedSkills?.skills || []).map((s) => (
          <SkillPill key={s.skill_id} label={`${s.name}`} sublabel={`Level: ${s.level} | Confidence: ${s.confidence.toFixed(2)}`} />
        ))}
      </div>
      {(extractedSkills?.skills || []).length === 0 ? (
        <div className="mt-3 text-sm text-slate-400">No skills extracted yet.</div>
      ) : null}
    </div>
  );
}

