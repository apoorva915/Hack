const API_URL = "http://127.0.0.1:8000/analyze/resume-jd";

function renderSkills(containerId, skills) {
  const el = document.getElementById(containerId);
  el.innerHTML = "";

  const items = Array.isArray(skills) ? skills : [];
  if (items.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No skills";
    el.appendChild(li);
    return;
  }

  items.forEach((skill) => {
    const li = document.createElement("li");
    li.textContent = skill;
    el.appendChild(li);
  });
}

function renderEstimatedTime(hours) {
  const el = document.getElementById("estimatedLearningTime");
  const value = Number.isFinite(hours) ? hours : 0;
  el.textContent = `Estimated Learning Time: ${value} hours`;
}

function renderPhases(phases) {
  const container = document.getElementById("phaseContainer");
  container.innerHTML = "";

  const phaseKeys = ["Phase 1", "Phase 2", "Phase 3"];
  phaseKeys.forEach((phaseName) => {
    const skills = phases?.[phaseName] || [];
    const column = document.createElement("div");
    column.className = "phase-column";

    const title = document.createElement("h3");
    title.textContent = phaseName;
    column.appendChild(title);

    const ul = document.createElement("ul");
    ul.className = "skill-list";
    if (skills.length === 0) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "No skills";
      ul.appendChild(li);
    } else {
      skills.forEach((skill) => {
        const li = document.createElement("li");
        li.textContent = skill;
        ul.appendChild(li);
      });
    }

    column.appendChild(ul);
    container.appendChild(column);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderGraph(phases, graph) {
  const canvas = document.getElementById("graphCanvas");
  canvas.innerHTML = "";

  const phaseOrder = ["Phase 1", "Phase 2", "Phase 3"];
  const phaseSkillMap = {};
  phaseOrder.forEach((phaseName) => {
    phaseSkillMap[phaseName] = phases?.[phaseName] || [];
  });

  const allSkills = phaseOrder.flatMap((phaseName) => phaseSkillMap[phaseName]);
  if (allSkills.length === 0) {
    canvas.innerHTML = '<p class="graph-empty">No graph data to display.</p>';
    return;
  }

  const xByPhase = {
    "Phase 1": 160,
    "Phase 2": 520,
    "Phase 3": 880,
  };

  const nodePositions = {};
  const nodesHtml = [];

  phaseOrder.forEach((phaseName) => {
    const skills = phaseSkillMap[phaseName];
    skills.forEach((skill, index) => {
      const y = 80 + index * 70;
      nodePositions[skill] = { x: xByPhase[phaseName], y };
      const phaseClass = phaseName.toLowerCase().replace(/\s+/g, "-");
      nodesHtml.push(
        `<div class="graph-node ${phaseClass}" style="left:${xByPhase[phaseName]}px; top:${y}px;" title="${escapeHtml(skill)}">${escapeHtml(skill)}</div>`,
      );
    });
  });

  const edges = Array.isArray(graph?.edges) ? graph.edges : [];
  const paths = [];
  const edgeLabels = [];
  edges.forEach((edge) => {
    const source = edge?.source;
    const target = edge?.target;
    if (!source || !target || !nodePositions[source] || !nodePositions[target]) {
      return;
    }

    const startX = nodePositions[source].x + 95;
    const startY = nodePositions[source].y + 18;
    const endX = nodePositions[target].x + 95;
    const endY = nodePositions[target].y + 18;
    const controlOffset = Math.max(60, Math.abs(endX - startX) / 2);
    const c1x = startX + controlOffset;
    const c1y = startY;
    const c2x = endX - controlOffset;
    const c2y = endY;
    paths.push(`<path d="M ${startX} ${startY} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${endX} ${endY}" />`);

    const labelX = (startX + endX) / 2;
    const labelY = (startY + endY) / 2 - 8;
    edgeLabels.push(`
      <g class="edge-label-group">
        <rect x="${labelX - 95}" y="${labelY - 11}" width="190" height="20" rx="4" ry="4"></rect>
        <text x="${labelX}" y="${labelY + 3}" text-anchor="middle">${escapeHtml(source)} -> ${escapeHtml(target)}</text>
      </g>
    `);
  });

  const maxRows = Math.max(phaseSkillMap["Phase 1"].length, phaseSkillMap["Phase 2"].length, phaseSkillMap["Phase 3"].length);
  const height = Math.max(280, 120 + maxRows * 70);
  const edgeCount = paths.length;

  canvas.innerHTML = `
    <div class="graph-legend">
      <span class="legend-item"><i class="legend-dot phase-1"></i> Phase 1</span>
      <span class="legend-item"><i class="legend-dot phase-2"></i> Phase 2</span>
      <span class="legend-item"><i class="legend-dot phase-3"></i> Phase 3</span>
      <span class="legend-divider"></span>
      <span class="legend-item"><strong>Edges:</strong> ${edgeCount}</span>
      <span class="legend-item graph-note">arrow: prerequisite -> skill</span>
    </div>
    <div class="graph-columns">
      <span>Phase 1</span>
      <span>Phase 2</span>
      <span>Phase 3</span>
    </div>
    <svg class="graph-svg" viewBox="0 0 1080 ${height}" preserveAspectRatio="xMinYMin meet">
      <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7"></polygon>
        </marker>
      </defs>
      ${paths.join("")}
      ${edgeLabels.join("")}
    </svg>
    <div class="graph-nodes">${nodesHtml.join("")}</div>
  `;
}

async function uploadFiles() {
  const resumeFile = document.getElementById("resumeFile").files[0];
  const jdFile = document.getElementById("jdFile").files[0];

  if (!resumeFile || !jdFile) {
    alert("Please select both Resume and JD PDF files.");
    return;
  }

  const formData = new FormData();
  formData.append("resume_file", resumeFile);
  formData.append("jd_file", jdFile);

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(errorText || "Request failed");
    }

    const data = await res.json();
    renderSkills("resumeSkills", data.resume?.final_skills || []);
    renderSkills("jdSkills", data.jd?.final_skills || []);
    renderSkills("missingSkills", data.gap?.missing_skills || []);
    renderSkills("matchingSkills", data.gap?.matching_skills || []);
    renderSkills("learningPath", data.adaptive_learning?.learning_path || []);
    renderEstimatedTime(data.adaptive_learning?.estimated_learning_time_hours || 0);
    renderPhases(data.adaptive_learning?.phases || {});
    renderGraph(data.adaptive_learning?.phases || {}, data.adaptive_learning?.graph || {});

    const learningSection = document.querySelector(".learning-section");
    if (learningSection) {
      learningSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (e) {
    console.error(e);
    renderSkills("resumeSkills", []);
    renderSkills("jdSkills", []);
    renderSkills("missingSkills", []);
    renderSkills("matchingSkills", []);
    renderSkills("learningPath", []);
    renderEstimatedTime(0);
    renderPhases({});
    renderGraph({}, {});
    alert("Request failed. Ensure backend is running at http://127.0.0.1:8000.");
  }
}

document.getElementById("uploadBtn").addEventListener("click", uploadFiles);
renderSkills("learningPath", []);
renderEstimatedTime(0);
renderPhases({});
renderGraph({}, {});
