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
  } catch (e) {
    console.error(e);
    renderSkills("resumeSkills", []);
    renderSkills("jdSkills", []);
    renderSkills("missingSkills", []);
    renderSkills("matchingSkills", []);
    alert("Request failed. Ensure backend is running at http://127.0.0.1:8000.");
  }
}

document.getElementById("uploadBtn").addEventListener("click", uploadFiles);
