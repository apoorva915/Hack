import axios from "axios";

const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: backendUrl,
  timeout: 120000,
});

export async function uploadResumeAndJD({ resumeFile, jobDescriptionText, jobDescriptionFile }) {
  const formData = new FormData();
  formData.append("resume_file", resumeFile);

  if (jobDescriptionText && jobDescriptionText.trim().length > 0) {
    formData.append("job_description_text", jobDescriptionText);
  }
  if (jobDescriptionFile) {
    formData.append("job_description_file", jobDescriptionFile);
  }

  const res = await api.post("/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function analyze({ analysisId }) {
  const res = await api.post("/analyze", { analysis_id: analysisId });
  return res.data;
}

export async function getRoadmap({ analysisId }) {
  const res = await api.get("/roadmap", { params: { analysis_id: analysisId } });
  return res.data;
}

