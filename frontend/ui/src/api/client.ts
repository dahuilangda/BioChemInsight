import axios from 'axios';
import type {
  ArtifactResponse,
  AssayResult,
  AssayTaskRequest,
  StructureTaskRequest,
  StructuresResult,
  TaskStatus,
  UploadPDFResponse,
} from '../types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE ?? '/api',
  withCredentials: false,
});

export async function uploadPdf(
  file: File,
  onUploadProgress?: (percent: number) => void,
): Promise<UploadPDFResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post<UploadPDFResponse>('/pdfs', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (event) => {
      if (!event.total) return;
      const percent = Math.round((event.loaded / event.total) * 100);
      if (onUploadProgress) {
        onUploadProgress(Math.max(0, Math.min(100, percent)));
      }
    },
  });
  return response.data;
}

export async function fetchPdfInfo(pdfId: string): Promise<UploadPDFResponse> {
  const response = await api.get<UploadPDFResponse>(`/pdfs/${pdfId}`);
  return response.data;
}

export async function fetchPdfPage(
  pdfId: string,
  page: number,
  options?: { zoom?: number; maxWidth?: number },
): Promise<string> {
  const { zoom = 1.5, maxWidth } = options ?? {};
  const response = await api.get<{ page: number; image: string }>(`/pdfs/${pdfId}/pages/${page}`, {
    params: { zoom, max_width: maxWidth },
  });
  return response.data.image;
}

export async function queueStructureTask(payload: StructureTaskRequest): Promise<TaskStatus> {
  const response = await api.post<TaskStatus>('/tasks/structures', payload);
  return response.data;
}

export async function fetchTask(taskId: string): Promise<TaskStatus> {
  const response = await api.get<TaskStatus>(`/tasks/${taskId}`);
  return response.data;
}

export async function fetchTaskStructures(taskId: string): Promise<StructuresResult> {
  const response = await api.get<StructuresResult>(`/tasks/${taskId}/structures`);
  return response.data;
}

export async function updateTaskStructures(taskId: string, records: StructuresResult['records']): Promise<StructuresResult> {
  const response = await api.put<StructuresResult>(`/tasks/${taskId}/structures`, { records });
  return response.data;
}

export async function queueAssayTask(payload: AssayTaskRequest): Promise<TaskStatus> {
  const response = await api.post<TaskStatus>('/tasks/assays', payload);
  return response.data;
}

export async function fetchTaskAssays(taskId: string): Promise<AssayResult> {
  const response = await api.get<AssayResult>(`/tasks/${taskId}/assays`);
  return response.data;
}

export async function fetchArtifact(path: string): Promise<ArtifactResponse> {
  const response = await api.get<ArtifactResponse>('/artifacts', { params: { path } });
  return response.data;
}

export function getTaskDownloadUrl(taskId: string): string {
  const base = api.defaults.baseURL ?? '/api';
  if (base.endsWith('/')) {
    return `${base}tasks/${taskId}/download`;
  }
  return `${base}/tasks/${taskId}/download`;
}

export const getDownloadUrl = getTaskDownloadUrl;

export async function renderSmiles(
  smiles: string,
  options?: { width?: number; height?: number },
): Promise<string> {
  const response = await api.post<RenderSmilesResponse>('/chem/render', {
    smiles,
    width: options?.width,
    height: options?.height,
  });
  return response.data.image;
}

interface RenderSmilesResponse {
  smiles: string;
  image: string;
}

export interface EditorAtom {
  id: number;
  element: string;
  x: number;
  y: number;
}

export interface EditorBond {
  a1: number;
  a2: number;
  order: number;
}

export interface MoleculeGraph {
  atoms: EditorAtom[];
  bonds: EditorBond[];
}

interface ParseSmilesResponse extends MoleculeGraph {
  smiles: string;
}

interface BuildMoleculeResponse {
  smiles: string;
  image: string;
}

export async function parseSmilesToGraph(smiles: string): Promise<ParseSmilesResponse> {
  const response = await api.post<ParseSmilesResponse>('/chem/parse', { smiles });
  return response.data;
}

export async function buildMoleculeFromGraph(graph: MoleculeGraph): Promise<BuildMoleculeResponse> {
  const response = await api.post<BuildMoleculeResponse>('/chem/build', graph);
  return response.data;
}
