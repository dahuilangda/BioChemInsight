import axios from 'axios';
import type {
  ArtifactResponse,
  AutoDetectTaskRequest,
  AssayResult,
  AssayTaskRequest,
  FullPipelineRequest,
  StructureTaskRequest,
  StructuresResult,
  TaskListParams,
  TaskListResponse,
  TaskStatus,
  UploadPDFResponse,
} from '../types';

function stripTrailingSlash(value: string): string {
  const trimmed = value.replace(/\/+$/, '');
  return trimmed.length > 0 ? trimmed : '/';
}

function resolveApiBase(): string {
  const explicit = import.meta.env.VITE_API_BASE;
  if (explicit && explicit.trim().length > 0) {
    return stripTrailingSlash(explicit.trim());
  }

  if (typeof window !== 'undefined') {
    const { protocol, hostname, port } = window.location;
    const configuredPort = import.meta.env.VITE_API_PORT?.toString().trim();
    const hostPrefix = `${protocol}//${hostname}`;

    if (configuredPort) {
      return `${hostPrefix}:${configuredPort}/api`;
    }

    if (port === '3000' || port === '5173') {
      return `${hostPrefix}:8000/api`;
    }

    const portSegment = port ? `:${port}` : '';
    return `${hostPrefix}${portSegment}/api`;
  }

  return '/api';
}

function assertUsableId(value: string, label: string): void {
  if (!value || value === 'undefined' || value === 'null') {
    throw new Error(`${label} is missing`);
  }
}

const api = axios.create({
  baseURL: resolveApiBase(),
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
  assertUsableId(pdfId, 'PDF ID');
  const response = await api.get<UploadPDFResponse>(`/pdfs/${pdfId}`);
  return response.data;
}

export async function fetchPdfPage(
  pdfId: string,
  page: number,
  options?: { zoom?: number; maxWidth?: number },
): Promise<string> {
  assertUsableId(pdfId, 'PDF ID');
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

export async function queueAutoDetectTask(payload: AutoDetectTaskRequest): Promise<TaskStatus> {
  const response = await api.post<TaskStatus>('/tasks/auto-detect', payload);
  return response.data;
}

export async function queueFullPipelineTask(payload: FullPipelineRequest): Promise<TaskStatus> {
  const response = await api.post<TaskStatus>('/tasks/full-pipeline', payload);
  return response.data;
}

export async function fetchTask(taskId: string): Promise<TaskStatus> {
  assertUsableId(taskId, 'Task ID');
  const response = await api.get<TaskStatus>(`/tasks/${taskId}`);
  return response.data;
}

export async function cancelTask(taskId: string): Promise<TaskStatus> {
  assertUsableId(taskId, 'Task ID');
  const response = await api.post<TaskStatus>(`/tasks/${taskId}/cancel`);
  return response.data;
}

export async function fetchTasks(params: TaskListParams = {}): Promise<TaskListResponse> {
  const response = await api.get<TaskListResponse>('/tasks', { params });
  return response.data;
}

export async function fetchTaskStructures(taskId: string): Promise<StructuresResult> {
  assertUsableId(taskId, 'Task ID');
  const response = await api.get<StructuresResult>(`/tasks/${taskId}/structures`);
  return response.data;
}

export async function updateTaskStructures(taskId: string, records: StructuresResult['records']): Promise<StructuresResult> {
  assertUsableId(taskId, 'Task ID');
  const response = await api.put<StructuresResult>(`/tasks/${taskId}/structures`, { records });
  return response.data;
}

export async function queueAssayTask(payload: AssayTaskRequest): Promise<TaskStatus> {
  const response = await api.post<TaskStatus>('/tasks/assays', payload);
  return response.data;
}

export async function fetchTaskAssays(taskId: string): Promise<AssayResult> {
  assertUsableId(taskId, 'Task ID');
  const response = await api.get<AssayResult>(`/tasks/${taskId}/assays`);
  return response.data;
}

export async function fetchArtifact(path: string): Promise<ArtifactResponse> {
  const response = await api.get<ArtifactResponse>('/artifacts', { params: { path } });
  return response.data;
}

export function getTaskDownloadUrl(taskId: string): string {
  assertUsableId(taskId, 'Task ID');
  const base = api.defaults.baseURL ?? '/api';
  if (base.endsWith('/')) {
    return `${base}tasks/${taskId}/download`;
  }
  return `${base}/tasks/${taskId}/download`;
}

export const getDownloadUrl = getTaskDownloadUrl;

export async function renderSmiles(
  smiles: string,
  options?: { width?: number; height?: number; molblock?: string },
): Promise<string> {
  const response = await api.post<RenderSmilesResponse>('/chem/render', {
    smiles,
    width: options?.width,
    height: options?.height,
    molblock: options?.molblock,
  });
  return response.data.image;
}

export interface RenderSmilesBatchItem {
  key: string;
  smiles?: string;
  width?: number;
  height?: number;
  molblock?: string;
}

export interface RenderSmilesBatchResult {
  key: string;
  smiles: string;
  image: string;
  error?: string | null;
}

export async function renderSmilesBatch(items: RenderSmilesBatchItem[]): Promise<RenderSmilesBatchResult[]> {
  if (!items.length) return [];
  const response = await api.post<{ results: RenderSmilesBatchResult[] }>('/chem/render-batch', { items });
  return response.data.results;
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
