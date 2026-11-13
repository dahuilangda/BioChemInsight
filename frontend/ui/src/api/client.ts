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
    const { port } = window.location;

    // Same-origin requests: the 3000 static server proxies /api to the
    // backend and the Vite dev server proxies /api via vite.config.ts. This
    // avoids requiring the browser to reach port 8000 directly.
    if (!port || port === '3000' || port === '5173') {
      return '/api';
    }

    // Custom origins (e.g. serving the bundle elsewhere) keep the previous
    // behavior unless VITE_API_PORT overrides it.
    const { protocol, hostname } = window.location;
    const hostPrefix = `${protocol}//${hostname}`;
    const configuredPort = import.meta.env.VITE_API_PORT?.toString().trim();
    if (configuredPort) {
      return `${hostPrefix}:${configuredPort}/api`;
    }
    return `${hostPrefix}:${port}/api`;
  }

  return '/api';
}

function assertUsableId(value: string, label: string): void {
  if (!value || value === 'undefined' || value === 'null') {
    throw new Error(`${label} is missing`);
  }
}

export const api = axios.create({
  baseURL: resolveApiBase(),
  withCredentials: false,
  timeout: 600000,
});

// Keep the server's actionable error (FastAPI detail / capability error) rather
// than Axios's generic "Request failed with status code …" throughout the UI.
api.interceptors.response.use((response) => response, (error: unknown) => {
  if (axios.isAxiosError(error)) {
    const payload = error.response?.data;
    const detail = payload?.detail ?? payload?.error;
    if (typeof detail === 'string' && detail.trim()) {
      error.message = detail;
    } else if (Array.isArray(detail)) {
      const messages = detail.map((item: { msg?: string }) => item?.msg).filter(Boolean);
      if (messages.length) error.message = messages.join('; ');
    } else if (!error.response) {
      error.message = error.code === 'ECONNABORTED'
        ? 'The request timed out. Check the job before starting it again.'
        : 'Cannot reach the service. Check your connection and try again.';
    }
  }
  return Promise.reject(error);
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

export async function fetchPatentPdf(number: string): Promise<UploadPDFResponse> {
  const response = await api.post<UploadPDFResponse>('/patents/fetch', { number });
  return response.data;
}

export async function fetchPdfInfo(pdfId: string): Promise<UploadPDFResponse> {
  assertUsableId(pdfId, 'PDF ID');
  const response = await api.get<UploadPDFResponse>(`/pdfs/${pdfId}`);
  return response.data;
}

export function getPdfPreviewUrl(pdfId: string, page: number): string {
  return `${api.defaults.baseURL}/pdfs/${encodeURIComponent(pdfId)}/preview/${page}`;
}

export function getPdfFileUrl(pdfId: string): string {
  assertUsableId(pdfId, 'PDF ID');
  return `${(api.defaults.baseURL ?? '/api').replace(/\/$/, '')}/pdfs/${encodeURIComponent(pdfId)}/file`;
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

const structureRevisions = new Map<string, string>();

export async function fetchTaskStructures(taskId: string): Promise<StructuresResult> {
  assertUsableId(taskId, 'Task ID');
  const response = await api.get<StructuresResult>(`/tasks/${taskId}/structures`);
  structureRevisions.set(taskId, String(response.data.task?.params?.records_revision || '0'));
  return response.data;
}

export async function updateTaskStructures(taskId: string, records: StructuresResult['records']): Promise<StructuresResult> {
  assertUsableId(taskId, 'Task ID');
  const response = await api.put<StructuresResult>(`/tasks/${taskId}/structures`, { records, base_revision: structureRevisions.get(taskId) ?? '0' });
  structureRevisions.set(taskId, String(response.data.task?.params?.records_revision || '0'));
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

export interface AnnotationItem {
  row_key: string;
  kind: string;
  compound_id: string;
  smiles: string;
  molblock: string;
  segment_file: string;
  page?: string | number | null;
}

export interface AnnotationRecord {
  row_key: string;
  status: 'confirmed' | 'wrong' | 'corrected';
  original_smiles?: string | null;
  corrected_smiles?: string | null;
  corrected_molblock?: string | null;
  segment_file?: string | null;
  compound_id?: string | null;
  kind?: string | null;
  updated_at?: string | null;
}

export interface AnnotationUpsertPayload {
  row_key: string;
  status: 'confirmed' | 'wrong' | 'corrected';
  corrected_smiles?: string;
  corrected_molblock?: string;
  original_smiles?: string;
  segment_file?: string;
  compound_id?: string;
  kind?: string;
}

export async function fetchAnnotationItems(taskId: string): Promise<{ items: AnnotationItem[]; counts: Record<string, number> }> {
  assertUsableId(taskId, 'Task ID');
  const response = await api.get(`/tasks/${taskId}/annotation-items`);
  return response.data;
}

export async function fetchAnnotations(taskId: string): Promise<Record<string, AnnotationRecord>> {
  assertUsableId(taskId, 'Task ID');
  const response = await api.get(`/tasks/${taskId}/annotations`);
  return response.data.annotations || {};
}

export async function upsertAnnotation(taskId: string, payload: AnnotationUpsertPayload): Promise<Record<string, AnnotationRecord>> {
  assertUsableId(taskId, 'Task ID');
  const response = await api.post(`/tasks/${taskId}/annotations`, payload);
  return response.data.annotations || {};
}

export function getAnnotationsExportUrl(taskId: string): string {
  assertUsableId(taskId, 'Task ID');
  const base = api.defaults.baseURL ?? '/api';
  return `${base.endsWith('/') ? base : `${base}/`}tasks/${taskId}/annotations/export`;
}

const artifactRequests = new Map<string, Promise<ArtifactResponse>>();
const boxArtifacts = new Map<string, Promise<ArtifactResponse>>();
export function fetchArtifact(path: string): Promise<ArtifactResponse> {
  const pending = artifactRequests.get(path);
  if (pending) return pending;
  const request = api.get<ArtifactResponse>('/artifacts', { params: { path } })
    .then((response) => response.data).finally(() => artifactRequests.delete(path));
  artifactRequests.set(path, request);
  return request;
}

// Box coordinates belong to immutable completed-task artifacts. Share their
// requests between crop generation and PDF navigation; keep images in the
// workspace's separate, bounded image cache instead.
export function fetchBoxArtifact(path: string): Promise<ArtifactResponse> {
  const existing = boxArtifacts.get(path);
  if (existing) { boxArtifacts.delete(path); boxArtifacts.set(path, existing); return existing; }
  const request = fetchArtifact(path).catch((error) => { boxArtifacts.delete(path); throw error; });
  boxArtifacts.set(path, request);
  if (boxArtifacts.size > 128) boxArtifacts.delete(boxArtifacts.keys().next().value!);
  return request;
}

export function getTaskDownloadUrl(taskId: string): string {
  assertUsableId(taskId, 'Task ID');
  const base = api.defaults.baseURL ?? '/api';
  if (base.endsWith('/')) {
    return `${base}tasks/${taskId}/download`;
  }
  return `${base}/tasks/${taskId}/download`;
}

export async function fetchTaskDownload(taskId: string): Promise<Blob> {
  assertUsableId(taskId, 'Task ID');
  try {
    const response = await api.get<Blob>(`/tasks/${encodeURIComponent(taskId)}/download`, { responseType: 'blob' });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.data instanceof Blob) {
      try {
        const payload = JSON.parse(await error.response.data.text());
        if (typeof payload.detail === 'string') error.message = payload.detail;
      } catch { /* Preserve the network error if the body is not JSON. */ }
    }
    throw error;
  }
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

export async function prepareJSMEMolblock(molblock: string): Promise<string> {
  const normalized = (molblock || '').trimEnd();
  if (!normalized) return '';
  const response = await api.post<{ molblock: string; changed: boolean }>('/chem/jsme-molblock', {
    molblock: normalized,
  });
  return response.data.molblock || normalized;
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

export async function undoStructureEdit(taskId: string): Promise<StructuresResult> {
  const response = await api.post<StructuresResult>(`/tasks/${taskId}/structures/undo`, { base_revision: structureRevisions.get(taskId) ?? '0' });
  structureRevisions.set(taskId, String(response.data.task?.params?.records_revision || '0'));
  return response.data;
}
