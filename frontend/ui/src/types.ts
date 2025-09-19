export interface UploadPDFResponse {
  pdf_id: string;
  filename: string;
  total_pages: number;
}

export interface TaskStatus {
  task_id: string;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  pdf_id?: string;
  result_path?: string;
  error?: string | null;
  params: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface StructureTaskRequest {
  pdf_id: string;
  pages?: string;
  page_numbers?: number[];
  engine?: string;
}

export type StructureRecord = Record<string, string | number | string[] | null>;

export interface StructuresResult {
  task: TaskStatus;
  records: StructureRecord[];
}

export interface ArtifactResponse {
  path: string;
  content: string;
  mime_type: string;
}

export interface AssayTaskRequest {
  pdf_id: string;
  assay_names: string[];
  pages?: string;
  page_numbers?: number[];
  lang?: string;
  ocr_engine?: string;
}

export type AssayRecord = Record<string, string | number | string[] | null>;

export interface AssayResult {
  task: TaskStatus;
  records: AssayRecord[];
}
