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
  queue_position?: number | null;
}

export interface TaskListResponse {
  tasks: TaskStatus[];
  running_count: number;
  pending_count: number;
  max_concurrent_tasks: number;
  structure_task_concurrency: number;
}

export interface StructureTaskRequest {
  pdf_id: string;
  pages?: string;
  page_numbers?: number[];
  auto_detect_pages?: boolean;
  engine?: string;
  structure_filter_strictness?: 'strict' | 'balanced' | 'permissive';
}

export interface AutoDetectTaskRequest {
  pdf_id: string;
  assay_names?: string[];
  detect_structure_pages?: boolean;
  detect_assay_pages?: boolean;
  detect_assay_names?: boolean;
}

export type StructureRecord = Record<string, string | number | string[] | null>;

export interface StructuresResult {
  task: TaskStatus;
  records: StructureRecord[];
  filtered_records: StructureRecord[];
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
  auto_detect_pages?: boolean;
  auto_detect_assay_names?: boolean;
  lang?: string;
  structure_task_id?: string;
}

export type AssayRecord = Record<string, string | number | string[] | null>;

export interface AssayResult {
  task: TaskStatus;
  records: AssayRecord[];
}

export interface FullPipelineRequest {
  pdf_id: string;
  structure_filter_strictness?: 'strict' | 'balanced' | 'permissive';
  lang?: string;
}
