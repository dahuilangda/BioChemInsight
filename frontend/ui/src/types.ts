export interface UploadPDFResponse {
  pdf_id: string;
  filename: string;
  total_pages: number;
}

export interface TaskStatus {
  task_id: string;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'canceled';
  progress: number;
  message: string;
  pdf_id?: string;
  filename?: string | null;
  result_path?: string;
  error?: string | null;
  params: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  queue_position?: number | null;
}

export interface TaskListResponse {
  tasks: TaskStatus[];
  revision: string;
  running_count: number;
  pending_count: number;
  max_concurrent_tasks: number;
  structure_task_concurrency: number;
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
  search: string;
  status_filter: string;
  type_filter: string;
  date_from: string;
  date_to: string;
  sort_by: string;
  sort_dir: 'asc' | 'desc';
}

export interface TaskListParams {
  page?: number;
  page_size?: number;
  search?: string;
  status?: string;
  task_type?: string;
  date_from?: string;
  date_to?: string;
  sort_by?: string;
  sort_dir?: 'asc' | 'desc';
}

export interface StructureTaskRequest {
  pdf_id: string;
  pages?: string;
  page_numbers?: number[];
  auto_detect_pages?: boolean;
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
  markush_relationships?: Record<string, unknown>[];
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
