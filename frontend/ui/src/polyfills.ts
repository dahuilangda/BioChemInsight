import process from 'process';
import { Buffer } from 'buffer';

declare global {
  interface Window {
    process?: typeof process;
    Buffer?: typeof Buffer;
  }
}

if (typeof window !== 'undefined') {
  if (!window.process) {
    window.process = process;
  } else {
    // Merge to keep any existing fields while ensuring browser polyfill methods exist
    window.process = Object.assign(window.process, process);
  }
  if (!window.Buffer) {
    window.Buffer = Buffer;
  }
}

export {};
