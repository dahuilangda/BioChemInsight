import React from 'react';
import { loadDocument, retainDocument, releaseDocument } from '../utils/pdfDocument';

// Keep the preload lease until the reader mounts.
export default function PdfPreload({ pdfId }: { pdfId: string }) {
  React.useEffect(() => {
    retainDocument(pdfId);
    void loadDocument(pdfId).catch(() => undefined);
    return () => releaseDocument(pdfId);
  }, [pdfId]);
  return null;
}
