import React from 'react';

function summary(message: string): string {
  if (/structure.page.detection|vision.*detection failed/i.test(message)) return 'Page analysis failed. The document is still available.';
  if (/connection error|network error|econnrefused|cannot reach/i.test(message)) return 'Could not reach the service. Check your connection and try again.';
  if (/timeout|timed out/i.test(message)) return 'The request took too long. Check the job before starting it again.';
  const firstLine = message.split('\n')[0];
  return firstLine.length <= 160 ? firstLine : 'The request could not be completed.';
}

export default function ErrorNotice({ message, onClose }: { message: string; onClose: () => void }) {
  const title = summary(message);
  return (
    <div role="alert" className="error-notice">
      <div className="error-notice__content">
        <strong>{title}</strong>
        {title !== message && (
          <details><summary>Details</summary><pre>{message}</pre></details>
        )}
      </div>
      <button type="button" className="secondary" onClick={onClose} aria-label="Dismiss error">Close</button>
    </div>
  );
}
