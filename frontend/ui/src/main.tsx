import './polyfills';
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles.css';
import './product-theme.css';
import './review-workspace.css';
import './resizable-split.css';
import './review-grid.css';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(<App />);
