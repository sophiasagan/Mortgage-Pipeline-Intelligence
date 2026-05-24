import React from 'react';
import { createRoot } from 'react-dom/client';
import PipelineDashboard from './pages/PipelineDashboard';

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <PipelineDashboard apiBaseUrl="/api" />
  </React.StrictMode>
);
