import React from 'react';
import { createRoot } from 'react-dom/client';
import PipelineDashboard from './pages/PipelineDashboard';

// Priority order:
//   1. VITE_API_URL env var (set in Vercel dashboard for direct Railway calls)
//   2. /api           (works locally via vite.config.js proxy,
//                      and in production via vercel.json rewrite)
const apiBaseUrl = import.meta.env.VITE_API_URL ?? '/api';

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <PipelineDashboard apiBaseUrl={apiBaseUrl} />
  </React.StrictMode>
);
