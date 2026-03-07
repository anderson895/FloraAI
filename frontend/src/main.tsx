import './index.css'   // ← ADD THIS LINE
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import FlowerClassifier from './FlowerClassifier'

createRoot(document.getElementById('root')!).render(
  <StrictMode><FlowerClassifier /></StrictMode>
)