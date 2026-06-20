# Waste Detection Web App

A full-stack web application for detecting waste categories (Plastic, Paper, Metal, Glass, Organic) using Machine Learning.

## Features
- **Frontend**: React-based UI for real-time waste classification via webcam.
- **Backend**: FastAPI server for predictions and model management.
- **Manual Training**: Train your own custom model using the terminal.

## Quick Start

### 1. Start the Backend
```bash
cd backend
.\venv\Scripts\activate
uvicorn main:app --reload
```
Runs on `http://localhost:8000`.

### 2. Start the Frontend
```bash
cd frontend
npm run dev
```
Runs on `http://localhost:5173`.

## How to Train Your Model
We support manual training via the terminal for maximum control.

1.  **Add Data**: Place images in `backend/dataset/{category}/`.
2.  **Run Training**:
    ```bash
    cd backend
    python train.py
    ```
3.  **Reload**: Restart the backend server to use the new model.

