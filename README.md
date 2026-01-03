🚀 Vision AI Object Detector (GroundingDINO – Realtime, CPU)

A realtime object detection system built using GroundingDINO and OpenCV, optimized to run on CPU-only environments without requiring any paid APIs or cloud services.

This project performs prompt-based object detection directly from a webcam feed, allowing users to specify objects in natural language (e.g., "person, dog, mobile phone").

🛠️ Tech Stack:
Python 3.10
GroundingDINO
PyTorch (CPU)
OpenCV
TorchVision
NumPy

📁 Project Structure
vision-ai-object-detector/
│
├── realtime_groundingdino.py      # Main realtime detection script
├── test_camera.py                 # Camera test utility
├── GroundingDINO/                 # GroundingDINO source code
├── weights/
│   └── groundingdino_swint_ogc.pth # Model weights
├── .gitignore
└── README.md

⚡ CPU Optimizations Used
Reduced frame resolution (320×240)
Frame skipping (detect every N frames)
Torch inference with no_grad
CPU-only execution (no CUDA dependency)


🧪 Example Use Cases
Smart surveillance
Assistive vision systems
AI learning projects
Edge AI / low-resource environments
Foundation for FastAPI / LLM / RAG systems



🚧 Future Enhancements (Planned)
✅ FastAPI inference server
✅ Image upload + prompt API
✅ LLM-based prompt generation
✅ RAG for contextual detection
✅ Docker deployment
✅ Frontend dashboard



👨‍💻 Author
Dhruv Khatri
Aspiring Data Scientist | AI & ML Enthusiast


⭐ Acknowledgements
GroundingDINO
PyTorch & OpenCV communities
