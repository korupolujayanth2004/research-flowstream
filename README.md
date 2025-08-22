# Project README
# 1. Clone & setup
git clone https://github.com/korupolujayanth2004/research-flowstream.git

cd multi_agent_research_assistant

python3 -m venv .venv && source .venv/bin/activate

# 2. Install requirements
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# 3. Setup .env
nano .env
# (Add GROQ_API_KEY, QDRANT keys, BACKEND_URL)

# 4. Run backend
uvicorn backend.main:app --reload --port 8000

# 5. Run frontend
streamlit run frontend/streamlit_app.py
