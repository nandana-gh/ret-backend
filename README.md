# Initial config:
- First clone the repo
- Then go into the folder 
- Then go to terminal and create a venv using :
```bash
python -m venv .venv
```
- Activate the venv by doing:(for windows)
```bash
.venv/Scripts/activate
```

- Then install the dependencies:
```bash
pip install -r requirements.txt
```
# Running the program:
- To run the program just run using uvicorn:
```bash
uvicorn server:app --reload
```
