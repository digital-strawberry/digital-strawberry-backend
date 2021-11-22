# How to deploy
1. `git clone https://github.com/digital-strawberry/digital-strawberry-backend.git`
2. `cd digital-strawberry-backend`
3. `python3.9 -m venv venv`
4. `source venv/bin/activate`
5. start redis on port 6379 (or change in `app/worker/config.py`) e.g. with docker `docker run -d -p 6379:6379 redis`
6. `screen -dmS worker python worker.py`
7. `python main.py`