from app.worker import app

if __name__ == '__main__':
    app.worker_main(['worker', '--loglevel=INFO', '--pool=solo'])
