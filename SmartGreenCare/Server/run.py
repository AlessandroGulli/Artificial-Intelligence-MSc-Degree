from web import app
from web.db import db
from web.scheduler import my_job, scheduler


if __name__ == '__main__':
    db.create_all()
    #app.run(debug=True,use_reloader=False)
    scheduler.add_job(func=my_job, args=['job run'], trigger='interval', id='job', seconds=5)
    scheduler.start()
    port = 5000
    interface = '0.0.0.0'
    app.run(host=interface,port=port,use_reloader=False)