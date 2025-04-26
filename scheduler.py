from db import fetch_links_from_db
from apscheduler.schedulers.background import BackgroundScheduler
from logger import logger


scheduler = BackgroundScheduler()

def start_scheduler():
    fetch_links_from_db()  # Chargement initial
    scheduler.add_job(fetch_links_from_db, 'interval', hours=24) # Scheduler qui recharge toutes les 12h( minutes=2 )
    scheduler.start()
    logger.info("Scheduler lancé (toutes les 24H heures.)")
