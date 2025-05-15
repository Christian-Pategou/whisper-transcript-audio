# Initialisation de la base de données
import sqlite3
import time
import pendulum
from logs import logger


def init_db():
    conn = sqlite3.connect("./data/chatbot.db")
    c = conn.cursor()
    
    # Table des utilisateurs
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone TEXT UNIQUE NOT NULL,
                    name TEXT,
                    registered_at REAL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
                    last_active REAL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')) -- Suivi de l'activité
                )''')

    # Table des conversations
    c.execute('''CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    conversation_id TEXT UNIQUE,
                    state TEXT DEFAULT 'active', -- Ajout d'un état de conversation
                    created_at REAL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
                    expires_at REAL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )''')

    # Table des messages
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    sender TEXT CHECK(sender IN ('user', 'bot')),
                    message TEXT NOT NULL,
                    status TEXT CHECK(status IN ('sent', 'delivered', 'read')) DEFAULT 'sent', -- Statut du message
                    created_at REAL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                )''')

    conn.commit()
    conn.close()
    logger.success("Succefful created database")

# creation d'un nouvel utilsateur
def get_or_create_user(phone, name="Unknow"):
    conn = sqlite3.connect("./data/chatbot.db")
    c = conn.cursor()

    # Vérifier si l'utilisateur existe déjà
    c.execute("SELECT id FROM users WHERE phone = ?", (phone,))
    user = c.fetchone()

    if user:
        logger.debug(f"User already exist: *user_id* -->> {user[0]}")
        user_id = user[0]
        # Mise à jour de la dernière activité de l'utilisateur
        c.execute("UPDATE users SET last_active = ? WHERE id = ?", (time.strftime("%Y-%m-%d %H:%M:%S"), user_id))
    else:
        # Créer un nouvel utilisateur
        c.execute("INSERT INTO users (phone, name, last_active) VALUES (?, ?, ?)", (phone, name, time.strftime("%Y-%m-%d %H:%M:%S")))
        user_id = c.lastrowid  # Récupérer l'ID du nouvel utilisateur
        logger.success(f"Sucessfull create user: *user_id* -->> {user_id}")

    conn.commit()
    conn.close()
    return user_id


# Génère un nouvel ID si nécessaire
def get_or_create_conversation_id(user_id):
    new_conversation = False
    """Récupère une conversation existante ou en crée une nouvelle valable 24h"""
    conn = sqlite3.connect("./data/chatbot.db")
    c = conn.cursor()

    current_time = pendulum.now()

    # Vérifier si une conversation existante est toujours valide
    c.execute("SELECT conversation_id FROM conversations WHERE user_id = ? AND expires_at > ?", (user_id, current_time.strftime("%Y-%m-%d %H:%M:%S")))
    conversation = c.fetchone()

    if conversation:
        conversation_id = conversation[0]
        logger.debug(f"Conversaton already exist: *conv_id* -->> {conversation_id}")
    else:
        # Créer une nouvelle conversation
        new_conversation = True
        conversation_id = f"conv_{user_id}_{current_time.strftime("%Y-%m-%d %H:%M:%S").replace("-","_").replace(" ", "_")}"  # Générer un ID unique
        expiration_time = current_time.add(days=1)  # 24 heures
        c.execute("INSERT INTO conversations (user_id, conversation_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                  (user_id, conversation_id, current_time.strftime("%Y-%m-%d %H:%M:%S"), expiration_time.strftime("%Y-%m-%d %H:%M:%S")))
        logger.success(f"Sucessfull create Conversaton : *conv_id* -->> {conversation_id}")
        
    conn.commit()
    conn.close()

    return conversation_id, new_conversation

# sauvegarder les messages
def save_message(conversation_id, sender, message, status="sent"):
    """Enregistre un message avec son statut dans la base de données"""
    conn = sqlite3.connect("./data/chatbot.db")
    c = conn.cursor()

    c.execute("INSERT INTO messages (conversation_id, sender, message, status, created_at) VALUES (?, ?, ?, ?, ?)",
              (conversation_id, sender, message, status, time.strftime("%Y-%m-%d %H:%M:%S")))
    logger.success(f"Sucessfull save message. Sender: {sender} : *conv_id* -->> {conversation_id}")

    conn.commit()
    conn.close()

# mettre a jour le status d'un message
def update_message_status(conversation_id, sender, new_status):
    """Met à jour le statut des messages envoyés"""
    conn = sqlite3.connect("./data/chatbot.db")
    c = conn.cursor()

    c.execute("UPDATE messages SET status = ? WHERE conversation_id = ? AND sender = ?",
              (new_status, conversation_id, sender))
    logger.success(f"Message statut update. New status: {new_status} : *conv_id* -->> {conversation_id}")

    conn.commit()
    conn.close()

# recuperer l'historique des conversations
def get_conversation_history(conversation_id):
    """Retourne l'historique des messages pour une conversation donnée"""
    conn = sqlite3.connect("./data/chatbot.db")
    c = conn.cursor()

    c.execute("SELECT sender, message, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
              (conversation_id,))
    history = c.fetchall()

    conn.close()
    return history


if __name__ == "__main__":
    init_db()