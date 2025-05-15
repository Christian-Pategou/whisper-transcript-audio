import re

def escape_markdown(text):
    """Échappe les caractères spéciaux de WhatsApp Markdown."""
    special_chars = r"[*_~`]"  # Caractères spéciaux
    return re.sub(f"([{special_chars}])", r"\\\1", text)

def format_whatsapp_message(text):
    """Corrige le formatage pour WhatsApp et échappe les caractères spéciaux."""
    text = escape_markdown(text)  # Échapper les caractères spéciaux
    text = text.replace("\n", "\n\n")  # Ajouter un saut de ligne double
    return text

def format_whatsapp_markdown(text):
    """Convertit le Markdown en un format compatible WhatsApp."""
    
    # Convertir les titres Markdown en gras avec séparateurs
    text = re.sub(r"^# (.+)$", r"📢 *\1*\n", text, flags=re.MULTILINE)  # Titre de niveau 1
    text = re.sub(r"^## (.+)$", r"🔹 *\1*\n", text, flags=re.MULTILINE)  # Titre de niveau 2
    text = re.sub(r"^### (.+)$", r"▪️ *\1*\n", text, flags=re.MULTILINE)  # Titre de niveau 3

    # Remplacement des caractères spéciaux Markdown
    text = text.replace("**", "*")  # Gras
    text = text.replace("__", "_")  # Italique

    # Ajouter des sauts de ligne pour éviter le texte collé
    text = text.replace("\n", " "*20)
    text = text.replace("\n\n", " "*40)

    return text