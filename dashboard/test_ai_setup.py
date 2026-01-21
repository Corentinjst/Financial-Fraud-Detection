"""
Script de test pour vérifier que la fonctionnalité d'explication IA est correctement installée
"""

def test_openai_import():
    """Teste si OpenAI peut être importé"""
    try:
        from openai import OpenAI
        print("OpenAI correctement installé")
        return True
    except ImportError:
        print("OpenAI n'est pas installé")
        print("   Installez-le avec: pip install openai")
        return False

def test_api_key():
    """Teste si la clé API est configurée"""
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"Clé API OpenAI trouvée (commence par: {api_key[:10]}...)")
        return True
    else:
        print("Clé API OpenAI non trouvée dans les variables d'environnement")
        print("   Vous pourrez la saisir directement dans l'interface Streamlit")
        return False

def test_api_connection():
    """Teste la connexion à l'API OpenAI"""
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Test de connexion ignoré (pas de clé API)")
        return False
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Test simple
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Dis juste 'OK'"}],
            max_tokens=5
        )
        
        print("Connexion API OpenAI fonctionnelle")
        return True
    except Exception as e:
        print(f"Erreur de connexion API: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("Test de la fonctionnalité d'explication IA")
    print("=" * 60)
    print()

    from dotenv import load_dotenv
    load_dotenv()  
    
    # Tests
    openai_ok = test_openai_import()
    print()
    
    if openai_ok:
        api_key_ok = test_api_key()
        print()
        
        if api_key_ok:
            test_api_connection()
            print()
    
    print("=" * 60)
    print("Résumé")
    print("=" * 60)
    
    if openai_ok:
        print("La fonctionnalité est prête à être utilisée")
        print()
        print("Pour lancer le dashboard:")
        print("   cd dashboard")
        print("   streamlit run app.py")
    else:
        print("Installation requise:")
        print("   pip install openai")

if __name__ == "__main__":
    main()
