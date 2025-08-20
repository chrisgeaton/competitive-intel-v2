#!/usr/bin/env python3
"""
Security Key Generation Utility
Generates secure random keys for production deployment
"""

import secrets
import os
from pathlib import Path


def generate_secret_key(length: int = 64) -> str:
    """Generate a cryptographically secure random key."""
    return secrets.token_urlsafe(length)


def generate_secure_env():
    """Generate a secure .env file from template."""
    template_path = Path(__file__).parent.parent / ".env.template"
    env_path = Path(__file__).parent.parent / ".env"
    
    if not template_path.exists():
        print("ERROR: .env.template file not found!")
        return
    
    if env_path.exists():
        response = input(".env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Read template
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Generate secure keys
    secret_key = generate_secret_key(64)
    
    # Replace placeholder values
    content = content.replace("SECRET_KEY=", f"SECRET_KEY={secret_key}")
    content = content.replace("ENVIRONMENT=development", "ENVIRONMENT=production")
    
    # Write .env file
    with open(env_path, 'w') as f:
        f.write(content)
    
    print("SUCCESS: Secure .env file generated!")
    print(f"SECRET_KEY: {secret_key[:32]}... (64 characters total)")
    print("\nIMPORTANT SECURITY NOTES:")
    print("1. Update DATABASE_URL with your production database credentials")
    print("2. Set appropriate CORS_ORIGINS for your frontend domains")
    print("3. Configure SMTP settings if using email features")
    print("4. Never commit .env files to version control!")
    print("5. Store SECRET_KEY securely in production environment")


def main():
    """Main function."""
    print("=== COMPETITIVE INTELLIGENCE V2 - SECURITY SETUP ===")
    print()
    
    print("Available options:")
    print("1. Generate a single SECRET_KEY")
    print("2. Generate complete secure .env file from template")
    print("3. Show security best practices")
    print()
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        key = generate_secret_key(64)
        print(f"\nGenerated SECRET_KEY:")
        print(key)
        print(f"\nKey length: {len(key)} characters")
        print("Add this to your .env file as: SECRET_KEY=" + key)
    
    elif choice == "2":
        generate_secure_env()
    
    elif choice == "3":
        print("\n=== SECURITY BEST PRACTICES ===")
        print("1. Use strong, randomly generated SECRET_KEY (64+ characters)")
        print("2. Set ENVIRONMENT=production in production")
        print("3. Use secure database credentials")
        print("4. Configure HTTPS/TLS for all connections")
        print("5. Set appropriate CORS origins")
        print("6. Enable rate limiting")
        print("7. Use strong bcrypt rounds (12+)")
        print("8. Store secrets in environment variables, not code")
        print("9. Regular security audits and updates")
        print("10. Monitor authentication logs")
    
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()