# Security Setup Guide

## Quick Start - Production Deployment

### 1. Generate Secure Environment Configuration

**Option A: Automated Setup**
```bash
cd competitive-intel-v2
python scripts/generate_keys.py
```

**Option B: Manual Setup**
```bash
# Copy template
cp .env.template .env

# Generate secure key
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(64))"

# Add the generated key to your .env file
```

### 2. Essential Security Configuration

Update your `.env` file with these critical settings:

```env
# CRITICAL: Use generated secure key
SECRET_KEY=your-generated-64-character-key-here

# Set production environment
ENVIRONMENT=production

# Update database with secure credentials
DATABASE_URL=postgresql+asyncpg://secure_user:secure_password@db_host:5432/db_name

# Configure allowed origins for your frontend
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### 3. Environment Variable Security

**Production Deployment Methods:**

**Docker/Container:**
```dockerfile
ENV SECRET_KEY=your-secure-key-here
ENV ENVIRONMENT=production
```

**System Environment:**
```bash
export SECRET_KEY="your-secure-key-here"
export ENVIRONMENT="production"
```

**Cloud Platforms:**
- AWS: Use Parameter Store or Secrets Manager
- Google Cloud: Use Secret Manager
- Azure: Use Key Vault
- Heroku: Use Config Vars

## Security Requirements Checklist

### Critical Requirements (MUST HAVE)
- [ ] Set secure SECRET_KEY (64+ characters, cryptographically random)
- [ ] Set ENVIRONMENT=production
- [ ] Use secure database credentials
- [ ] Configure HTTPS/TLS encryption
- [ ] Set appropriate CORS_ORIGINS
- [ ] Enable rate limiting (RATE_LIMIT_ENABLED=true)

### Recommended Security (SHOULD HAVE)
- [ ] Use high bcrypt rounds (BCRYPT_ROUNDS=12+)
- [ ] Configure secure session settings
- [ ] Set up email for password reset (SMTP settings)
- [ ] Enable comprehensive logging
- [ ] Configure security headers
- [ ] Set up monitoring/alerting

### Advanced Security (NICE TO HAVE)  
- [ ] Use external key management service
- [ ] Implement API key rotation
- [ ] Set up intrusion detection
- [ ] Configure backup/recovery
- [ ] Implement audit logging
- [ ] Use container security scanning

## Key Generation Details

### SECRET_KEY Requirements
- **Minimum length**: 32 characters (enforced)
- **Recommended length**: 64+ characters
- **Type**: Cryptographically secure random
- **Encoding**: URL-safe base64

### Generation Methods

**Python (recommended):**
```python
import secrets
secret_key = secrets.token_urlsafe(64)
print(f"SECRET_KEY={secret_key}")
```

**OpenSSL:**
```bash
openssl rand -base64 64 | tr -d "=+/" | cut -c1-64
```

**Node.js:**
```javascript
const crypto = require('crypto');
const key = crypto.randomBytes(48).toString('base64url');
console.log(`SECRET_KEY=${key}`);
```

## Common Security Issues and Fixes

### Issue: "JWT secret key is insecure"
**Problem**: Using default or weak SECRET_KEY
**Fix**: Generate secure key with `secrets.token_urlsafe(64)`

### Issue: Development key in production
**Problem**: Default key detected in production environment
**Fix**: Set ENVIRONMENT=production and secure SECRET_KEY

### Issue: Key too short
**Problem**: SECRET_KEY less than 32 characters
**Fix**: Generate longer key (64+ characters recommended)

## Environment-Specific Configurations

### Development
```env
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=auto-generated-secure-key
DATABASE_URL=postgresql+asyncpg://dev_user:dev_pass@localhost:5432/dev_db
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

### Staging
```env
ENVIRONMENT=staging  
DEBUG=false
SECRET_KEY=staging-specific-secure-key
DATABASE_URL=postgresql+asyncpg://staging_user:staging_pass@staging_db:5432/staging_db
CORS_ORIGINS=https://staging.yourdomain.com
```

### Production
```env
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=production-secure-key-64-characters-minimum
DATABASE_URL=postgresql+asyncpg://prod_user:secure_password@prod_db:5432/prod_db
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

## Security Validation

After setup, validate your configuration:

```bash
# Run comprehensive security validation
python scripts/comprehensive_qa.py

# Expected output: "JWT Secret Key: PASS - Using secure key"
```

## Best Practices Summary

1. **Never hardcode secrets** - Always use environment variables
2. **Use strong keys** - 64+ character cryptographically random keys
3. **Environment separation** - Different keys for dev/staging/prod
4. **Secure storage** - Use cloud key management services
5. **Regular rotation** - Change keys periodically
6. **Monitor access** - Log and alert on authentication events
7. **HTTPS everywhere** - Encrypt all communications
8. **Principle of least privilege** - Minimal required permissions
9. **Regular updates** - Keep dependencies updated
10. **Security testing** - Regular penetration testing and audits

## Emergency Key Rotation

If you suspect key compromise:

1. **Generate new key immediately**
2. **Update all deployment environments**
3. **Revoke all existing sessions**
4. **Force user re-authentication**
5. **Audit access logs**
6. **Monitor for suspicious activity**

## Support and Contact

For security questions or issues:
- Review this guide thoroughly
- Run the QA validation script
- Check application logs for security warnings
- Follow the principle of least privilege