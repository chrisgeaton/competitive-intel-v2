# Competitive Intelligence Platform

A competitive intelligence platform with real-time content discovery, AI-powered analysis, and automated strategic report delivery. Features JWT authentication, RSS monitoring, OpenAI GPT-4 integration, and email reporting via SendGrid.

## 🚀 Quick Start

### Start the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Start development server
python app/main.py
```

**Access Points:**
- **API Base**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Test the System
```bash
# Run comprehensive QA validation
python scripts/comprehensive_qa.py

# Test API endpoints
python scripts/test_api_endpoints.py

# Send real test email
python send_real_email.py
```

## 🎯 Features

### Core Intelligence Platform
- **📡 Real Content Discovery** - Automated RSS monitoring from TechCrunch AI, Hacker News, AI News
- **🧠 AI-Powered Analysis** - OpenAI GPT-4 integration for strategic insights and relevance scoring
- **📧 Email Delivery** - Professional strategic intelligence reports via SendGrid
- **🎯 Strategic Profiles** - Personalized user context for targeted analysis and content curation
- **📊 Focus Areas** - Industry-specific monitoring with entity tracking and keyword matching
- **🔄 End-to-End Orchestration** - Complete Discovery → Analysis → Reports → Delivery automation

### Enterprise Security & Management
- **🔐 JWT Authentication** - Secure token-based authentication with refresh tokens
- **👤 User Management** - Complete profile management and preferences system
- **🛡️ Enterprise Security** - Rate limiting, security headers, and validation
- **💼 Session Management** - Multi-device session handling
- **📚 Interactive Documentation** - Swagger UI with comprehensive API docs
- **⚡ Performance Optimized** - Async operations with database connection pooling

### Operational Status ✅
- **✅ End-to-End Testing** - Full pipeline validation with real strategic intelligence delivery
- **📈 21 Real Articles** - Successfully fetching and processing AI/ML industry content from RSS feeds
- **🔄 Pipeline Tested** - Complete automation from content discovery through email delivery
- **💰 OpenAI Integration** - Real GPT-4 analysis processing actual RSS content
- **📡 SendGrid Integration** - Professional email delivery with tracking

## 🏗️ Architecture

```
app/
├── main.py                 # FastAPI application initialization
├── routers/               # API endpoint routers
│   ├── auth.py           # Authentication (/api/v1/auth)
│   ├── users.py          # User management (/api/v1/users)
│   ├── strategic_profile.py # Strategic profiles (/api/v1/strategic-profile)
│   ├── focus_areas.py    # Focus areas (/api/v1/users/focus-areas)
│   ├── discovery.py      # Content discovery (/api/v1/discovery)
│   ├── analysis.py       # AI analysis (/api/v1/analysis)
│   ├── reports.py        # Report generation (/api/v1/reports)
│   └── orchestration.py  # End-to-end orchestration (/api/v1/orchestration)
├── services/             # Business logic services
│   ├── discovery_service.py    # RSS content discovery with real fetching
│   ├── analysis_service.py     # OpenAI GPT-4 strategic analysis
│   ├── sendgrid_service.py     # Professional email delivery
│   └── orchestration_service.py # Complete pipeline coordination
├── models/               # SQLAlchemy database models
├── schemas/              # Pydantic request/response models
├── middleware.py         # Security and authentication middleware
├── auth.py              # Authentication services and JWT handling
├── database.py          # Database connection and session management
└── config.py            # Configuration and environment settings
```

## 📚 Documentation

Complete documentation is organized in the [`docs/`](./docs/) directory:

### 🎯 Essential Reading
- **[API Documentation](./docs/api/API_DOCUMENTATION.md)** - Complete endpoint guide with examples
- **[Security Setup](./docs/security/SECURITY_SETUP.md)** - Production security configuration
- **[Project Status](./docs/project-status.md)** - Current status and operational capabilities
- **[Phase 5 Testing Report](./docs/PHASE_5_END_TO_END_TESTING_REPORT.md)** - Complete end-to-end validation results

### 🛠️ Development
- **[FastAPI Implementation](./docs/development/FASTAPI_IMPLEMENTATION_SUMMARY.md)** - Implementation details
- **[Build Plan](./docs/build-plan.md)** - Development roadmap and milestones
- **[Architecture Decisions](./docs/architecture-decisions.md)** - Technical decisions and rationale

### 📊 Reports & Analysis
- **[Optimization Report](./docs/reports/optimization_report.md)** - Performance and cleanup metrics
- **[Security Fix Report](./docs/security/JWT_SECURITY_FIX_REPORT.md)** - Security vulnerability resolution

## 🔒 Security

### Production Setup
```bash
# Generate secure keys
python scripts/generate_keys.py

# Set environment variables
export SECRET_KEY="your-64-character-secure-key"
export ENVIRONMENT="production"
export DATABASE_URL="postgresql+asyncpg://user:pass@host/db"
export OPENAI_API_KEY="your-openai-api-key"
export SENDGRID_API_KEY="your-sendgrid-api-key"
```

### Security Features
- **JWT Security**: 64-character secure keys with automatic validation
- **Password Strength**: Comprehensive validation with special characters
- **Rate Limiting**: 60 requests/minute with login attempt limits
- **Security Headers**: Complete OWASP recommended headers
- **Session Management**: Secure multi-device session handling

## ✅ Quality Assurance

**System Status**: ✅ **100% QA Success Rate** (35/35 tests passing)
**API Status**: ✅ **100% Endpoint Tests** (14/14 tests passing)
**Phase 5 Status**: ✅ **Complete End-to-End Success** - Real strategic intelligence delivered

### Validation Coverage
- ✅ Real content discovery from RSS feeds (21 AI/ML articles processed)
- ✅ OpenAI GPT-4 analysis integration with strategic insights
- ✅ Professional email delivery via SendGrid
- ✅ Database operations and integrity
- ✅ Authentication security and functionality
- ✅ JWT token validation and refresh
- ✅ API endpoint functionality
- ✅ Security vulnerability scanning
- ✅ Performance benchmarking
- ✅ Error handling robustness

## 🛠️ Development

### Environment Setup
```bash
# Clone repository
git clone https://github.com/chrisgeaton/competitive-intel-v2.git
cd competitive-intel-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.template .env
# Edit .env with your configuration including:
# - DATABASE_URL
# - OPENAI_API_KEY
# - SENDGRID_API_KEY
# - JWT SECRET_KEY
```

### Database Setup
- **PostgreSQL**: localhost:5432
- **Database**: competitive_intelligence
- **User**: admin / **Password**: [set in .env]
- **Auto-creation**: Tables created automatically on first run

### Development Tools
- **Auto-reload**: `uvicorn app.main:app --reload`
- **Testing**: `python scripts/comprehensive_qa.py`
- **Key Generation**: `python scripts/generate_keys.py`
- **API Testing**: `python scripts/test_api_endpoints.py`
- **Email Testing**: `python send_real_email.py`

## 🌐 API Endpoints

### Authentication (`/api/v1/auth`)
- `POST /register` - User registration
- `POST /login` - JWT authentication
- `POST /logout` - Session termination
- `POST /refresh` - Token refresh

### User Management (`/api/v1/users`)
- `GET /profile` - Complete user profile
- `PUT /profile` - Update basic profile
- `POST /change-password` - Secure password change
- `DELETE /account` - Account deletion

### Strategic Intelligence (`/api/v1/strategic-profile`)
- `POST /` - Create strategic profile
- `GET /` - Get strategic profile
- `PUT /` - Update strategic profile

### Content Discovery (`/api/v1/discovery`)
- `POST /jobs` - Start content discovery job
- `GET /jobs/{job_id}` - Get job status
- `GET /content` - Get discovered content

### AI Analysis (`/api/v1/analysis`)
- `POST /batches` - Create analysis batch
- `POST /batches/{batch_id}/analyze` - Perform deep analysis
- `GET /batches/{batch_id}/results` - Get analysis results

### Reports (`/api/v1/reports`)
- `POST /generate` - Generate strategic intelligence report
- `GET /{report_id}` - Get report details
- `POST /{report_id}/email/send` - Send report via email

### Orchestration (`/api/v1/orchestration`)
- `POST /pipelines` - Execute complete intelligence pipeline
- `GET /pipelines/{pipeline_id}` - Get pipeline status

See [API Documentation](./docs/api/API_DOCUMENTATION.md) for complete details with examples.

## 🤝 Contributing

1. **Read Documentation**: Start with [`docs/build-plan.md`](./docs/build-plan.md)
2. **Follow Security**: Review [`docs/security/SECURITY_SETUP.md`](./docs/security/SECURITY_SETUP.md)
3. **Test Changes**: Run `python scripts/comprehensive_qa.py`
4. **Maintain Quality**: Ensure 100% test success rate
5. **End-to-End Testing**: Use `python send_real_email.py` to validate email delivery

## 📄 License

MIT License - See LICENSE file for details

## 🔗 Links

- **GitHub**: https://github.com/chrisgeaton/competitive-intel-v2
- **Documentation**: [`docs/`](./docs/)
- **API Docs**: http://localhost:8000/docs (when running)
- **Issues**: https://github.com/chrisgeaton/competitive-intel-v2/issues
- **Phase 5 Report**: [docs/PHASE_5_END_TO_END_TESTING_REPORT.md](./docs/PHASE_5_END_TO_END_TESTING_REPORT.md)
